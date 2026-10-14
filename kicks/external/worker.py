"""Local-only upstream inference worker, also executable in a separate venv.

Optional packages are imported only for the requested model family. Cached
text encoders and codec dependencies must also be present: HF offline mode is
set before importing any provider, so this worker cannot accept terms or fetch
new model assets as a hidden side effect.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

# Version-1 wire contract. Keep this worker independent of the main package so
# old upstream Python environments can run it. The parent audits both values.
AUDIO_LENGTH,SAMPLE_RATE=65536,44100


def resample(audio,source_rate,target_rate):
    if source_rate==target_rate:return audio
    divisor=math.gcd(int(source_rate),int(target_rate))
    return resample_poly(audio,target_rate//divisor,source_rate//divisor)


def fit_length(audio,length):
    return np.pad(audio,(0,max(0,length-len(audio))))[:length]

KINDS=('stable-audio-3','stable-audio-open-small','rave','audiogen','musicgen')


def validate_job(job):
    if job.get('version')!=1 or job.get('kind') not in KINDS:
        raise ValueError('unknown external job kind/version')
    if not isinstance(job.get('checkpoint'),str) or not job['checkpoint'] or not Path(job['checkpoint']).exists():
        raise FileNotFoundError('external job requires a local checkpoint or model directory')
    if not isinstance(job.get('seed',0),int) or not 0<=job.get('seed',0)<2**32:
        raise ValueError('invalid external seed')
    if not isinstance(job.get('count',1),int) or not 1<=job.get('count',1)<=32:
        raise ValueError('external count must be 1..32')
    if not isinstance(job.get('duration',2), (int,float)) or not 0<job.get('duration',2)<=10:
        raise ValueError('external duration must be positive and at most ten seconds')
    if job.get('seed',0)+job.get('count',1)>2**32:
        raise ValueError('external seed range exceeds uint32')
    if job.get('device','cpu') not in ('cpu','cuda'):
        raise ValueError('external device must be cpu or cuda')
    if job['kind'].startswith('stable-audio') and not Path(job.get('config','')).is_file():
        raise ValueError('Stable Audio requires a local model configuration')
    if job['kind']!='rave' and (not isinstance(job.get('prompt'),str) or not job['prompt'].strip()):
        raise ValueError('text-conditioned models require a prompt')
    steps=job.get('steps',8 if job['kind']=='stable-audio-3' else 100)
    if not isinstance(steps,int) or not 1<=steps<=250:
        raise ValueError('external steps must be 1..250')


def load_provider(job):
    kind=job['kind'];path=str(Path(job['checkpoint']).resolve());device=job.get('device','cpu')
    if kind.startswith('stable-audio'):
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.inference.generation import generate_diffusion_cond,generate_diffusion_cond_inpaint
        config=json.loads(Path(job['config']).read_text())
        model=create_model_from_config(config)
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state=load_file(path)
        else:
            state=torch.load(path,map_location='cpu',weights_only=True)
            state=state.get('state_dict',state)
        model.load_state_dict(state,strict=True);model.eval().to(device)
        rate=int(config['sample_rate'])
        size=config['sample_size']
        generate=generate_diffusion_cond_inpaint if kind=='stable-audio-3' else generate_diffusion_cond
        def render(seed):
            conditioning={'prompt':job['prompt'],'seconds_total':job.get('duration',2)}
            if kind!='stable-audio-3': conditioning['seconds_start']=0
            return generate(model,conditioning=[conditioning],sample_size=size,seed=seed,device=device,
                steps=job.get('steps',8 if kind=='stable-audio-3' else 100),
                cfg_scale=job.get('guidance',1. if kind=='stable-audio-3' else 6.),
                duration_padding_sec=.25,adapt_duration_to_conditioning=kind=='stable-audio-3',
                sampler_type='pingpong' if kind=='stable-audio-3' else 'dpmpp-3m-sde')[0]
        return render,rate,'stable-audio-tools'
    if kind in ('audiogen','musicgen'):
        from audiocraft.models import AudioGen,MusicGen
        model=(AudioGen if kind=='audiogen' else MusicGen).get_pretrained(path,device=device)
        model.set_generation_params(duration=job.get('duration',2))
        def render(seed):
            torch.manual_seed(seed)
            return model.generate([job['prompt']],progress=False)[0]
        return render,int(model.sample_rate),'audiocraft'
    model=torch.jit.load(path,map_location=device).eval()
    rate=int(getattr(model,'sr',job.get('sample_rate',0)))
    if rate<8000: raise ValueError('RAVE requires its native sample_rate when the export has no sr attribute')
    def render(seed):
        torch.manual_seed(seed)
        # Streaming exports may retain prior/convolution state. Reload for
        # independent one-shots so previous hits cannot change seed replay.
        current=torch.jit.load(path,map_location=device).eval()
        if job.get('source'):
            audio,sr=sf.read(job['source'],always_2d=True,dtype='float32')
            audio=np.stack([resample(audio[:,i],sr,rate) for i in range(audio.shape[1])])
            return current(torch.from_numpy(audio)[None].to(device))[0]
        if not hasattr(current,'prior') or not hasattr(current,'decode_params'):
            raise ValueError('RAVE generation requires a TorchScript export with its trained prior')
        ratio=int(current.decode_params[1])
        frames=math.ceil(job.get('duration',2)*rate/ratio)
        latent=current.prior(torch.full((1,1,frames),float(job.get('temperature',1.)),device=device))
        if latent.ndim!=3 or not latent.numel():
            raise ValueError('RAVE export has no trained prior; refusing random-latent generation')
        return current.decode(latent)[0]
    return render,rate,'acids-rave'


def conform(audio,rate):
    """Explicit mono/resample/onset-crop adapter, with unmodified native output retained."""
    array=np.asarray(audio,dtype=np.float32)
    if array.ndim==1: array=array[None]
    if array.ndim!=2 or not array.shape[1] or not np.isfinite(array).all():
        raise ValueError('provider returned invalid audio')
    mono=resample(array.mean(0),rate,SAMPLE_RATE)
    peak=float(np.max(np.abs(mono)))
    active=np.flatnonzero(abs(mono)>peak*.02) if peak>1e-7 else np.array([],dtype=int)
    onset=max(0,int(active[0])-8) if len(active) else 0
    x=fit_length(mono[onset:],AUDIO_LENGTH)
    x=x*min(1,.9/max(float(np.max(abs(x))),1e-12))
    x[:8]*=np.linspace(0,1,8);x[-220:]*=np.linspace(1,0,220)
    return x.astype(np.float32),{'native_sample_rate':rate,'native_channels':array.shape[0],
        'native_samples':array.shape[1],'native_peak':float(abs(array).max()),
        'onset_offset_samples_44100':onset,'conversion':'mono_mean; polyphase_resample; first_onset_crop; bounded_peak; fades',
        'silent':peak<1e-7,'multiple_events':'not automatically judged; inspect native recording'}


def run_worker(job):
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    validate_job(job)
    out=Path(job['out'])
    (out/'native').mkdir();(out/'conformed').mkdir()
    render,rate,package=load_provider(job)
    records=[]
    with torch.inference_mode():
        for i in range(job.get('count',1)):
            seed=job.get('seed',0)+i
            started=time.perf_counter()
            native=render(seed).detach().float().cpu().numpy()
            elapsed=(time.perf_counter()-started)*1000
            if native.ndim==1:native=native[None]
            audio,conversion=conform(native,rate)
            name=f'case_{i:03d}.wav'
            sf.write(out/'native'/name,native.T,rate,subtype='FLOAT')
            sf.write(out/'conformed'/name,audio,SAMPLE_RATE,subtype='PCM_24')
            records.append({'file':name,'seed':seed,'render_ms':elapsed,**conversion})
    try: version=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:version='unknown'
    (out/'worker-report.json').write_text(json.dumps({'package':package,'version':version,
        'runtime':{'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':np.__version__},
        'prompt_conditioning_only':job['kind']!='rave','renders':records},indent=2)+'\n')


if __name__=='__main__':
    try:
        run_worker(json.loads(Path(sys.argv[1]).read_text()))
    except ImportError as exc:
        raise SystemExit(f'Optional model environment is incomplete: {exc}. Install its isolated requirements.')
