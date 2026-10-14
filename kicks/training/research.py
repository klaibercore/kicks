"""Bounded, manifest-bound training for the neural/hybrid research families."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..analysis.benchmark import file_hash, identity, load_manifest
from ..audio.constants import AUDIO_LENGTH, SAMPLE_RATE
from ..data.waveforms import PREPROCESSING, label_waveform, load_hit
from ..instruments import get_profile
from ..nn.ddsp import spectral_loss
from ..nn.diffusion import diffuse, v_target
from ..nn.research import KINDS, WaveCritic, create_model
from .tracking import TrainingRun, runs_root

CHECKPOINT_KIND = 'drum_research_v1'


def load_fold(document, instrument, split, limit, seed):
    """Verify selected bytes before applying the existing neural audio chain."""
    if split not in ('train', 'validation', 'test') or limit < 1:
        raise ValueError('invalid split or limit')
    profile = get_profile(instrument)
    rows = [r for r in document['rows'] if r['instrument'] == profile.name and r['split'] == split]
    if not rows:
        raise ValueError(f'no {split} rows for {profile.name}; no split fallback')
    indices = np.random.default_rng(seed).permutation(len(rows))[:limit]
    audio, labels, selected = [], [], []
    for i in indices:
        row = rows[i]
        if file_hash(row['path']) != row['sha256']:
            raise ValueError(f'corpus identity changed: {row["id"]}')
        x = load_hit(row['path'])
        if x is None:
            raise ValueError(f'unusable waveform: {row["id"]}')
        y = torch.from_numpy(label_waveform(x, profile)).float()
        if not torch.isfinite(y).all():
            raise ValueError(f'nonfinite descriptors: {row["id"]}')
        audio.append(x); labels.append(y)
        selected.append({'id': row['id'], 'sha256': row['sha256'], 'split_group': row['split_group']})
    return torch.stack(audio), torch.stack(labels), selected


def load_codec(path, device='cpu'):
    if not Path(path).is_file():
        raise FileNotFoundError('a local matched DAC checkpoint is required')
    return _cached_codec(str(Path(path).resolve()),file_hash(path),str(device))


@lru_cache(maxsize=1)
def _cached_codec(path,digest,device):
    import dac
    codec = dac.DAC.load(str(path)).eval().to(device)
    codec.requires_grad_(False)
    if codec.sample_rate != SAMPLE_RATE:
        raise ValueError('codec requires a native 44.1 kHz DAC checkpoint')
    return codec


@torch.no_grad()
def encode_codec(codec, audio, device):
    # One hit at a time limits the frozen codec's activation memory on 8 GB Macs.
    return torch.cat([codec.encode(codec.preprocess(x[None].to(device), SAMPLE_RATE))[0].cpu()
                      for x in audio])


def bank_data(path, document, instrument):
    bank = json.loads(Path(path).read_text())
    if identity({k:v for k,v in bank.items() if k != 'sha256'}) != bank.get('sha256'):
        raise ValueError('fitted bank identity changed')
    if bank.get('split') != 'train' or bank['instrument'] != instrument or bank['manifest_sha256'] != document['sha256']:
        raise ValueError('bank must match the instrument, manifest and training split')
    training = {r['id']:r for r in document['rows'] if r['split']=='train' and r['instrument']==instrument}
    positions, labels, rows = [], [], []
    profile = get_profile(instrument)
    for fit in bank['fits']:
        row = training.get(fit['source_id'])
        if row is None or row['sha256'] != fit['source_sha256'] or file_hash(row['path']) != row['sha256']:
            raise ValueError('fit source is not verified training data')
        profile.procedural.resolve(fit['preset']['parameters'])
        positions.append([p.position(fit['preset']['parameters'][p.key]) for p in profile.procedural.parameters])
        labels.append([fit['report']['descriptors'][k] for k in profile.descriptor_keys])
        rows.append({'id':row['id'], 'sha256':row['sha256'], 'split_group':row['split_group']})
    if len(rows)<2:
        raise ValueError('at least two fitted training hits are required')
    return torch.tensor(positions), torch.tensor(labels), rows, bank['sha256']


def train_research(kind, manifest, instrument='kick', *, out, bank=None, codec_checkpoint=None,
                   epochs=2, max_steps=32, limit=32, validation_limit=8, batch_size=2,
                   learning_rate=1e-3, seed=42, device='cpu', architecture=None,
                   run_name=None, intent=None, hypothesis=None, success_criteria=None, runs_dir=None):
    """Train a candidate in a new directory; never replace the active instrument."""
    if kind not in KINDS or min(epochs,max_steps,limit,validation_limit,batch_size)<1 or not 0<learning_rate<1:
        raise ValueError('invalid model or training budget')
    if device not in ('cpu','cuda'):
        raise ValueError('research training supports cpu or cuda; FFT branches need those devices')
    if not all(isinstance(v,str) and v.strip() for v in (intent,hypothesis,success_criteria)):
        raise ValueError('declare intent, hypothesis and success criteria before training')
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'candidate output must be new: {out}')
    document = load_manifest(manifest)
    profile = get_profile(instrument)
    instrument = profile.name
    architecture = dict(architecture or {})
    codec, codec_meta, bank_sha = None, None, None
    if kind == 'parameters':
        if not bank: raise ValueError('parameters requires a verified fitted --bank')
        train_x, train_y, train_rows, bank_sha = bank_data(bank,document,instrument)
        val_x=val_y=None; val_rows=[]
    else:
        train_x,train_y,train_rows=load_fold(document,instrument,'train',limit,seed)
        val_x,val_y,val_rows=load_fold(document,instrument,'validation',validation_limit,seed+1)
    if kind == 'codec':
        if not codec_checkpoint: raise ValueError('codec prior requires a matched --codec-checkpoint')
        codec=load_codec(codec_checkpoint,device)
        train_x=encode_codec(codec,train_x,device)
        val_x=encode_codec(codec,val_x,device)
        architecture.update(channels=train_x.shape[1],frames=train_x.shape[2])
        codec_meta={'path':str(Path(codec_checkpoint).resolve()),'sha256':file_hash(codec_checkpoint),
                    'sample_rate':SAMPLE_RATE,'representation':'quantized_continuous_z',
                    'pretrained_training_overlap':'unknown'}
    torch.manual_seed(seed)
    model=create_model(kind,instrument,architecture).to(device)
    model.set_labels(train_y.to(device))
    if kind=='codec':
        model.latent_mean.copy_(train_x.mean((0,2),keepdim=True).to(device))
        model.latent_std.copy_(train_x.std((0,2),unbiased=False,keepdim=True).clamp_min(.01).to(device))
        train_x=(train_x-model.latent_mean.cpu())/model.latent_std.cpu()
        val_x=(val_x-model.latent_mean.cpu())/model.latent_std.cpu()
    metadata={'model_kind':CHECKPOINT_KIND,'kind':kind,'instrument':instrument,
              'descriptors':profile.descriptor_keys,'sample_rate':SAMPLE_RATE,'length':AUDIO_LENGTH,
              'architecture':architecture,'manifest_sha256':document['sha256'],
              'train_sources':train_rows,'validation_sources':val_rows,'bank_sha256':bank_sha,
              'codec':codec_meta,'preprocessing':PREPROCESSING,
              'experimental':True,'listening_judged':False,
              'training':{'seed':seed,'epochs':epochs,'max_steps':max_steps,'batch_size':batch_size,
                          'learning_rate':learning_rate,'intent':intent,'hypothesis':hypothesis,
                          'success_criteria':success_criteria}}
    out.mkdir(parents=True)
    objective={'parameters':'physical parameter MSE + 0.002 KL',
               'ddsp':'multi-window waveform magnitude + envelope + 0.002 KL',
               'noiseband':'multi-window waveform magnitude + envelope + 0.002 KL',
               'hybrid':'multi-window waveform magnitude + envelope + 0.002 KL',
               'wavevae':'multi-window waveform magnitude + envelope + 0.002 KL',
               'drumgan':'conditional WGAN-GP + 0.1 stochastic spectral regularization',
               'codec':'normalized matched-DAC latent angular v-prediction MSE'}[kind]
    tracking_config={**metadata,**metadata['training'],'backend':kind,'epochs':epochs,
        'model_dir':str(out.resolve()),'checkpoint':str((out/'best.pth').resolve()),
        'device':device,'optimizer':'Adam','corpus_samples':len(train_rows)+len(val_rows),
        'train_samples':len(train_rows),'val_samples':len(val_rows),
        'parameters':sum(p.numel() for p in model.parameters()),'objective':objective,
        'validation_objective':'unavailable for training-only parameter bank' if val_x is None else 'fixed-noise, fixed-posterior-RNG held-out objective',
        'split_fingerprint':identity({'manifest':document['sha256'],'train':train_rows,'validation':val_rows})}
    run=TrainingRun(runs_root(runs_dir,profile.paths.output_root),
                    tracking_config,name=run_name,
                    notes={'objective':intent,'hypothesis':hypothesis,'success_criteria':success_criteria})
    metadata['training']['run_id']=run.id
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    critic=WaveCritic(model.n_descriptors).to(device) if kind=='drumgan' else None
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=learning_rate,betas=(.5,.9)) if critic else None
    rng=torch.Generator().manual_seed(seed+3)
    step,best=0,float('inf')

    def objective(x,y,trial_seed,training=True):
        if kind=='parameters':
            pred,kl=model(x,y)
            return F.mse_loss(pred,x)+.002*kl
        if kind in ('ddsp','noiseband','hybrid'):
            rendered,kl,positions=model(x,y,trial_seed)
            # Weak regularization avoids degenerate boundary parameter fits.
            return spectral_loss(rendered,x)+.002*kl+.001*(positions-.5).square().mean()
        if kind=='wavevae':
            rendered,kl=model(x,y)
            return spectral_loss(rendered,x)+.002*kl
        if kind=='codec':
            noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(trial_seed)).to(x)
            sigmas=torch.rand(len(x),generator=torch.Generator().manual_seed(trial_seed+1)).to(x) if training else x.new_full((len(x),),.5)
            return F.mse_loss(model(diffuse(x,noise,sigmas),sigmas,y),v_target(x,noise,sigmas))
        fake=model.sample(y,trial_seed)
        if not training: return spectral_loss(fake,x)
        normalized=model.condition(y)
        critic_optimizer.zero_grad(set_to_none=True)
        alpha=torch.rand(len(x),1,1,device=x.device)
        mixed=(alpha*x+(1-alpha)*fake.detach()).requires_grad_(True)
        score=critic(mixed,normalized)
        gradient=torch.autograd.grad(score.sum(),mixed,create_graph=True)[0]
        penalty=(gradient.flatten(1).norm(2,dim=1)-1).square().mean()
        dloss=critic(fake.detach(),normalized).mean()-critic(x,normalized).mean()+10*penalty
        dloss.backward(); critic_optimizer.step()
        return -critic(fake,normalized).mean()+.1*spectral_loss(fake,x)

    def validate():
        if val_x is None: return None
        model.eval()
        losses=[]
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(seed+99)
            for i in range(0,len(val_x),batch_size):
                losses.append(float(objective(val_x[i:i+batch_size].to(device),val_y[i:i+batch_size].to(device),seed+99+i,False)))
        model.train()
        return sum(losses)/len(losses)

    def save(filename,epoch):
        temporary=out/(filename+'.tmp')
        torch.save({**metadata,'model':model.state_dict(),'label_bank':model.label_bank,
                    'epoch':epoch,'steps_trained':step,'validation_loss':best if val_x is not None and np.isfinite(best) else None,
                    'best_objective':best if np.isfinite(best) else None,'selection_split':'train' if val_x is None else 'validation',
                    'critic':critic.state_dict() if critic else None},temporary)
        temporary.replace(out/filename)

    try:
        baseline=validate()
        run.epoch({'epoch':0,'val_loss':baseline,'train_loss':None,'steps':0})
        for epoch in range(1,epochs+1):
            started=time.perf_counter(); losses=[]
            order=torch.randperm(len(train_x),generator=rng)
            for offset in range(0,len(order),batch_size):
                if step>=max_steps: break
                ids=order[offset:offset+batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss=objective(train_x[ids].to(device),train_y[ids].to(device),seed+step)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                optimizer.step();step+=1;losses.append(float(loss.detach()))
                run.progress(phase='training',epoch=epoch,batch=offset//batch_size+1,steps=step)
            val=validate()
            train_loss=sum(losses)/len(losses)
            run.epoch({'epoch':epoch,'train_loss':train_loss,'val_loss':val,'steps':step,
                       'learning_rate':learning_rate,
                       'epoch_seconds':time.perf_counter()-started})
            criterion=train_loss if val is None else val
            if criterion<best:
                best=criterion;save('best.pth',epoch)
            save('last.pth',epoch)
            if step>=max_steps: break
        run.finish()
    except BaseException as exc:
        run.finish('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',str(exc));raise
    return {'checkpoint':str(out/'best.pth'),'run_id':run.id,'steps':step,'best_objective':best}
