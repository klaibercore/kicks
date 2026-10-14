"""Strict checkpoint loading, deterministic rendering and common audio audits."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from ..analysis.benchmark import audit_directory, file_hash, write_json
from ..audio.constants import AUDIO_LENGTH, SAMPLE_RATE
from ..instruments import get_profile
from ..nn.ddsp import finish_audio
from ..nn.research import create_model
from ..training.research import CHECKPOINT_KIND, load_codec
from .procedural import render as render_dsp


class ResearchEngine:
    def __init__(self,checkpoint,instrument=None,*,device='cpu',codec_checkpoint=None):
        self.path=Path(checkpoint).resolve()
        meta=torch.load(self.path,map_location='cpu',weights_only=True)
        self.profile=get_profile(instrument or meta.get('instrument'))
        if meta.get('model_kind')!=CHECKPOINT_KIND or meta.get('instrument')!=self.profile.name:
            raise ValueError('research checkpoint kind/instrument mismatch')
        if meta.get('sample_rate')!=SAMPLE_RATE or meta.get('length')!=AUDIO_LENGTH or meta.get('descriptors')!=self.profile.descriptor_keys:
            raise ValueError('research checkpoint signal/descriptor contract mismatch')
        if int(meta.get('steps_trained',0))<1:
            raise ValueError('untrained checkpoints cannot be served')
        self.kind=meta['kind'];self.metadata=meta;self.device=torch.device(device)
        self.model=create_model(self.kind,self.profile.name,meta['architecture']).to(device)
        self.model.load_state_dict(meta['model']);self.model.eval()
        bank=meta.get('label_bank')
        if bank is None or bank.ndim!=2 or bank.shape[1]!=self.model.n_descriptors or not len(bank) or not torch.isfinite(bank).all():
            raise ValueError('checkpoint has no valid training label bank')
        self.model.label_bank=bank.cpu()
        self.codec=None
        if self.kind=='codec':
            contract=meta.get('codec') or {}
            path=codec_checkpoint or contract.get('path')
            if not path or file_hash(path)!=contract.get('sha256'):
                raise ValueError('matched codec identity mismatch')
            self.codec=load_codec(path,device)
        self.sha256=file_hash(self.path)

    @torch.inference_mode()
    def render(self,positions=None,*,targets=None,seed=0,steps=25):
        from .diffusion import slider_targets
        if not isinstance(seed,int) or not 0<=seed<2**32: raise ValueError('invalid seed')
        if targets is None:
            targets=slider_targets(self.model,positions if positions is not None else [.5]*self.model.n_descriptors)
        labels=torch.as_tensor(targets,dtype=torch.float32,device=self.device).reshape(1,-1)
        if labels.shape[1]!=self.model.n_descriptors or not torch.isfinite(labels).all():
            raise ValueError('invalid descriptor targets')
        if self.kind=='parameters':
            params=self.profile.procedural.from_positions(self.model.sample(labels,seed)[0].cpu().tolist())
            audio=render_dsp(self.profile,params,seed=seed)
        elif self.kind=='codec':
            z=self.model.sample(labels,seed,steps)
            audio=finish_audio(self.codec.decode(z)[...,:AUDIO_LENGTH])[0,0].cpu().numpy()
        else:
            audio=self.model.sample(labels,seed)[0,0].cpu().numpy()
        if audio.shape!=(AUDIO_LENGTH,) or not np.isfinite(audio).all():
            raise RuntimeError('model violated the final-waveform contract')
        return audio.astype(np.float32),labels[0].cpu().numpy()


def generate_research(checkpoint,*,out,instrument=None,count=8,seed=0,steps=25,targets=None,codec_checkpoint=None,conditions=None):
    directory=Path(out)
    if directory.exists() or count<1 or not 0<=seed<2**32 or seed+count>2**32:
        raise ValueError('use a new output directory, a positive count and valid seed range')
    engine=ResearchEngine(checkpoint,instrument,codec_checkpoint=codec_checkpoint)
    if targets is not None:
        if set(targets)-set(engine.profile.descriptor_keys): raise ValueError('unknown descriptor target')
    if conditions is not None:
        if not isinstance(conditions,list) or not conditions or any(
            not isinstance(row,dict) or set(row)!=set(engine.profile.descriptor_keys) or not all(np.isfinite(v) for v in row.values())
            for row in conditions): raise ValueError('conditions must be a nonempty list of complete finite raw descriptor rows')
    directory.mkdir(parents=True);(directory/'native').mkdir()
    target_rows={};records=[]
    for i in range(count):
        row=engine.model.label_bank[i%len(engine.model.label_bank)].clone() if conditions is None else torch.tensor([
            conditions[i%len(conditions)][k] for k in engine.profile.descriptor_keys])
        for key,value in (targets or {}).items(): row[engine.profile.descriptor_keys.index(key)]=value
        start=time.perf_counter();audio,label=engine.render(targets=row,seed=seed+i,steps=steps)
        name=f'case_{i:03d}_seed_{seed+i}.wav'
        sf.write(directory/'native'/name,audio,SAMPLE_RATE,subtype='PCM_24')
        target_rows[name]=dict(zip(engine.profile.descriptor_keys,map(float,label)))
        records.append({'file':name,'seed':seed+i,'steps':steps,'render_ms':(time.perf_counter()-start)*1000})
    write_json(directory/'targets.json',target_rows)
    report=audit_directory(directory/'native',engine.profile.name,out=directory/'audit',backend=engine.kind,targets=directory/'targets.json')
    write_json(directory/'generation.json',{'checkpoint_sha256':engine.sha256,'kind':engine.kind,
               'manifest_sha256':engine.metadata['manifest_sha256'],'codec':engine.metadata.get('codec'),
               'experimental':True,'listening_judged':False,'renders':records,'summary':report['summary']})
    return report
