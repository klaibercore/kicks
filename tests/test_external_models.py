"""Optional upstream adapters stay offline, keep native audio and record conversions."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kicks.external.worker import conform,load_provider,validate_job
from kicks.synthesis.external import external_generate


def test_conversion_is_explicit_and_keeps_native_contract_separate():
    t=np.arange(16000)/16000
    native=np.stack([np.sin(2*np.pi*500*t),np.sin(2*np.pi*700*t)]).astype('float32')
    x,record=conform(native,16000)
    assert x.shape==(65536,) and abs(x).max()<=.900001
    assert record['native_sample_rate']==16000 and record['native_channels']==2
    assert record['native_samples']==16000 and 'mono_mean' in record['conversion']
    with pytest.raises(ValueError):conform(np.array([[float('nan')]]),44100)


def test_job_validation_never_resolves_missing_assets_to_working_directory(tmp_path):
    with pytest.raises(FileNotFoundError):validate_job({'version':1,'kind':'rave'})
    path=tmp_path/'model.pt';path.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='prompt'):
        validate_job({'version':1,'kind':'audiogen','checkpoint':str(path)})
    with pytest.raises(ValueError,match='seed range'):
        validate_job({'version':1,'kind':'rave','checkpoint':str(path),'seed':2**32-1,'count':2})


def test_stable_audio_adapter_uses_verified_local_config_and_correct_sampler(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model.ckpt';torch.save({'state_dict':{}},checkpoint)
    config=tmp_path/'model.json';config.write_text(json.dumps({'sample_rate':44100,'sample_size':100000}))
    class Model:
        pretransform=None
        def load_state_dict(self,state,strict): assert state=={} and strict
        def eval(self):return self
        def to(self,device):return self
    calls=[]
    def generate(model,**kwargs):calls.append(kwargs);return torch.ones(1,2,1024)
    monkeypatch.setitem(sys.modules,'stable_audio_tools.models.factory',SimpleNamespace(create_model_from_config=lambda c:Model()))
    monkeypatch.setitem(sys.modules,'stable_audio_tools.inference.generation',SimpleNamespace(generate_diffusion_cond=generate,generate_diffusion_cond_inpaint=generate))
    for kind,sampler in [('stable-audio-3','pingpong'),('stable-audio-open-small','dpmpp-3m-sde')]:
        render,rate,_=load_provider({'kind':kind,'checkpoint':str(checkpoint),'config':str(config),'prompt':'One isolated snare','duration':1})
        assert render(23).shape==(2,1024) and rate==44100
        assert calls[-1]['seed']==23 and calls[-1]['sampler_type']==sampler


def test_supervisor_retains_failed_worker_diagnostics_without_global_offline_changes(tmp_path,monkeypatch):
    checkpoint=tmp_path/'rave.ts';checkpoint.write_bytes(b'not a torchscript model')
    spec=tmp_path/'job.json';spec.write_text(json.dumps({'version':1,'kind':'rave','checkpoint':str(checkpoint)}))
    monkeypatch.delenv('HF_HUB_OFFLINE',raising=False)
    with pytest.raises(RuntimeError,match='diagnostics retained'):
        external_generate(spec,out=tmp_path/'failed',python=sys.executable)
    assert 'HF_HUB_OFFLINE' not in os.environ
    execution=json.loads((tmp_path/'failed/execution.json').read_text())
    assert execution['returncode']!=0 and execution['model_identity'] and execution['stderr']


def test_rave_adapter_uses_exported_prior_and_refuses_a_missing_prior(tmp_path,monkeypatch):
    path=tmp_path/'rave.ts';path.write_bytes(b'fixture')
    enabled={'prior':True}
    class Model:
        sr=48000
        decode_params=torch.tensor([4,512,1,1])
        def __init__(self):self.calls=0
        def eval(self):return self
        def prior(self,temperature):
            self.calls+=1
            return (temperature+self.calls).repeat(1,4,1) if enabled['prior'] else torch.tensor(0)
        def decode(self,z):return z.mean(1,keepdim=True).repeat_interleave(512,-1)
    monkeypatch.setattr(torch.jit,'load',lambda *a,**kw:Model())
    render,rate,_=load_provider({'kind':'rave','checkpoint':str(path),'duration':1})
    assert rate==48000 and render(0).shape==(1,48128)
    torch.testing.assert_close(render(7),render(7),rtol=0,atol=0)
    enabled['prior']=False
    with pytest.raises(ValueError,match='no trained prior'):render(0)


@pytest.mark.parametrize('kind',['audiogen','musicgen'])
def test_audiocraft_adapter_uses_local_assets_and_explicit_seed(kind,tmp_path,monkeypatch):
    calls=[]
    class Model:
        sample_rate=32000
        @staticmethod
        def get_pretrained(path,device):calls.append(path);return Model()
        def set_generation_params(self,**kw):assert kw['duration']==1
        def generate(self,prompts,progress):
            assert prompts==['one isolated drum'] and not progress
            return torch.randn(1,1,100)
    monkeypatch.setitem(sys.modules,'audiocraft.models',SimpleNamespace(AudioGen=Model,MusicGen=Model))
    render,rate,_=load_provider({'kind':kind,'checkpoint':str(tmp_path),'prompt':'one isolated drum','duration':1})
    torch.testing.assert_close(render(9),render(9),rtol=0,atol=0)
    assert rate==32000 and calls==[str(tmp_path.resolve())]
