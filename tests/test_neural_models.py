"""Neural family gradients, split protection, matched codecs and replay."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from kicks.analysis.benchmark import freeze_corpus
from kicks.instruments import get_profile
from kicks.nn.ddsp import NoiseBands,modal_body,spectral_loss
from kicks.nn.research import KINDS,create_model
from kicks.synthesis.procedural import render
from kicks.synthesis.research import ResearchEngine,generate_research
from kicks.training.research import train_research


@pytest.fixture
def manifest(tmp_path):
    root=tmp_path/'data';labels={}
    directory=root/'hihats';directory.mkdir(parents=True)
    for i in range(8):
        name=f'hit_{i}.wav'
        sf.write(directory/name,render('hihat',seed=i,parameters={'decay_ms':60+i*80}),44100)
        labels[f'hihats/{name}']={'group':f'pack_{i}','license':'fixture'}
    groups=tmp_path/'groups.json';groups.write_text(json.dumps(labels))
    path=tmp_path/'manifest.json'
    freeze_corpus(root,path,registry=tmp_path/'none',groups=groups)
    return path


@pytest.mark.parametrize('instrument',['kick','snare','hihat'])
def test_physical_body_is_differentiable_with_finite_gradients(instrument):
    positions=torch.full((2,len(get_profile(instrument).procedural.parameters)),.5,requires_grad=True)
    x=modal_body(instrument,positions,8192)
    x.square().mean().backward()
    assert torch.isfinite(positions.grad).all() and positions.grad.abs().sum()>0


def test_noise_bank_replays_phase_but_learns_time_varying_energy():
    bank=NoiseBands(8,8192,low_hz=1000)
    gains=torch.full((1,8,16),.05,requires_grad=True)
    a=bank(gains,seed=4);b=bank(gains,seed=4)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(bank(gains*.5,seed=4),a*.5)
    assert not torch.equal(bank(gains,seed=5),a)
    a.square().mean().backward()
    assert gains.grad.abs().sum()>0 and torch.isfinite(gains.grad).all()


@pytest.mark.parametrize('kind',KINDS)
def test_all_model_families_receive_gradients_and_replay(kind):
    torch.set_num_threads(2)
    architecture={'channels':8,'frames':16} if kind=='codec' else {'length':8192} if kind!='parameters' else {}
    model=create_model(kind,'hihat',architecture)
    labels=torch.randn(2,5);model.set_labels(labels)
    if kind=='parameters':
        predicted,kl=model(torch.rand(2,7),labels);loss=predicted.square().mean()+kl*.01
    elif kind=='codec':
        predicted=model(torch.randn(2,8,16),torch.tensor([.2,.6]),labels);loss=predicted.square().mean()
    elif kind=='drumgan':
        predicted=model.sample(labels,3);loss=predicted.square().mean()
    elif kind=='wavevae':
        predicted,kl=model(torch.randn(2,1,8192)*.1,labels);loss=spectral_loss(predicted,torch.randn_like(predicted)*.1)+.01*kl
    else:
        predicted,kl,_=model(torch.randn(2,1,8192)*.1,labels);loss=spectral_loss(predicted,torch.randn_like(predicted)*.1)+.01*kl
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
    model.eval()
    with torch.no_grad():
        a=model.sample(labels,seed=7);b=model.sample(labels,seed=7)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.fixture
def tiny_codec(monkeypatch,tmp_path):
    class Codec(torch.nn.Module):
        sample_rate=44100
        def preprocess(self,x,rate): return x
        def encode(self,x): return (x.unfold(-1,512,512).mean(-1).repeat(1,8,1),)
        def decode(self,z): return z.mean(1,keepdim=True).repeat_interleave(512,-1)
    monkeypatch.setattr('kicks.training.research.load_codec',lambda *a,**kw:Codec())
    monkeypatch.setattr('kicks.synthesis.research.load_codec',lambda *a,**kw:Codec())
    path=tmp_path/'matched.pth';path.write_bytes(b'test codec identity')
    return path


@pytest.mark.parametrize('kind',['hybrid','drumgan','wavevae','codec'])
def test_bounded_training_checkpoint_reload_and_audio_audit(kind,manifest,tmp_path,tiny_codec):
    out=tmp_path/kind
    result=train_research(kind,manifest,'hihat',out=out,epochs=1,max_steps=1,limit=2,validation_limit=1,
        codec_checkpoint=tiny_codec if kind=='codec' else None,
        intent='Exercise integration',hypothesis='Optimization and decoding remain finite',
        success_criteria='A bounded step and source-isolated checkpoint replay',runs_dir=tmp_path/'runs')
    assert result['steps']==1
    engine=ResearchEngine(result['checkpoint'])
    a,_=engine.render(seed=2,steps=2);b,_=engine.render(seed=2,steps=2)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(65536,) and np.isfinite(a).all() and abs(a).max()<=.900001
    trained={r['id'] for r in engine.metadata['train_sources']}
    validated={r['id'] for r in engine.metadata['validation_sources']}
    assert trained.isdisjoint(validated) and trained and validated
    report=generate_research(result['checkpoint'],out=tmp_path/'generation',count=1,steps=2)
    assert report['summary']['generated']==1 and not report['held_out']
    assert report['generation'][0]['target_error_raw'] is not None
    with pytest.raises(FileExistsError):
        train_research(kind,manifest,'hihat',out=out,intent='x',hypothesis='x',success_criteria='x')
    if kind=='codec':
        tiny_codec.write_bytes(b'changed codec')
        with pytest.raises(ValueError,match='codec identity'):
            ResearchEngine(result['checkpoint'])


def test_loader_refuses_wrong_instrument_or_signal_contract(manifest,tmp_path):
    result=train_research('noiseband',manifest,'hihat',out=tmp_path/'weights',epochs=1,max_steps=1,limit=2,validation_limit=1,
        intent='Test identity guards',hypothesis='Incompatible metadata is refused',success_criteria='No silent reinterpretation',runs_dir=tmp_path/'runs')
    with pytest.raises(ValueError,match='instrument mismatch'):
        ResearchEngine(result['checkpoint'],'kick')
    data=torch.load(result['checkpoint'],weights_only=True)
    for field,value in [('sample_rate',16000),('steps_trained',0),('descriptors',['incorrect'])]:
        modified={**data,field:value};path=tmp_path/f'{field}.pth';torch.save(modified,path)
        with pytest.raises(ValueError):ResearchEngine(path)
