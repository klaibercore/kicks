"""New neural engines preserve preview/export/kit identities and final audio."""
import importlib
import io

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from kicks.api.state import ServerState,parse_research_checkpoints
from kicks.instruments import get_profile
from kicks.nn.research import create_model
from kicks.training.research import CHECKPOINT_KIND

api=importlib.import_module('kicks.api.app')


@pytest.fixture
def client(tmp_path,monkeypatch):
    profile=get_profile('hihat');model=create_model('hybrid','hihat')
    bank=torch.tensor([[1.,2.,3.,4.,20.],[2.,3.,4.,5.,120.],[3.,4.,5.,6.,320.]])
    model.set_labels(bank)
    path=tmp_path/'hybrid.pth'
    torch.save({'model_kind':CHECKPOINT_KIND,'kind':'hybrid','instrument':'hihat',
        'sample_rate':44100,'length':65536,'architecture':{},'descriptors':profile.descriptor_keys,
        'model':model.state_dict(),'label_bank':bank,'epoch':1,'steps_trained':1,
        'training':{'run_id':'fixture'},'manifest_sha256':'fixture'},path)
    monkeypatch.setenv('KICKS_MODEL_DIR',str(tmp_path/'absent'))
    monkeypatch.setenv('KICKS_DATA_DIR',str(tmp_path/'data'))
    monkeypatch.setenv('KICKS_BACKEND','dsp')
    monkeypatch.setenv('KICKS_RESEARCH',f'hihat:hybrid={path}')
    monkeypatch.delenv('KICKS_DIFFUSION',raising=False)
    monkeypatch.setattr(api,'state',ServerState())
    api.audio_cache.clear();api.render_cache.clear()
    api.app.dependency_overrides[api.rate_limiter]=lambda:None
    with TestClient(api.app) as client:yield client
    api.app.dependency_overrides.clear();api.audio_cache.clear();api.render_cache.clear()


def test_candidate_discovery_config_and_preview_export(client):
    info=next(i for i in client.get('/instruments').json()['instruments'] if i['name']=='hihat')
    assert 'hybrid' in info['backends'] and 'codec' not in info['backends']
    config=client.get('/config?instrument=hihat&backend=hybrid').json()
    assert config['backend']=='hybrid' and config['checkpoint']['sha256'] and config['device']=='cpu'
    query='instrument=hihat&backend=hybrid&seed=20'
    a=client.get('/generate?'+query);b=client.post('/export?'+query)
    assert a.status_code==b.status_code==200 and a.content==b.content
    assert 'hihat_hybrid_' in b.headers['content-disposition']
    wave,sr=sf.read(io.BytesIO(a.content))
    assert sr==44100 and wave.shape==(65536,) and np.isfinite(wave).all()
    assert client.get('/generate?'+query+'&seed=21').content!=a.content


def test_candidate_evaluation_and_spectrogram_are_measured_from_render(client):
    query='instrument=hihat&backend=hybrid'
    report=client.get('/evaluate?'+query).json()
    assert report['control_fit']['targets'] and report['experimental'] and 'score' not in report
    spec=client.get('/spectrogram?'+query).json()
    assert spec['source']=='render' and spec['shape']==[128,256]
    assert client.get('/generate?instrument=hihat&backend=noiseband').status_code==503


def test_candidate_spec_rejects_unknown_models_and_duplicate_slots():
    with pytest.raises(ValueError):parse_research_checkpoints('hihat:bogus=a.pth')
    with pytest.raises(ValueError):parse_research_checkpoints('hihat:hybrid=a.pth,hihat:hybrid=b.pth')


def test_cli_can_serve_a_candidate_root_without_promoting_weights(tmp_path,monkeypatch):
    import os
    import uvicorn
    from typer.testing import CliRunner
    from kicks.cli import app
    root=tmp_path/'candidate root'
    path=root/'hihat-hybrid/best.pth';path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
    for key,value in {'KICKS_BACKEND':'dsp','KICKS_INSTRUMENT':'kick','KICKS_CONTROL':'descriptor',
                      'KICKS_RESEARCH':'','KICKS_VOCODER':''}.items():monkeypatch.setenv(key,value)
    calls=[];monkeypatch.setattr(uvicorn,'run',lambda *a,**kw:calls.append((a,kw)))
    result=CliRunner().invoke(app,['serve','--backend','dsp','--neural-root',str(root)])
    assert result.exit_code==0,result.output
    assert calls and os.environ['KICKS_BACKEND']=='dsp'
    assert parse_research_checkpoints(os.environ['KICKS_RESEARCH'])=={('hihat','hybrid'):str(path)}
    assert path.read_bytes()==b'fixture'
