"""The local studio API works without accounts, a corpus or neural checkpoints."""
import importlib
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from kicks.api.state import ServerState
api = importlib.import_module('kicks.api.app')


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv('KICKS_MODEL_DIR', str(tmp_path/'models'))
    monkeypatch.setenv('KICKS_DATA_DIR', str(tmp_path/'data'))
    monkeypatch.setenv('KICKS_BACKEND', 'dsp')
    monkeypatch.delenv('KICKS_DIFFUSION', raising=False)
    monkeypatch.setattr(api, 'state', ServerState())
    api.audio_cache.clear(); api.render_cache.clear()
    api.app.dependency_overrides[api.rate_limiter] = lambda: None
    with TestClient(api.app) as client:
        yield client
    api.app.dependency_overrides.clear()
    api.audio_cache.clear(); api.render_cache.clear()


def test_no_checkpoint_startup_and_physical_configuration(client):
    health = client.get('/health').json()
    assert health['default_backend'] == 'dsp' and 'auth' not in health
    for info in client.get('/instruments').json()['instruments']:
        assert 'dsp' in info['backends'] and not info['trained']
        config = client.get(f'/config?instrument={info["name"]}&backend=dsp').json()
        assert config['control'] == 'physical' and config['presets']
        assert len(config['sliders']) >= 7
        assert all(s['key'].startswith('voice_') for s in config['sliders'])
    assert client.get('/me').status_code == 404


def test_preview_export_and_replay_are_identical_without_accounts(client):
    query = 'instrument=snare&backend=dsp&seed=8&voice_wire_ms=0.7'
    preview = client.get('/generate?' + query)
    exported = client.post('/export?' + query)
    assert preview.status_code == exported.status_code == 200
    assert preview.content == exported.content
    assert 'snare_dsp_' in exported.headers['content-disposition']
    assert 'X-Kicks-Credits-Remaining' not in exported.headers
    data, sr = sf.read(io.BytesIO(preview.content))
    assert sr == 44100 and len(data) == 65536 and abs(data).max() <= .9
    assert client.get('/generate?' + query).content == preview.content
    assert client.get('/generate?' + query + '&seed=9').content != preview.content


def test_api_omitted_physical_controls_match_voice_defaults(client):
    from kicks.synthesis.procedural import render
    for instrument in ('kick', 'snare', 'hihat'):
        response = client.get(f'/generate?backend=dsp&instrument={instrument}')
        audio, _ = sf.read(io.BytesIO(response.content), dtype='float32')
        np.testing.assert_allclose(audio, render(instrument), atol=2**-22)


def test_physical_parameter_does_not_collide_with_effect(client):
    dry = client.get('/generate?backend=dsp&instrument=hihat&voice_decay_ms=1')
    shaped = client.get('/generate?backend=dsp&instrument=hihat&voice_decay_ms=1&decay_ms=20')
    a, _ = sf.read(io.BytesIO(dry.content)); b, _ = sf.read(io.BytesIO(shaped.content))
    assert np.sum(a[4000:]**2) > np.sum(b[4000:]**2) * 100
    report = client.get('/evaluate?backend=dsp&instrument=hihat').json()
    assert report['control_fit'] is None and not report['corpus_available']
    assert report['descriptors'] and 'score' not in report
    assert client.get('/generate?backend=dsp&seed=-1').status_code == 422
    assert client.get('/generate?backend=dsp&s1=nan').status_code == 422
    assert client.get('/generate?backend=dsp&instrument=bad').status_code == 404
    assert client.get('/spectrogram?backend=dsp&instrument=hihat').json()['source'] == 'render'
