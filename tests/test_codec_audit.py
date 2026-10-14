"""Exercise the codec adapter contract without downloading or loading weights."""
import sys
from types import SimpleNamespace

import pytest
import soundfile as sf

from kicks.analysis.benchmark import freeze_corpus
from kicks.analysis.codec_audit import audit_codec
from kicks.audio.constants import SAMPLE_RATE
from kicks.synthesis.procedural import render


@pytest.fixture
def codec_inputs(tmp_path):
    directory = tmp_path/'data/kicks'
    directory.mkdir(parents=True)
    for name, preset in zip(('808_a', 'mck_rx5_a', 'spi_a'), ('round', 'punch', 'dry')):
        sf.write(directory/f'{name}.wav', render('kick', preset=preset), SAMPLE_RATE)
    manifest = tmp_path/'manifest.json'
    document = freeze_corpus(tmp_path/'data', manifest, registry=tmp_path/'none')
    checkpoint = tmp_path/'codec.pth'
    checkpoint.write_bytes(b'fixture; not model weights')
    return checkpoint, manifest, document


def test_matched_codec_roundtrip_uses_only_held_out_sources(codec_inputs, tmp_path, monkeypatch):
    checkpoint, manifest, document = codec_inputs
    calls = []

    class Codec:
        sample_rate = SAMPLE_RATE
        def eval(self): return self
        def cpu(self): return self
        def __call__(self, audio, sample_rate):
            assert audio.ndim == 3 and sample_rate == SAMPLE_RATE
            calls.append(audio.shape)
            return {'audio': audio * .98}

    monkeypatch.setitem(sys.modules, 'dac', SimpleNamespace(DAC=SimpleNamespace(load=lambda _: Codec())))
    result = audit_codec(checkpoint, manifest, 'kick', out=tmp_path/'audit')
    expected = {r['id'] for r in document['rows'] if r['split'] == 'test'}
    assert expected and {r['source_id'] for r in result['samples']} == expected
    assert len(calls) == len(expected)
    assert result['held_out'] and all('paired_reconstruction' in r for r in result['samples'])
    assert sf.info(tmp_path/'audit'/result['samples'][0]['file']).subtype == 'FLOAT'


def test_codec_refuses_missing_holdout_or_wrong_native_rate(codec_inputs, tmp_path, monkeypatch):
    checkpoint, manifest, _ = codec_inputs
    monkeypatch.setitem(sys.modules, 'dac', SimpleNamespace(DAC=SimpleNamespace(load=lambda _: None)))
    with pytest.raises(ValueError, match='no test rows'):
        audit_codec(checkpoint, manifest, 'hihat', out=tmp_path/'empty')
    assert not (tmp_path/'empty').exists()
    codec = SimpleNamespace(sample_rate=24000)
    codec.eval = lambda: codec
    codec.cpu = lambda: codec
    monkeypatch.setitem(sys.modules, 'dac', SimpleNamespace(DAC=SimpleNamespace(load=lambda _: codec)))
    with pytest.raises(ValueError, match='native 44.1'):
        audit_codec(checkpoint, manifest, 'kick', out=tmp_path/'wrong-rate')
    assert not (tmp_path/'wrong-rate').exists()
