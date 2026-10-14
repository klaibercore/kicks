"""Frozen splits, duplicate leakage checks and comparison provenance."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from kicks.analysis.benchmark import freeze_corpus, load_manifest, listening_comparison, run_dsp_benchmark
from kicks.audio.constants import SAMPLE_RATE


def corpus(root):
    rng = np.random.default_rng(42)
    directory = root/'kicks'
    directory.mkdir(parents=True)
    (directory/'SOURCES.md').write_text('Test fixtures; not a released sample pack')
    x = rng.normal(0, .1, 9000)
    for name in ('808_a.wav', 'mck_rx5_copy.wav'):
        sf.write(directory/name, x, SAMPLE_RATE)
    sf.write(directory/'808_b.wav', rng.normal(0, .1, 9000), SAMPLE_RATE)
    sf.write(directory/'unknown.wav', rng.normal(0, .1, 9000), SAMPLE_RATE)


def test_manifest_groups_sources_and_exact_duplicates(tmp_path):
    root = tmp_path/'data'
    corpus(root)
    result = freeze_corpus(root, tmp_path/'manifest.json', registry=tmp_path/'absent')
    rows = {Path(r['path']).name: r for r in result['rows']}
    assert rows['808_a.wav']['split_group'] == rows['mck_rx5_copy.wav']['split_group']
    assert rows['808_a.wav']['split'] == rows['808_b.wav']['split']
    assert rows['unknown.wav']['split'] == 'train'
    assert rows['unknown.wav']['license'] == 'unresolved'
    assert result['warnings']
    assert load_manifest(tmp_path/'manifest.json')['sha256'] == result['sha256']
    with pytest.raises(FileExistsError):
        freeze_corpus(root, tmp_path/'manifest.json', registry=tmp_path/'absent')
    result['rows'][0]['split'] = 'train' if result['rows'][0]['split'] == 'test' else 'test'
    (tmp_path/'manifest.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='identity'):
        load_manifest(tmp_path/'manifest.json')


def test_unresolved_duplicate_component_stays_training_only(tmp_path):
    root = tmp_path/'data'
    corpus(root)
    directory = root/'kicks'
    # A resolved source sorts before "unresolved" in the union-find root.
    (directory/'unknown.wav').write_bytes((directory/'808_a.wav').read_bytes())
    result = freeze_corpus(root, tmp_path/'manifest.json', registry=tmp_path/'absent')
    assert {r['split'] for r in result['rows']} == {'train'}


def test_benchmark_and_blind_comparison_bind_actual_audio(tmp_path):
    left, right, out = [tmp_path/n for n in ('left', 'right', 'compare')]
    report = run_dsp_benchmark('hihat', left, settings=1, seeds=1)
    assert report['summary']['generated'] == 1
    assert report['summary']['silent'] == report['summary']['clipping'] == 0
    assert not report['held_out'] and 'paired_reconstruction' not in report['generation'][0]
    run_dsp_benchmark('hihat', right, settings=1, seeds=1)
    result = listening_comparison(left, right, out)
    assert result['summary']['pairs'] == 1
    key = json.loads((out/'listening/key.json').read_text())
    assert set(key[0]) == {'pair', 'a_is', 'b_is', 'a', 'b', 'a_hf', 'b_hf'}
    assert (out/'listening/listening.html').exists()
    page = (out/'listening/listening.html').read_text()
    assert 'Two generated candidates' in page
    assert 'reference recording' not in page and 'VAE reconstruction' not in page
    from kicks.analysis.fidelity import _write_listening_page
    legacy = tmp_path/'legacy'
    legacy.mkdir()
    _write_listening_page(legacy, key)
    assert 'VAE reconstruction' in (legacy/'listening.html').read_text()
    (right/'generation/case_000_seed_0.wav').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity'):
        listening_comparison(left, right, tmp_path/'bad')


def test_fit_bank_only_uses_manifest_training_rows(tmp_path):
    from kicks.synthesis.fitting import fit_corpus, sample_bank
    root = tmp_path/'data'
    corpus(root)
    document = freeze_corpus(root, tmp_path/'manifest.json', registry=tmp_path/'none')
    bank = fit_corpus(tmp_path/'manifest.json', 'kick', out=tmp_path/'bank.json', count=1, evaluations=8)
    training_ids = {r['id'] for r in document['rows'] if r['split'] == 'train'}
    assert {r['source_id'] for r in bank['fits']} <= training_ids
    assert sample_bank(tmp_path/'bank.json', 1, out=tmp_path/'generated') == 1


def test_final_audio_adapter_separates_generation_and_explicit_reconstruction(tmp_path):
    from kicks.analysis.benchmark import audit_directory
    from kicks.synthesis.procedural import render
    generated = tmp_path/'generated'
    generated.mkdir()
    sf.write(generated/'hit.wav', render('kick'), SAMPLE_RATE, subtype='PCM_24')
    fresh = audit_directory(generated, out=tmp_path/'fresh')
    assert 'paired_reconstruction' not in fresh['generation'][0]
    paired = audit_directory(generated, out=tmp_path/'paired', reference_directory=generated)
    assert paired['generation'][0]['paired_reconstruction']['envelope_mae_db'] == pytest.approx(0, abs=1e-6)
    assert paired['held_out'] is False
