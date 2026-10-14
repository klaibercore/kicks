"""Signal, preset, fitting and artifact integrity contracts for the DSP engine."""
import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from kicks.audio.constants import AUDIO_LENGTH, SAMPLE_RATE
from kicks.instruments import available, get_profile
from kicks.synthesis.procedural import Preset, choke, generate_dsp, hybrid, phase_accumulate, render


@pytest.mark.parametrize('name', available())
def test_control_schema_and_rendered_signal(name):
    profile = get_profile(name)
    spec = profile.procedural
    for p in spec.parameters:
        for position in (0, .1, .5, .9, 1):
            assert p.position(p.value(position)) == pytest.approx(position)
    for preset in spec.presets:
        x = render(profile, preset=preset, seed=42)
        assert x.dtype == np.float32 and x.shape == (AUDIO_LENGTH,)
        assert np.isfinite(x).all() and abs(x).max() == pytest.approx(.9)
        assert x[0] == x[-1] == 0
        np.testing.assert_array_equal(x, render(profile, preset=preset, seed=42))
    assert not np.array_equal(render(profile, seed=1), render(profile, seed=2))
    for values in ([0.] * len(spec.parameters), [1.] * len(spec.parameters)):
        assert np.isfinite(render(profile, spec.from_positions(values))).all()
    with pytest.raises(ValueError):
        render(profile, {'unknown': 1})
    with pytest.raises(ValueError):
        render(profile, {spec.parameters[0].key: float('nan')})


def test_phase_integrates_frequency_and_tuning_is_preserved():
    frequency = np.linspace(200, 50, 1000)
    phase = phase_accumulate(frequency, 44100)
    np.testing.assert_allclose(np.diff(phase), 2 * np.pi * frequency[:-1] / 44100)
    for tune in (40, 75):
        x = render('kick', {'tune_hz': tune, 'pitch_drop_hz': 0, 'tone': 0, 'click': 0, 'drive': 0, 'decay_ms': 1800})
        spectrum = abs(np.fft.rfft(x[4410:22050] * np.hanning(17640), n=131072))
        hz = np.fft.rfftfreq(131072, 1 / SAMPLE_RATE)[np.argmax(spectrum)]
        assert hz == pytest.approx(tune, abs=1)


def test_hat_articulation_decay_and_choke():
    closed = render('hihat', preset='closed', seed=8)
    opened = render('hihat', preset='open', seed=8)
    after = int(.2 * SAMPLE_RATE)
    assert np.sum(opened[after:] ** 2) > np.sum(closed[after:] ** 2) * 100
    cut = choke(opened, 80, 5)
    start, end = round(.08 * SAMPLE_RATE), round(.085 * SAMPLE_RATE)
    np.testing.assert_array_equal(cut[:start], opened[:start])
    assert np.max(abs(cut[end + 1:])) == 0
    assert cut[start] == opened[start]


def test_preset_and_hybrid_are_reproducible_and_bind_texture_bytes(tmp_path):
    texture = tmp_path/'texture.wav'
    sf.write(texture, render('snare'), SAMPLE_RATE, subtype='PCM_24')
    from kicks.analysis.benchmark import file_hash
    values = get_profile('kick').procedural.resolve()
    preset = Preset('kick', values, seed=9, texture={'path': str(texture), 'sha256': file_hash(texture), 'mix': .2})
    preset.save(tmp_path/'preset.json')
    restored = Preset.load(tmp_path/'preset.json')
    np.testing.assert_array_equal(restored.audio(), preset.audio())
    sf.write(texture, render('snare', seed=10), SAMPLE_RATE)
    with pytest.raises(ValueError, match='texture bytes'):
        restored.audio()
    data = json.loads((tmp_path/'preset.json').read_text())
    data['seed'] = 17
    (tmp_path/'preset.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='identity'):
        Preset.load(tmp_path/'preset.json')
    base = render('kick')
    np.testing.assert_array_equal(hybrid(base, base, mix=0), base)


def test_generation_writes_24bit_and_replayable_presets(tmp_path):
    paths = generate_dsp('hihat', 2, tmp_path, seed=23, preset='open')
    assert len(paths) == 2
    assert sf.info(paths[0]).subtype == 'PCM_24'
    from pathlib import Path
    config = Preset.load(Path(paths[0]).with_suffix('.json'))
    saved, _ = sf.read(paths[0], dtype='float32')
    np.testing.assert_allclose(saved, config.audio(), atol=2**-22)
    with pytest.raises(FileExistsError):
        generate_dsp('hihat', 2, tmp_path, seed=23, preset='open')
    with pytest.raises(FileExistsError):
        generate_dsp('hihat', 3, tmp_path, seed=22, preset='open')
    assert not (tmp_path/'hihat_open_22.wav').exists()


def test_core_import_does_not_load_torch_or_plotting():
    result = subprocess.run([sys.executable, '-c',
        "import kicks.synthesis.procedural; import kicks.analysis.benchmark; import sys; assert 'torch' not in sys.modules; assert 'matplotlib' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_budgeted_parameter_fit_improves_or_preserves_initial_objective():
    from kicks.synthesis.fitting import fit_hit
    ref = render('kick', preset='punch', length=16384)
    preset, report = fit_hit(ref, 'kick', evaluations=12, seed=0)
    assert report['best_loss'] <= report['initial_loss']
    assert 4 <= report['evaluations'] <= 12
    assert 'paired_reconstruction' in report
    assert np.isfinite(preset.audio()).all()
