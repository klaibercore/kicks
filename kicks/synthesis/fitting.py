"""Budgeted inverse DSP fitting and a joint empirical physical-parameter prior.

These are analysis/resynthesis experiments, not a trained neural generator.
Noise phase is intentionally excluded from the fitting objective.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
from scipy import optimize, signal
import soundfile as sf

from ..analysis.benchmark import descriptors, file_hash, identity, load_manifest, measure, write_json
from ..audio.constants import SAMPLE_RATE
from ..audio.waveform import load_audio
from ..instruments import get_profile
from .procedural import Preset, RENDERER_VERSION, render


def _features(audio):
    features = []
    for size in (128, 512, 2048, 8192):
        _, _, spec = signal.stft(audio, nperseg=size, noverlap=size * 3 // 4, boundary='zeros')
        # Equal per-window contribution. Magnitudes tolerate stochastic phase.
        magnitude = np.abs(spec)
        features.append(20 * np.log10(np.maximum(magnitude, 1e-5)))
    return features


def fit_hit(reference, instrument='kick', *, evaluations=64, seed=0):
    if evaluations < 8:
        raise ValueError('at least eight objective evaluations are required')
    profile = get_profile(instrument)
    voice = profile.procedural
    ref = np.asarray(reference, dtype=np.float32)
    if ref.ndim != 1 or len(ref) < 8192 or not np.isfinite(ref).all() or np.max(np.abs(ref)) < 1e-6:
        raise ValueError('reference must be finite, audible and at least 8192 samples long')
    ref = ref / np.max(np.abs(ref)) * .9
    target = _features(ref)
    calls, best_loss, best = 0, float('inf'), None
    started = time.perf_counter()

    def objective(positions):
        nonlocal calls, best_loss, best
        calls += 1
        audio = render(profile, voice.from_positions(np.clip(positions, 0, 1).tolist()), seed=seed, length=len(ref))
        features = _features(audio)
        # Audible reference bins plus a symmetric all-bin term for unwanted noise.
        loss = np.mean([np.abs(a[b > -65] - b[b > -65]).mean() * .8 + np.abs(a - b).mean() * .2
                        for a, b in zip(features, target)])
        if loss < best_loss:
            best_loss, best = float(loss), positions.copy()
        return float(loss)

    initial = []
    for name in voice.presets:
        values = voice.resolve(preset=name)
        positions = np.array([p.position(values[p.key]) for p in voice.parameters])
        initial.append((objective(positions), positions))
    first = min(initial, key=lambda pair: pair[0])[1]
    initial_loss = min(pair[0] for pair in initial)
    optimize.minimize(objective, first, method='Powell', bounds=[(0., 1.)] * len(first),
                      options={'maxfev': max(1, evaluations - calls), 'xtol': .02, 'ftol': .002})
    values = voice.from_positions(np.clip(best, 0, 1).tolist())
    preset = Preset(profile.name, values, seed, length=len(ref))
    audio = preset.audio()
    return preset, {"objective": 'active-bin multi-window magnitude MAE dB; FFT 128/512/2048/8192',
                    "initial_loss": float(initial_loss), "best_loss": best_loss, "evaluations": calls,
                    "seconds": time.perf_counter() - started,
                    "descriptors": descriptors(audio, profile), **measure(audio, profile, ref)}


def fit_corpus(manifest, instrument='kick', *, out='models/experiments/dsp-fit.json', count=8, evaluations=64, seed=42):
    document = load_manifest(manifest)
    profile = get_profile(instrument)
    rows = [r for r in document['rows'] if r['instrument'] == profile.name and r['split'] == 'train']
    if count < 1 or not rows:
        raise ValueError('positive count and a nonempty training fold are required')
    if Path(out).exists():
        raise FileExistsError(f'refusing to overwrite fitted bank {out}')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(rows), min(count, len(rows)), replace=False)
    fitted = []
    for index in chosen:
        row = rows[index]
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('training audio no longer matches the frozen manifest')
        reference = load_audio(row['path'])
        if reference is None:
            raise ValueError(f'unusable training hit: {row["id"]}')
        preset, report = fit_hit(reference, profile.name, evaluations=evaluations, seed=seed)
        fitted.append({"source_id": row['id'], "source_sha256": row['sha256'],
                       "preset": asdict(preset), "report": report})
    spec = {"version": 1, "renderer_version": RENDERER_VERSION, "instrument": profile.name,
            "manifest_sha256": document['sha256'], "split": 'train', "seed": seed, "fits": fitted,
            "method": 'bounded physical fitting; joint empirical prior, no neural weights'}
    write_json(Path(out), {**spec, "sha256": identity(spec)})
    return spec


def sample_bank(bank_path, count=4, *, out='output/dsp-fitted', seed=0, jitter=.02):
    bank = __import__('json').loads(Path(bank_path).read_text())
    if identity({k: v for k, v in bank.items() if k != 'sha256'}) != bank.get('sha256'):
        raise ValueError('fitted bank identity changed')
    if bank['renderer_version'] != RENDERER_VERSION or bank.get('split') != 'train':
        raise ValueError('incompatible fitted bank')
    if not 0 <= jitter <= .1 or count < 1:
        raise ValueError('count must be positive and jitter must be in [0, .1]')
    profile = get_profile(bank['instrument'])
    rng = np.random.default_rng(seed)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite generated bank {out}')
    out.mkdir(parents=True)
    for i in range(count):
        fit = bank['fits'][int(rng.integers(len(bank['fits'])))]['preset']
        positions = [p.position(fit['parameters'][p.key]) for p in profile.procedural.parameters]
        values = profile.procedural.from_positions(np.clip(np.array(positions) + rng.normal(0, jitter, len(positions)), 0, 1).tolist())
        preset = Preset(profile.name, values, int(rng.integers(2**32)))
        path = out / f'{profile.name}_fitted_{i:03d}.wav'
        sf.write(path, preset.audio(), SAMPLE_RATE, subtype='PCM_24')
        preset.save(path.with_suffix('.json'))
    return count
