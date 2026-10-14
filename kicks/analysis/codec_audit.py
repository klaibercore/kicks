"""Evaluate a local, matched DAC codec before training any latent prior."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import soundfile as sf

from .benchmark import file_hash, load_manifest, measure, write_json
from ..audio.constants import AUDIO_LENGTH, SAMPLE_RATE
from ..audio.waveform import fit_length, load_audio
from ..instruments import get_profile


def audit_codec(checkpoint, manifest, instrument='hihat', *, out='output/codec-audit', split='test', count=8):
    import dac
    import torch

    if not Path(checkpoint).is_file():
        raise FileNotFoundError('a local DAC checkpoint is required; this command never downloads weights')
    if split not in ('validation', 'test') or count < 1:
        raise ValueError('choose validation or test and a positive count')
    document = load_manifest(manifest)
    profile = get_profile(instrument)
    rows = [r for r in document['rows'] if r['instrument'] == profile.name and r['split'] == split]
    if not rows:
        raise ValueError(f'manifest has no {split} rows for {profile.name}; no fallback to training data')
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite codec audit {out}')
    # DAC.load loads encoder, quantizer and decoder from one local artifact.
    model = dac.DAC.load(checkpoint).eval().cpu()
    if model.sample_rate != SAMPLE_RATE:
        raise ValueError('the first audit requires a native 44.1 kHz DAC model')
    out.mkdir(parents=True)
    results = []
    for row in rows[:count]:
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('codec reference bytes no longer match the manifest')
        audio = load_audio(row['path'])
        if audio is None:
            raise ValueError(f'unusable reference {row["id"]}')
        start = time.perf_counter()
        with torch.inference_mode():
            result = model(torch.from_numpy(audio)[None, None], SAMPLE_RATE)
        reconstructed = fit_length(result['audio'][0, 0].cpu().numpy(), AUDIO_LENGTH)
        path = out / f'roundtrip_{len(results):03d}.wav'
        sf.write(path, reconstructed, SAMPLE_RATE, subtype='FLOAT')
        results.append({"source_id": row['id'], "source_sha256": row['sha256'],
                        "seconds": time.perf_counter() - start, "file": path.name,
                        **measure(reconstructed, profile, reference=audio)})
    report = {"instrument": profile.name, "method": 'matched_DAC_roundtrip_v1',
              "checkpoint_sha256": file_hash(checkpoint), "manifest_sha256": document['sha256'],
              "split": split, "held_out": True, "samples": results, "listening_judged": False,
              "limitation": 'Held out from this project split; pretrained codec training overlap is unknown.'}
    write_json(out/'report.json', report)
    return report
