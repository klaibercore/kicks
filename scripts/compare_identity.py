"""Listen to corpus identity, reconstruction and both control algorithms.

Run after validate_controls.py --control identity --texture-seeds 1 2 3.
The algorithm comparison uses identical physical descriptor targets, separating
the anchor/correction change from the new percentile mapping of the sliders.
Outputs stay in the private output directory; no corpus paths are published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

from kicks.analysis.calibration import fit_or_load_basis, calibration_fingerprint
from kicks.analysis.descriptors import DecoderResponse
from kicks.analysis.fidelity import compare_waveforms, level_match
from kicks.audio import io
from kicks.audio.constants import SAMPLE_RATE
from kicks.audio.controls import correct_waveform
from kicks.audio.vocoder import load_vocoder, spec_to_audio
from kicks.config import get_device, load_vae_from_checkpoint
from kicks.data.dataset import load_spectrogram
from kicks.instruments import get_profile
from kicks.synthesis.identity import IDENTITY_VERSION, fit_identity_basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads((args.audit / "report.json").read_text())
    checkpoint = Path(report["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["checkpoint_sha256"], "Checkpoint changed since audit"
    assert report["calibration"]["version"] == IDENTITY_VERSION
    profile = get_profile(args.instrument)
    device = get_device()
    model, _ = load_vae_from_checkpoint(str(checkpoint), device)
    paths = [str(Path(profile.paths.data_dir) / name) for name in io.list_wavs(profile.paths.data_dir)]
    dataset = SimpleNamespace(paths=paths)
    # Reuse verified caches without loading thousands of unused spectrograms.
    # A changed/unreadable corpus needs a fresh full audit before comparison.
    fingerprint = calibration_fingerprint(str(checkpoint), dataset, profile)
    for suffix, expected in [('.controls.npz', fingerprint), ('.identity.npz', hashlib.sha256(f'{IDENTITY_VERSION}:{fingerprint}'.encode()).hexdigest())]:
        with np.load(checkpoint.with_suffix(suffix), allow_pickle=False) as cache:
            assert str(cache["fingerprint"]) == expected, "Corpus/cache mismatch: run a fresh full audit first"
    basis = fit_identity_basis(model, dataset, profile, device, checkpoint=checkpoint)
    legacy = fit_or_load_basis(model, dataset, profile, device, checkpoint=checkpoint)
    vocoder = load_vocoder(device, report["vocoder"], weights_dir=profile.paths.vocoder_dir)
    response = DecoderResponse(model, profile)
    targets = np.array(report["results"][0]["target"])
    z = legacy.basis.solve(targets, response)
    with torch.no_grad():
        old_spec = model.decode(torch.tensor(z, dtype=torch.float32, device=device))
    old = correct_waveform(spec_to_audio(old_spec, vocoder, device)[0], targets,
                           np.array(legacy.maxs) - legacy.mins, profile).numpy()
    selection = np.random.default_rng(0).choice(len(paths), min(1500, len(paths)), replace=False)
    out = args.audit / "listening"
    out.mkdir(exist_ok=True)
    meter = io.make_meter()
    rows = []
    for seed in args.seeds:
        idx = basis.basis.anchor_index(seed)
        source = paths[selection[idx]]
        spec = load_spectrogram(source, meter, model.n_frames)
        with torch.no_grad():
            encoded, _ = model.encode(spec[None].to(device))
            np.testing.assert_allclose(encoded.cpu().numpy()[0], basis.basis.support_z[idx], atol=5e-4, rtol=5e-4)
            reconstruction = spec_to_audio(model.decode(encoded), vocoder, device)[0].numpy()
        reference = io.load_waveform(source, meter)[0].numpy()
        name = "centre" if seed == 0 else f"texture_{seed}"
        new, sr = sf.read(args.audit / f"{name}.wav", dtype="float32")
        assert sr == SAMPLE_RATE
        assert all(np.isfinite(x).all() for x in (reference, reconstruction, old, new))
        matched = level_match(reference, reconstruction, old, new)
        files = {}
        for label, waveform in zip(("source", "reconstruction", "previous", "new"), matched):
            filename = f"seed-{seed}-{label}.wav"
            sf.write(out / filename, waveform, SAMPLE_RATE, subtype="PCM_24")
            files[label] = filename
        row = {"seed": seed, "source": source, "files": files,
               "vs_source": {label: compare_waveforms(matched[0], x, profile)
                             for label, x in zip(("reconstruction", "previous", "new"), matched[1:])}}
        rows.append(row)
        print(f"seed {seed}: source encoding verified; matched comparison saved", flush=True)
    payload = {"instrument": profile.name, "checkpoint_sha256": report["checkpoint_sha256"],
               "comparison": "same physical descriptor targets; different texture anchors",
               "targets": targets.tolist(), "listening_verdict": "pending", "results": rows}
    (out / "report.json").write_text(json.dumps(payload, indent=2))
    # Blind algorithm order, with the corpus source and unmodified reconstruction
    # available as references. Source filenames remain in the private JSON only.
    rng = np.random.default_rng(20260917)
    cards = []
    for row in rows:
        labels = rng.permutation(["previous", "new"]).tolist()
        players = [("Corpus source", "source"), ("Reconstruction", "reconstruction"), ("A", labels[0]), ("B", labels[1])]
        players_html = ''.join(f'<label>{title}<audio controls preload="none" src="{row["files"][label]}"></audio></label>' for title, label in players)
        cards.append(f'<section><h2>Texture {row["seed"]}</h2><div class="players">{players_html}</div><details><summary>Reveal algorithms</summary>A: {labels[0]} · B: {labels[1]}</details></section>')
    (out / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Drum identity · listening</title>
<style>body{background:#10131a;color:#eff2f6;font:15px/1.6 system-ui;margin:0;padding:40px;max-width:1080px}h1{font-size:36px;letter-spacing:-1px}p{color:#a6b0c0;max-width:760px}section{border-top:1px solid #303747;padding:20px 0}h2{font-size:17px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:18px}label{display:grid;gap:10px}audio{width:100%;height:38px}details{color:#a6b0c0;margin-top:20px}summary{cursor:pointer}</style>
<p>KICKS / GENERATION RESEARCH</p><h1>Does the drum keep its identity?</h1><p>Compare the shell and wire rattle of the snare, or the stick contact and metallic shimmer of the hi-hat. A and B use the same physical control targets and checkpoint. All four clips share a matched RMS level and anti-clip gain. A brighter or louder-sounding clip is not automatically better.</p><p>The source and reconstruction show what this texture was before slider control. These are generation comparisons, not held-out reconstruction evaluations. Listening verdicts remain pending.</p>''' + ''.join(cards) + '</html>')
    print(out / "index.html", flush=True)


if __name__ == "__main__":
    main()
