"""Calibrate a checkpoint and render an independent control/quality audit.

This uses the same preprocessing, solver, vocoder and evaluator as the API.
Calibration probes and validation probes use different random seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

from kicks.analysis.basis import slider_positions_to_axis_values
from kicks.analysis.descriptors import DecoderResponse, descriptor_vector
from kicks.analysis.evaluation import analyze_hit, build_reference, score_sample
from kicks.audio.constants import SAMPLE_RATE
from kicks.audio.mel import spectrogram
from kicks.audio.vocoder import load_vocoder, resolve_vocoder_type
from kicks.config import get_device, load_vae_from_checkpoint
from kicks.data import DrumDataset
from kicks.instruments import get_profile
from kicks.synthesis.controlled import fit_control_basis, render_controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocoder", choices=("discoder", "bigvgan", "griffinlim"), default=None,
                        help="default: the profile's own backend (KICKS_VOCODER overrides)")
    parser.add_argument("--instrument", default="kick")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random", type=int, default=32)
    parser.add_argument("--corners", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--control", choices=("descriptor", "legacy", "identity"), default="descriptor")
    parser.add_argument("--texture-seeds", type=int, nargs="*", default=[],
                        help="Additional centre probes with reproducible texture seeds")
    parser.add_argument("--run", help="Training run ID (or unique prefix) to attach the summary to as 'controls' evidence")
    parser.add_argument("--runs-dir", help="Training records root (default: output/training or KICKS_RUNS_DIR)")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    profile = get_profile(args.instrument)
    device = get_device()
    model, checkpoint = load_vae_from_checkpoint(args.checkpoint, device)
    dataset = DrumDataset(profile=profile, n_frames=model.n_frames)
    basis = fit_control_basis(model, dataset, profile, device, mode=args.control,
                              checkpoint=args.checkpoint, refresh=args.refresh)
    response = DecoderResponse(model, profile)
    span = np.array(basis.maxs) - basis.mins
    dims = len(span)
    points = [("centre", np.full(dims, 0.5))]
    for j, key in enumerate(profile.descriptor_keys):
        for v in (0, 0.25, 0.75, 1):
            point = np.full(dims, 0.5); point[j] = v
            points.append((f"{key}_{v:.2f}", point))
    if args.corners:
        points.extend((f"corner_{i}", np.array(p)) for i, p in enumerate(itertools.product((0, 1), repeat=dims)))
    points.extend((f"random_{i}", p) for i, p in enumerate(np.random.default_rng(20260915).uniform(size=(args.random, dims))))
    points.extend((f"texture_{seed}", np.full(dims, .5)) for seed in args.texture_seeds)
    args.vocoder = resolve_vocoder_type(profile, args.vocoder)
    vocoder = load_vocoder(device, args.vocoder, weights_dir=profile.paths.vocoder_dir)
    reference = build_reference(profile.paths.data_dir, profile)
    rows = []
    for label, position in points:
        start = time.monotonic()
        target = np.array(slider_positions_to_axis_values(position, basis))
        seed = int(label.removeprefix("texture_")) if label.startswith("texture_") else 0
        spec, audio = render_controls(model, basis, profile, response, vocoder, device, position, seed=seed)
        waveform = audio[0].numpy()
        measured = descriptor_vector(spec, profile)
        # This second measurement catches vocoder-induced coupling; descriptor
        # independence alone must never be reported as waveform independence.
        audible = descriptor_vector(spectrogram(audio, n_frames=model.n_frames), profile)
        metrics = analyze_hit(waveform, profile)
        scored = score_sample(label, metrics, reference, profile) if metrics else None
        row = {"label": label, "seed": seed, "positions": position.tolist(), "target": target.tolist(),
               "descriptors": measured.tolist(), "waveform_descriptors": audible.tolist(),
               "target_error": ((audible-target)/span).tolist(),
               "decoder_target_error": ((measured-target)/span).tolist(),
               "score": scored.score if scored else 0, "metrics": metrics,
               "seconds": time.monotonic()-start}
        rows.append(row)
        sf.write(args.out / f"{label}.wav", waveform, SAMPLE_RATE, subtype="PCM_24")
        print(label, f"score={row['score']:.2f}", f"error={np.max(np.abs(row['target_error'])):.5f}",flush=True)
        (args.out / "report.json").write_text(json.dumps({"results": rows}, indent=2))
    matrix = np.array([r["target_error"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    crosstalk = {}
    for kind in ("descriptors", "waveform_descriptors"):
        measured = []
        for key in profile.descriptor_keys:
            axis_rows = [rows[0], *[r for r in rows if r["label"].startswith(key+"_")]]
            measured.append(np.ptp([r[kind] for r in axis_rows],axis=0)/span)
        crosstalk[kind] = np.array(measured).tolist()
    summary = {"n": len(rows), "mean_score": float(scores.mean()), "min_score": float(scores.min()),
               "multiple_onsets": sum(r["metrics"] is not None and r["metrics"]["n_onsets"] > 1 for r in rows),
               "pass_rate": float((scores>=70).mean()), "max_target_error": float(np.abs(matrix).max()),
               "p95_target_error": float(np.percentile(np.abs(matrix),95)),
               "axis_response": crosstalk,
               "median_seconds": float(np.median([r["seconds"] for r in rows]))}
    payload = {"checkpoint": args.checkpoint,
               "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
               "control": args.control, "vocoder": args.vocoder, "epoch": checkpoint.get("epoch"),
               "val_loss": checkpoint.get("val_loss"), "descriptor_keys": profile.descriptor_keys,
               "mins": basis.mins, "maxs": basis.maxs, "calibration": basis.calibration,
               "summary": summary, "results": rows}
    (args.out / "report.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2),flush=True)
    if args.run:
        from kicks.training.tracking import attach_report, find_run, runs_root

        run_dir = find_run(runs_root(args.runs_dir, profile.paths.output_root), args.run)
        attach_report(run_dir, "controls", args.out / "report.json",
                      {k: summary[k] for k in ("n", "mean_score", "min_score", "pass_rate",
                                               "max_target_error", "p95_target_error", "median_seconds")},
                      note=f"{args.vocoder} · {len(rows)} probes{' incl. corners' if args.corners else ''} · epoch {checkpoint.get('epoch')}")
        print(f"Attached controls report to run {run_dir.name}", flush=True)


if __name__ == "__main__":
    main()
