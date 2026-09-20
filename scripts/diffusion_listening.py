"""Blind A/B listening pairs for waveform diffusion samples, for the dashboard's Listening lab.

The diffusion backend has no encoder, so there is no held-out reconstruction to
report. What can be judged blind is generation: each sampled hit is paired with
the corpus hit whose descriptors it was asked for (targets are resampled from
the checkpoint's label bank, i.e. from real training rows), level-matched with
the fidelity module's shared anti-clip gain, and written as whole-hit and
2 kHz+ pairs in random order with a key. The report is written in the fidelity
layout so ``kicks dashboard`` discovers it, and is attached to the run as
``kind="fidelity"`` with ``held_out=false`` and ``method`` naming it a
generation A/B — it is not a reconstruction fidelity report and must not be
read as one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from kicks.analysis.fidelity import level_match
from kicks.audio import io
from kicks.audio.constants import SAMPLE_RATE
from kicks.audio.waveform import bandpass
from kicks.config import get_device, load_diffusion_from_checkpoint
from kicks.instruments import get_profile
from kicks.synthesis.diffusion import draw_labels, sample_hits
from kicks.training.tracking import attach_report, find_run, read_run, runs_root

METHOD = "diffusion_generation_ab_v1"
HF_BAND = (2000.0, None)


def nearest_corpus_hits(targets: np.ndarray, std: np.ndarray, report_path: Path, keys: list[str]):
    """For each target row, the corpus file whose report descriptors lie closest (standardised)."""
    rows = json.loads(report_path.read_text())["samples"]
    matrix = np.array([[row["descriptors"][k] for k in keys] for row in rows], dtype=np.float64)
    scale = np.maximum(std, 1e-6)
    chosen = []
    for target in targets:
        distance = np.sqrt(np.sum(((matrix - target) / scale) ** 2, axis=1))
        index = int(np.argmin(distance))
        chosen.append((rows[index]["original_path"], float(distance[index])))
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instrument", default="hihat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run", default=None, help="run id or unique prefix to attach the report to")
    parser.add_argument("--out", type=Path, required=True, help="report directory, e.g. output/fidelity/<slug>")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11, help="texture seed (starting noise)")
    parser.add_argument("--label-seed", type=int, default=5, help="seed for the descriptor targets")
    parser.add_argument("--order-seed", type=int, default=0, help="seed for the blind A/B order")
    parser.add_argument("--cluster-report", type=Path, default=None,
                        help="default: the profile's cluster_analysis.json (per-file descriptors)")
    args = parser.parse_args(argv)

    started = time.monotonic()
    profile = get_profile(args.instrument)
    keys = [d.key for d in profile.descriptors]
    device = get_device()
    model, meta = load_diffusion_from_checkpoint(args.checkpoint, device, profile)
    if model.label_bank is None:
        print("error: checkpoint has no label bank; targets would not correspond to corpus hits", file=sys.stderr)
        return 2

    labels = draw_labels(model, args.count, None, profile, args.label_seed)
    waves = sample_hits(model, labels, steps=args.steps, seed=args.seed, device=device).numpy()
    std = model.label_std.detach().cpu().numpy()
    references = nearest_corpus_hits(labels.numpy(), std, args.cluster_report or Path(profile.paths.cluster_analysis), keys)

    out = args.out
    (out / "listening").mkdir(parents=True, exist_ok=True)
    (out / "generation").mkdir(parents=True, exist_ok=True)
    meter = io.make_meter()
    rng = random.Random(args.order_seed)
    pairs, hits, generation = [], [], []
    for index, (audio, (ref_path, distance)) in enumerate(zip(waves, references), 1):
        reference = io.load_waveform(ref_path, meter, length=model.length)
        if reference is None:
            print(f"  {index:02d}: could not load {ref_path}, skipped", file=sys.stderr)
            continue
        ref, gen = level_match(reference[0].numpy(), audio)
        name = f"diff_{index:02d}.wav"
        sf.write(out / "generation" / name, gen.astype(np.float32), SAMPLE_RATE, subtype="PCM_24")
        a_is, b_is = rng.sample(["reference", "diffusion"], 2)
        chosen = {"reference": ref, "diffusion": gen}
        for label, kind in (("A", a_is), ("B", b_is)):
            sf.write(out / "listening" / f"pair_{index:02d}_{label}.wav", chosen[kind], SAMPLE_RATE, subtype="PCM_24")
            sf.write(out / "listening" / f"pair_{index:02d}_{label}_hf.wav",
                     bandpass(chosen[kind], HF_BAND), SAMPLE_RATE, subtype="PCM_24")
        pairs.append({"pair": index, "a_is": a_is, "b_is": b_is,
                      "a": f"pair_{index:02d}_A.wav", "b": f"pair_{index:02d}_B.wav",
                      "a_hf": f"pair_{index:02d}_A_hf.wav", "b_hf": f"pair_{index:02d}_B_hf.wav"})
        hits.append(Path(ref_path).name)
        generation.append({"file": name, "score": None, "reference": Path(ref_path).name,
                           "target": {k: float(v) for k, v in zip(keys, labels[index - 1].tolist())},
                           "reference_distance_sd": distance})
        print(f"  {index:02d}: {name} vs {Path(ref_path).name} (target distance {distance:.2f} sd)")
    (out / "listening" / "key.json").write_text(json.dumps(pairs, indent=2))

    run_dir = find_run(runs_root(None, profile.paths.output_root), args.run) if args.run else None
    run = read_run(run_dir) if run_dir else None
    checkpoint = Path(args.checkpoint).resolve()
    summary = {"samples": 0, "held_out": False, "vocoder": None, "pairs": len(pairs),
               "generation_mean_score": None, "generation_pass_rate": None,
               "seconds": time.monotonic() - started}
    report = {
        "instrument": profile.name, "corpus": str(Path(profile.paths.data_dir).resolve()),
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "epoch": meta.get("epoch"), "val_loss": meta.get("val_loss"), "architecture": model.architecture,
        "vocoder": None, "run_id": run["id"] if run else meta.get("training", {}).get("run_id"),
        "held_out": False,
        "hit_origin": "corpus hits nearest to each sample's requested descriptors (label bank rows); not held out",
        "hits": hits, "method": METHOD, "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling": {"steps": args.steps, "seed": args.seed, "label_seed": args.label_seed, "guidance": 1.0},
        "summary": summary, "samples": [], "generation": generation,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    if run:
        attach_report(run_dir, "fidelity", out / "report.json", summary,
                      note=f"{METHOD}: {len(pairs)} blind pairs, diffusion vs descriptor-matched corpus hits; "
                           "generation A/B only, no reconstruction, not held out")
    print(f"Wrote {len(pairs)} pairs to {out}/listening/ (key.json holds the assignment).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
