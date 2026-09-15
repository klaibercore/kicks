"""Quarantine loops, double-hits and perceptual outliers from a corpus.

Whatever survives `strip` still contains files that are not one-shots of the
instrument they are filed under, and the VAE will faithfully learn every one of
them. This scans the corpus with the same analyzer the evaluator uses and moves
offenders into a quarantine directory — moved, never deleted, with a manifest,
so a bad threshold is reversible.

Rules:

* ``loop``       — at least ``OnsetSpec.loop_onsets`` detected onsets: a loop or roll.
* ``double_hit`` — exactly two onsets more than ``OnsetSpec.double_hit_ms`` apart,
  i.e. a genuine second hit. Two onsets closer than that (beater bounce,
  envelope wobble) are kept, because the detector over-counts those.
* ``outlier``    — the bottom N% by Mahalanobis likeness among the survivors:
  mislabeled percussion, FX, broken files.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from ..analysis.evaluation import analyze_hit, reference_from_rows
from ..audio.waveform import detect_onsets, load_audio
from ..instruments import InstrumentProfile, get_profile


def run_clean(
    data: str | None = None,
    instrument: str | None = None,
    quarantine_dir: str | None = None,
    outlier_pct: float = 2.0,
    apply: bool = False,
) -> dict:
    """Scan a corpus and quarantine what is not a clean one-shot. Dry-run by default."""
    from rich.console import Console

    profile: InstrumentProfile = get_profile(instrument)
    data = data or profile.paths.data_dir
    quarantine_dir = quarantine_dir or profile.paths.quarantine_dir
    manifest_path = profile.paths.clean_manifest

    console = Console()
    files = sorted(f for f in os.listdir(data) if f.lower().endswith(".wav"))
    console.print(f"Scanning {len(files)} files in {data} as {profile.plural}...")

    decisions: dict[str, str] = {}
    rows: dict[str, list[float]] = {}
    kept: list[tuple[str, dict[str, float]]] = []

    for i, f in enumerate(files):
        x = load_audio(os.path.join(data, f))
        if x is None:
            decisions[f] = "unreadable"
            continue
        onsets = detect_onsets(x, profile.onsets)
        if len(onsets) >= profile.onsets.loop_onsets:
            decisions[f] = "loop"
            continue
        if len(onsets) == 2 and (onsets[1] - onsets[0]) > profile.onsets.double_hit_ms:
            decisions[f] = "double_hit"
            continue
        m = analyze_hit(x, profile)
        if m is None:
            decisions[f] = "unreadable"
            continue
        kept.append((f, m))
        for k, v in m.items():
            rows.setdefault(k, []).append(v)
        if (i + 1) % 250 == 0:
            console.print(f"[dim]  {i + 1}/{len(files)}[/dim]")

    if not kept:
        console.print("[red]Nothing survived the onset filters — check the "
                      "instrument and the corpus directory.[/red]")
        return {"counts": {}, "keep": 0, "decisions": decisions}

    # Outliers: Mahalanobis distance measured against the survivors themselves,
    # so "atypical" means atypical for *this* corpus, not for some fixed idea.
    ref = reference_from_rows(
        {k: np.array(v) for k, v in rows.items()}, len(kept), profile,
    )
    d2 = np.array([
        float((ref.transform(m) - ref.mean_vec) @ ref.cov_inv
              @ (ref.transform(m) - ref.mean_vec))
        for _, m in kept
    ])
    cutoff = np.percentile(d2, 100.0 - outlier_pct)
    for (f, _), d in zip(kept, d2):
        if d > cutoff:
            decisions[f] = "outlier"

    counts: dict[str, int] = {}
    for reason in decisions.values():
        counts[reason] = counts.get(reason, 0) + 1
    n_keep = len(files) - len(decisions)

    console.print()
    for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        console.print(f"  {reason:12s} {n:5d} files")
    console.print(f"  {'keep':12s} {n_keep:5d} files "
                  f"({100.0 * n_keep / max(1, len(files)):.1f}% of corpus)")

    if apply:
        for f, reason in decisions.items():
            dest_dir = os.path.join(quarantine_dir, reason)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(os.path.join(data, f), os.path.join(dest_dir, f))
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w") as fh:
            json.dump({"instrument": profile.name, "data_dir": data,
                       "quarantine_dir": quarantine_dir, "decisions": decisions}, fh, indent=2)
        console.print(f"\nMoved {len(decisions)} files to {quarantine_dir}/ "
                      f"(manifest: {manifest_path})")
    else:
        console.print("\n[yellow]Dry run — no files moved. "
                      "Pass --apply to quarantine.[/yellow]")

    return {"counts": counts, "keep": n_keep, "decisions": decisions}
