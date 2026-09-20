"""Build a stratified corpus subset for budgeted training runs.

Reads the instrument's cluster report (``kicks cluster``), draws files in
proportion to each sound family and writes them as symlinks into a new
directory that ``kicks train --data`` / ``kicks diffusion-train --data`` can
read. The draw is seeded and *nested*: with the same seed, a 256-file subset
is a subset of the 2,000-file one, so a ladder of runs sees a growing corpus
rather than a different one each time.

Nesting comes from a single global order. Each family's files are shuffled
once with the seed; a file's key is its rank in that shuffle divided by the
family's size, so taking the ``size`` smallest keys gives every family a share
proportional to its population, and a larger ``size`` only appends.

The report must describe the live corpus: the file count and every path are
checked, and a stale report is refused rather than silently sampled — another
session may have moved files in since ``kicks cluster`` ran.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from kicks.audio.io import list_wavs
from kicks.instruments import get_profile


def load_report(path: Path) -> list[dict]:
    with open(path) as handle:
        report = json.load(handle)
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} has no per-sample rows; run `kicks cluster` first")
    for row in samples:
        if "original_path" not in row or "cluster" not in row:
            raise ValueError(f"{path} rows lack original_path/cluster; regenerate it")
    return samples


def check_current(samples: list[dict], corpus_dir: Path) -> None:
    """Refuse a report that no longer matches the directory it describes."""
    live = set(list_wavs(str(corpus_dir)))
    listed = {Path(row["original_path"]).name for row in samples}
    if len(samples) != len(live) or listed != live:
        missing = sorted(listed - live)[:3]
        extra = sorted(live - listed)[:3]
        raise ValueError(
            f"cluster report lists {len(samples)} files but {corpus_dir} holds {len(live)} "
            f"(report-only: {missing}..., corpus-only: {extra}...); rerun `kicks cluster` "
            "or pass --force to sample only the files both agree on",
        )


def stratified_order(samples: list[dict], seed: int) -> list[dict]:
    """Every sample once, in an order whose every prefix is family-proportional."""
    families: dict[int, list[dict]] = {}
    for row in sorted(samples, key=lambda r: r["original_path"]):
        families.setdefault(int(row["cluster"]), []).append(row)
    keyed = []
    rng = np.random.default_rng(seed)
    for family in sorted(families):
        rows = families[family]
        for rank, index in enumerate(rng.permutation(len(rows))):
            keyed.append(((rank + 0.5) / len(rows), family, rank, rows[index]))
    keyed.sort(key=lambda item: item[:3])
    return [item[3] for item in keyed]


def build_subset(
    samples: list[dict], size: int, seed: int, out_dir: Path, corpus_dir: Path,
    *, force: bool = False,
) -> dict:
    if size < 1 or size > len(samples):
        raise ValueError(f"size must be between 1 and {len(samples)}")
    if out_dir.exists():
        leftovers = [p for p in out_dir.iterdir() if p.name != "manifest.json"]
        if leftovers and not force:
            raise FileExistsError(f"{out_dir} is not empty; pass --force to rebuild it")
        for path in leftovers:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                raise FileExistsError(f"{path} is a directory; refusing to touch it")
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = stratified_order(samples, seed)[:size]
    counts: dict[int, int] = {}
    totals: dict[int, int] = {}
    for row in samples:
        totals[int(row["cluster"])] = totals.get(int(row["cluster"]), 0) + 1
    for row in chosen:
        name = Path(row["original_path"]).name
        target = (corpus_dir / name).resolve()
        if not target.exists():
            raise FileNotFoundError(f"{target} listed in the cluster report is missing")
        os.symlink(target, out_dir / name)
        counts[int(row["cluster"])] = counts.get(int(row["cluster"]), 0) + 1

    manifest = {
        "size": size,
        "seed": seed,
        "corpus_dir": str(corpus_dir.resolve()),
        "corpus_files": len(samples),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "families": [
            {"family": family, "corpus": totals[family], "subset": counts.get(family, 0)}
            for family in sorted(totals)
        ],
        "files": sorted(Path(row["original_path"]).name for row in chosen),
    }
    with open(out_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instrument", default="kick")
    parser.add_argument("--size", type=int, required=True, help="files in the subset")
    parser.add_argument("--seed", type=int, default=42,
                        help="same seed => smaller subsets nest inside larger ones")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: data/_subsets/<instrument>-<size>")
    parser.add_argument("--report", type=Path, default=None,
                        help="default: the profile's cluster_analysis.json")
    parser.add_argument("--force", action="store_true",
                        help="rebuild an existing directory; with a stale report, sample "
                             "only the files present in both report and corpus")
    args = parser.parse_args(argv)

    profile = get_profile(args.instrument)
    corpus_dir = Path(profile.paths.data_dir)
    report = args.report or Path(profile.paths.cluster_analysis)
    out_dir = args.out or Path(profile.paths.data_root) / "_subsets" / f"{profile.name}-{args.size}"

    samples = load_report(report)
    try:
        check_current(samples, corpus_dir)
    except ValueError as error:
        if not args.force:
            print(f"error: {error}", file=sys.stderr)
            return 2
        live = set(list_wavs(str(corpus_dir)))
        samples = [row for row in samples if Path(row["original_path"]).name in live]
        print(f"warning: stale report; sampling from the {len(samples)} files it shares "
              f"with {corpus_dir}", file=sys.stderr)

    manifest = build_subset(samples, args.size, args.seed, out_dir, corpus_dir, force=args.force)
    print(f"{profile.name}: {manifest['size']} of {manifest['corpus_files']} files -> {out_dir} "
          f"(seed {manifest['seed']})")
    print(f"{'family':>6} {'corpus':>7} {'subset':>7}")
    for entry in manifest["families"]:
        print(f"{entry['family']:>6} {entry['corpus']:>7} {entry['subset']:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
