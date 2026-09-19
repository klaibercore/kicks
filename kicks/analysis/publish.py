"""Publish corpus analysis reports as static assets for the website.

``kicks cluster`` writes one ``cluster_analysis.json`` per instrument under
``output/``. Those files are large (the kick report is ~3 MB), carry local
filesystem paths, and print floats to 17 digits. This module rewrites them into
``web/public/analysis/`` in a form a browser can load quickly: filenames and
paths dropped (the corpus itself is not published), per-sample GMM membership
reduced to a single confidence, floats rounded, and
each cluster's averaged ``.wav`` copied alongside. An ``index.json`` lists what
was published so the site discovers instruments instead of hardcoding them.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

from ..instruments import available, get_profile

#: Decimal places kept for every float in the published report.
PRECISION = 4

#: Per-sample keys copied through unchanged. ``filename`` and ``original_path``
#: are deliberately not among them: the site shows the shape of the corpus, not
#: which recordings are in it.
_SAMPLE_KEYS = ("sample_idx", "cluster", "duration_ms")


def _round(value: Any, precision: int) -> Any:
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, dict):
        return {k: _round(v, precision) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, precision) for v in value]
    return value


def _slim_sample(row: dict[str, Any], pc_keys: list[str], precision: int) -> dict[str, Any]:
    slim = {k: row[k] for k in _SAMPLE_KEYS if k in row}
    slim["descriptors"] = row["descriptors"]
    for pc in pc_keys:
        slim[pc] = row[pc]
    for key in ("latent1", "latent2", "latent3", "entropy"):
        if key in row:
            slim[key] = row[key]
    probs = row.get("probs") or []
    slim["confidence"] = max(probs) if probs else 1.0
    return _round(slim, precision)


def publish_report(report: dict[str, Any], precision: int = PRECISION) -> dict[str, Any]:
    """The browser-facing form of one ``cluster_analysis.json`` payload."""
    pc_keys = list(report["pca_loadings"])
    published = {
        k: report[k]
        for k in (
            "instrument", "descriptor_keys", "descriptor_labels",
            "pca_variance_explained", "pca_source", "n_clusters", "corpus",
            "pc_names", "pca_loadings", "pc_descriptor_correlations",
            "descriptor_correlations", "cluster_profiles", "descriptor_stats",
            "schema_version", "generated_at", "clustering", "latent_projection", "cluster_details",
        )
        if k in report
    }
    published["corpus"] = {
        k: v for k, v in published.get("corpus", {}).items() if k != "data_dir"
    }
    published["samples"] = [_slim_sample(s, pc_keys, precision) for s in report["samples"]]
    published["descriptor_docs"] = {
        d.key: d.doc for d in get_profile(report["instrument"]).descriptors
    }
    published["descriptor_units"] = {
        d.key: {"power_db_ratio": "dB", "centroid_ms": "ms"}.get(d.kind, "")
        for d in get_profile(report["instrument"]).descriptors
    }
    return _round(published, precision)


def publish_analysis(
    out_dir: str = os.path.join("web", "public", "analysis"),
    instruments: list[str] | None = None,
    precision: int = PRECISION,
) -> list[dict[str, Any]]:
    """Write ``<out_dir>/<instrument>.json`` and cluster audio for each analysed corpus.

    Instruments without a ``cluster_analysis.json`` are skipped with a note —
    publishing should not fail because one corpus has not been analysed yet.
    Returns the index that was written.
    """
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for name in instruments or available():
        profile = get_profile(name)
        src = profile.paths.cluster_analysis
        if not os.path.exists(src):
            print(f"  {profile.name}: no report at {src} — run `kicks cluster -i {profile.name}`")
            continue
        with open(src) as fh:
            report = json.load(fh)
        published = publish_report(report, precision)

        json_path = os.path.join(out_dir, f"{profile.name}.json")
        with open(json_path, "w") as fh:
            json.dump(published, fh, separators=(",", ":"))

        audio_dir = os.path.join(out_dir, profile.name)
        os.makedirs(audio_dir, exist_ok=True)
        for stale in os.listdir(audio_dir):
            os.remove(os.path.join(audio_dir, stale))
        copied = 0
        for k in range(published["n_clusters"]):
            wav = os.path.join(profile.paths.samples_dir, f"cluster_avg_{k}.wav")
            if os.path.exists(wav):
                shutil.copyfile(wav, os.path.join(audio_dir, f"cluster_avg_{k}.wav"))
                copied += 1

        entry = {
            "name": profile.name,
            "display_name": profile.display_name,
            "description": profile.description,
            "n_samples": len(published["samples"]),
            "n_clusters": published["n_clusters"],
            "descriptors": profile.descriptor_keys,
            "report": f"{profile.name}.json",
            "cluster_audio": copied,
        }
        index.append(entry)
        size_kb = os.path.getsize(json_path) / 1024
        print(f"  {profile.name}: {entry['n_samples']} samples, "
              f"{entry['n_clusters']} clusters, {copied} cluster wavs -> "
              f"{json_path} ({size_kb:.0f} kB)")

    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump({"instruments": index}, fh, indent=2)
    return index
