"""Reproducible API control sweeps, with waveform scores and cross-talk.

Run: .venv/bin/python scripts/benchmark_controls.py --out output/calibration/before
The report keeps individual measurements, including failed/silent renders.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


def get(server, endpoint, params):
    url = f"{server}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--instrument", default="kick")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random", type=int, default=16)
    parser.add_argument("--corners", action="store_true")
    parser.add_argument("--audio", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = {"instrument": args.instrument}
    config = json.loads(get(args.server, "config", base))
    keys = [s["key"] for s in config["sliders"]]
    dims = len(keys)
    points = [("centre", np.full(dims, 0.5))]
    for j, key in enumerate(keys):
        for v in (0.0, 0.25, 0.75, 1.0):
            point = np.full(dims, 0.5)
            point[j] = v
            points.append((f"{key}_{v:.2f}", point))
    if args.corners:
        points.extend((f"corner_{i}", np.array(p)) for i, p in enumerate(
            itertools.product((0.0, 1.0), repeat=dims)))
    points.extend((f"random_{i}", p) for i, p in enumerate(
        np.random.default_rng(20260915).uniform(size=(args.random, dims))))
    rows = []
    for label, point in points:
        params = {**base, **dict(zip(keys, point))}
        start = time.monotonic()
        result = json.loads(get(args.server, "evaluate", params))
        row = {"label": label, "positions": point.tolist(), "seconds": time.monotonic() - start,
               **result}
        rows.append(row)
        if args.audio:
            (args.out / f"{label}.wav").write_bytes(get(args.server, "generate", params))
        (args.out / "report.json").write_text(json.dumps({"config": config, "results": rows}, indent=2))
        print(label, round(row["seconds"], 2), result.get("score", result.get("error")),
              result.get("descriptors"), flush=True)
    scores = np.array([r.get("score", 0) for r in rows])
    descriptor_keys = list(rows[0]["descriptors"])
    response = []
    for key in keys:
        low, high = (next(r for r in rows if r["label"] == f"{key}_{v:.2f}") for v in (0, 1))
        response.append([high["descriptors"][d] - low["descriptors"][d] for d in descriptor_keys])
    summary = {"n": len(rows), "mean_score": float(scores.mean()), "min_score": float(scores.min()),
               "pass_rate": float((scores >= 70).mean()),
               "median_seconds": float(np.median([r["seconds"] for r in rows])),
               "descriptor_keys": descriptor_keys, "endpoint_response": response}
    payload = {"config": config, "summary": summary, "results": rows}
    (args.out / "report.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
