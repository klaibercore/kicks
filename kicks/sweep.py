"""Sweep the REST API's slider space and evaluate every result.

Exercises the API the way a client would — Latin hypercube over the sliders plus
the centre point — then reports which sliders correlate with eval quality. That
correlation is the useful part: it says whether a slider is a creative control
or a quality cliff. Requires a running ``kicks serve``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def _get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """n stratified samples in [0, 1]^dims.

    Spreads the samples across the slider space instead of clumping them the way
    independent uniform draws do, so a fixed budget covers more of the space.
    """
    u = (rng.random((n, dims)) + np.arange(n)[:, None]) / n
    for d in range(dims):
        u[:, d] = u[rng.permutation(n), d]
    return u


def run_sweep(
    server: str = "http://localhost:8080",
    instrument: str | None = None,
    count: int = 20,
    out_dir: str = "output/sweep",
    seed: int = 42,
    json_out: str = "output/sweep_report.json",
) -> dict:
    """Generate ``count`` diverse hits through the API and evaluate each."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    server = server.rstrip("/")
    query_base = {"instrument": instrument} if instrument else {}

    try:
        cfg = _get_json(f"{server}/config?" + urllib.parse.urlencode(query_base))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {server} — is `kicks serve` running? ({exc})")

    sliders = cfg["sliders"]
    names = [s["name"] for s in sliders]
    console.print(f"Server ready — {cfg['display_name']} sliders: {', '.join(names)} "
                  f"(vocoder: {cfg.get('vocoder', '?')}, control: {cfg.get('control', '?')})")

    rng = np.random.default_rng(seed)
    grid = (_latin_hypercube(count - 1, len(sliders), rng) if count > 1
            else np.empty((0, len(sliders))))
    points = np.vstack([np.full((1, len(sliders)), 0.5), grid])  # centre first

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, point in enumerate(points):
        params = dict(query_base)
        params.update({f"s{j + 1}": f"{v:.3f}" for j, v in enumerate(point)})
        query = urllib.parse.urlencode(params)

        report = _get_json(f"{server}/evaluate?{query}")
        wav_path = os.path.join(out_dir, f"sweep_{i + 1:02d}.wav")
        with open(wav_path, "wb") as fh:
            fh.write(_get_bytes(f"{server}/generate?{query}"))
        report["sliders"] = {n: float(v) for n, v in zip(names, point)}
        report["wav"] = wav_path
        results.append(report)

        score = report.get("score", 0.0)
        color = "green" if score >= 70 else ("yellow" if score >= 50 else "red")
        worst = next((v["text"].split(" — ")[0] for v in report.get("verdicts", [])
                      if v["symbol"] == "✗"), "")
        slider_str = " ".join(f"{n}={v:.2f}" for n, v in report["sliders"].items())
        console.print(f"[{color}]{i + 1:2d}. {score:5.1f}  {slider_str}"
                      f"{'  ✗ ' + worst if worst else ''}[/{color}]")

    scored = [r for r in results if "score" in r]
    scores = np.array([r["score"] for r in scored])

    table = Table(title="Slider ↔ eval-score correlation", show_header=True)
    table.add_column("Slider")
    table.add_column("r", justify="right")
    for j, n in enumerate(names):
        vals = np.array([r["sliders"][n] for r in scored])
        r_val = (float(np.corrcoef(vals, scores)[0, 1])
                 if len(scored) > 2 and vals.std() > 0 else float("nan"))
        table.add_row(n, f"{r_val:+.2f}")
    console.print(table)

    summary = {
        "instrument": cfg.get("instrument"),
        "mean_score": float(scores.mean()) if len(scores) else 0.0,
        "pass_rate": float((scores >= 70).mean()) if len(scores) else 0.0,
        "min": float(scores.min()) if len(scores) else 0.0,
        "max": float(scores.max()) if len(scores) else 0.0,
    }
    console.print(
        f"\nSweep: mean {summary['mean_score']:.1f}/100, "
        f"pass rate {100 * summary['pass_rate']:.0f}%, "
        f"range [{summary['min']:.0f}, {summary['max']:.0f}] "
        f"({len(scored)}/{len(results)} evaluated)"
    )

    payload = {"summary": summary, "results": results}
    if json_out:
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        console.print(f"[dim]Wrote {json_out}[/dim]")
    return payload
