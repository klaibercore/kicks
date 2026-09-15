"""Automatic perceptual evaluation of generated drum hits.

Scores generated .wav files against a reference corpus and translates the
numbers into plain-English verdicts ("sounds like a real kick" / "high-end tail
is smeared"). Every metric is computed on the final waveform — post-vocoder —
because that is what the listener hears and where vocoder artefacts (HF smear,
hiss, metallic ringing) actually live.

"Good" is defined statistically rather than by fixed thresholds: a generated hit
passes when each metric falls inside the distribution of the same metric over
the real corpus. Per-metric robust z-scores become verdicts, a Mahalanobis
distance in metric space gives an overall likeness percentile, and a Fréchet
distance between the generated set and the corpus gives one set-level number.

Which metrics are scored, how heavily, and how a failure is worded all come from
the instrument profile — a hi-hat has no fundamental to check and no reason to
be penalised for a sustained high end.

numpy / scipy only, no torch import, so this starts fast.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.signal

from ..audio.constants import SAMPLE_RATE
from ..audio.waveform import (
    EPS,
    band_power,
    detect_onsets,
    load_audio,
    rms_envelope,
    stft_power,
)
from ..instruments import InstrumentProfile, get_profile

PASS_SCORE = 70.0


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def analyze_hit(x: np.ndarray, profile: InstrumentProfile) -> dict[str, float] | None:
    """Compute the profile's perceptual metrics for one waveform.

    Returns a dict of scalars (times in ms, ratios in dB) or None when the file
    contains no usable transient. Only metrics the profile actually scores are
    computed, so an unpitched instrument never pays for a pitch track.
    """
    w = profile.eval_windows
    wanted = set(profile.metric_keys)
    out: dict[str, float] = {}

    env = rms_envelope(x)
    peak = env.max()
    peak_i = int(env.argmax())

    # Onset: first time the envelope exceeds 10% of peak
    above = np.flatnonzero(env > 0.1 * peak)
    if len(above) == 0:
        return None
    start_i = int(above[0])

    attack_ms = max(0.0, (peak_i - start_i) / SAMPLE_RATE * 1000.0)

    # Decay: peak -> last time the envelope is above the floor, relative to peak
    thresh = peak * 10 ** (w.decay_floor_db / 20)
    tail_idx = np.flatnonzero(env[peak_i:] > thresh)
    decay_ms = (tail_idx[-1] if len(tail_idx) else 0) / SAMPLE_RATE * 1000.0

    out["attack_ms"] = float(attack_ms)
    out["decay_ms"] = float(decay_ms)
    out["n_onsets"] = float(max(1, len(detect_onsets(x, profile.onsets))))

    # Crest factor over the active region
    active = x[start_i: start_i + max(1, int(decay_ms / 1000 * SAMPLE_RATE) + 1)]
    rms = np.sqrt((active ** 2).mean() + EPS)
    out["crest_db"] = float(20 * np.log10(np.abs(x).max() / (rms + EPS)))

    # Tail noise floor: trailing window relative to full scale (vocoder hiss check)
    tail = x[-int(w.noise_floor_ms / 1000 * SAMPLE_RATE):]
    out["noise_floor_db"] = float(20 * np.log10(np.sqrt((tail ** 2).mean() + EPS) + EPS))

    # --- High-frequency behaviour ------------------------------------------
    hf_lo, hf_hi = profile.hf_band
    f, t, P = stft_power(x)
    t_ms = t * 1000.0
    hf_pow = band_power(P, f, hf_lo, hf_hi)
    hf_total = hf_pow.sum() + EPS

    # HF click presence: how much of the transient window is high-frequency energy
    early = t_ms < w.click_ms
    frame_pow = P.sum(axis=0) + EPS
    out["hf_click_db"] = float(10 * np.log10(
        (hf_pow[early].sum() + EPS) / (frame_pow[early].sum() + EPS)
    ))

    # HF decay: time for the HF envelope to fall 30 dB below its own peak
    hf_peak_i = int(hf_pow.argmax())
    hf_above = np.flatnonzero(hf_pow[hf_peak_i:] > hf_pow[hf_peak_i] * 1e-3)
    out["hf_decay_ms"] = float(
        t_ms[hf_peak_i + hf_above[-1]] - t_ms[hf_peak_i] if len(hf_above) else 0.0
    )

    # HF tail ratio: energy after the split point vs before
    split = t_ms >= w.tail_split_ms
    out["hf_tail_ratio_db"] = float(10 * np.log10(
        (hf_pow[split].sum() + EPS) / (hf_pow[~split].sum() + EPS)
    ))

    # Tail character in the HF band: spectral flatness.
    # High flatness = hiss; sustained HF with LOW flatness = metallic ringing.
    tf_lo, tf_hi = w.tail_flatness_ms
    hf_bins = (f >= hf_lo) & (f < hf_hi)
    tail_frames = (t_ms >= tf_lo) & (t_ms <= tf_hi)
    Ptail = P[np.ix_(hf_bins, tail_frames)]
    fw = Ptail.sum(axis=0)  # weight frames by their energy
    # Only judge the tail's character when it carries audible energy: below
    # -40 dB of the total HF energy the "spectrum" is fade and dither residue
    # and its flatness means nothing.
    if Ptail.size and fw.sum() > 1e-4 * hf_total:
        gmean = np.exp(np.log(Ptail + EPS).mean(axis=0))
        amean = Ptail.mean(axis=0) + EPS
        out["tail_flatness"] = float((gmean / amean * fw).sum() / fw.sum())
    else:
        out["tail_flatness"] = 0.5  # no tail energy -> neutral

    # --- Pitch (pitched instruments only) ----------------------------------
    if profile.pitched and {"sub_hz", "pitch_glide"} & wanted:
        f_lo, f_hi = profile.fundamental_band
        b_lo, b_hi = w.body_ms
        body = x[int(b_lo / 1000 * SAMPLE_RATE): int(b_hi / 1000 * SAMPLE_RATE)]
        win = scipy.signal.windows.hann(len(body))
        spec = np.abs(np.fft.rfft(body * win, n=1 << 16))
        freqs = np.fft.rfftfreq(1 << 16, 1 / SAMPLE_RATE)
        band = (freqs >= f_lo) & (freqs <= f_hi)
        out["sub_hz"] = float(freqs[band][int(spec[band].argmax())])

        # Pitch glide: drum fundamentals typically sweep downward. f0 early vs
        # late via a high-resolution STFT restricted to the fundamental band.
        fg, tg, Zg = scipy.signal.stft(
            x, fs=SAMPLE_RATE, nperseg=4096, noverlap=4096 - 512,
            boundary=None, padded=False,
        )
        Pg = np.abs(Zg) ** 2
        lo_bins = (fg >= f_lo) & (fg <= f_hi * 5 / 3)
        f0 = fg[lo_bins][Pg[lo_bins].argmax(axis=0)]
        tg_ms = tg * 1000.0
        e_lo, e_hi = w.glide_early_ms
        l_lo, l_hi = w.glide_late_ms
        e_sel = (tg_ms >= e_lo) & (tg_ms <= e_hi)
        l_sel = (tg_ms >= l_lo) & (tg_ms <= l_hi)
        if e_sel.any() and l_sel.any():
            out["pitch_glide"] = float(
                np.median(f0[e_sel]) / (np.median(f0[l_sel]) + EPS)
            )
        else:
            out["pitch_glide"] = 1.0

    # Centroid drop: spectral centroid in the transient vs the body
    if "centroid_drop" in wanted:
        centroid = (f[:, None] * P).sum(axis=0) / (P.sum(axis=0) + EPS)
        c_lo, c_hi = w.centroid_late_ms
        c_early = centroid[t_ms < w.click_ms]
        c_late = centroid[(t_ms >= c_lo) & (t_ms <= c_hi)]
        out["centroid_drop"] = float(
            np.log2((c_early.mean() + EPS) / (c_late.mean() + EPS))
        ) if len(c_early) and len(c_late) else 0.0

    missing = wanted - out.keys()
    if missing:
        raise RuntimeError(
            f"{profile.name} profile scores {sorted(missing)} but the analyzer "
            "does not produce them"
        )
    return {k: out[k] for k in profile.metric_keys}


# ---------------------------------------------------------------------------
# Reference corpus statistics
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """Corpus metric distributions used to judge generated hits."""

    values: dict[str, np.ndarray]        # sorted per-metric corpus values
    median: dict[str, float]
    sigma: dict[str, float]              # robust sigma (1.4826 * MAD)
    p5: dict[str, float]
    p95: dict[str, float]
    mean_vec: np.ndarray                 # standardized-space Gaussian fit
    cov: np.ndarray
    cov_inv: np.ndarray
    mahal_ref: np.ndarray                # corpus samples' own Mahalanobis d²
    n_files: int
    keys: list[str] = field(default_factory=list)       # multivariate keys
    log_keys: frozenset[str] = frozenset()              # scored in log space

    def to_score_space(self, key: str, v: float) -> float:
        return float(np.log1p(max(v, 0.0))) if key in self.log_keys else float(v)

    def transform(self, m: dict[str, float]) -> np.ndarray:
        """Metric dict -> standardized vector (log-space where specified).

        Z-scores are clipped to ±6 so a single degenerate metric (a corpus whose
        tail is exact digital silence, say) cannot dominate the multivariate
        Mahalanobis / Fréchet statistics.
        """
        return np.asarray([
            np.clip((self.to_score_space(k, m[k]) - self.median[k]) / self.sigma[k], -6.0, 6.0)
            for k in self.keys
        ])


def build_reference(
    ref_dir: str,
    profile: InstrumentProfile,
    n_samples: int = 400,
    cache_path: str | None = None,
    refresh: bool = False,
    progress=None,
) -> Reference:
    """Compute (or load cached) corpus metric distributions.

    The cache is fingerprinted on the corpus contents *and* the instrument, so
    switching instruments or changing the corpus invalidates it automatically.
    """
    cache_path = cache_path or profile.paths.eval_reference
    files = sorted(glob.glob(os.path.join(ref_dir, "*.wav")))
    if not files:
        raise RuntimeError(f"No .wav files in {ref_dir}")
    if n_samples and len(files) > n_samples:
        files = sorted(random.Random(42).sample(files, n_samples))

    fingerprint = hashlib.sha1(
        json.dumps([profile.name, ref_dir, len(files), files[:5], files[-5:]]).encode()
    ).hexdigest()[:12]

    if not refresh and os.path.exists(cache_path):
        with open(cache_path) as fh:
            cached = json.load(fh)
        if cached.get("fingerprint") == fingerprint:
            return reference_from_rows(
                {k: np.array(v) for k, v in cached["metrics"].items()},
                cached["n_files"], profile,
            )

    rows: dict[str, list[float]] = {}
    n_ok = 0
    for i, path in enumerate(files):
        x = load_audio(path)
        m = analyze_hit(x, profile) if x is not None else None
        if m is None:
            continue
        n_ok += 1
        for k, v in m.items():
            rows.setdefault(k, []).append(v)
        if progress and (i + 1) % 50 == 0:
            progress(i + 1, len(files))

    metrics = {k: np.array(v) for k, v in rows.items()}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(
            {"instrument": profile.name, "fingerprint": fingerprint, "n_files": n_ok,
             "metrics": {k: v.tolist() for k, v in metrics.items()}},
            fh,
        )
    return reference_from_rows(metrics, n_ok, profile)


def reference_from_rows(
    metrics: dict[str, np.ndarray], n_files: int, profile: InstrumentProfile,
) -> Reference:
    """Build a :class:`Reference` from raw per-metric corpus values."""
    scored = profile.scored_metrics
    stat_keys = [s.key for s in scored]
    # Metrics flagged non-multivariate are still reported as standalone verdicts
    # but kept out of the Mahalanobis / Fréchet statistics — the noise floor of a
    # corpus of digital-silence tails saturates that dimension and would mask
    # every real difference.
    keys = [s.key for s in scored if s.multivariate]
    log_keys = frozenset(s.key for s in profile.metrics if s.log)

    def to_score(key: str, v):
        return np.log1p(np.maximum(v, 0.0)) if key in log_keys else np.asarray(v, dtype=float)

    median, sigma, p5, p95, values = {}, {}, {}, {}, {}
    for k in stat_keys:
        v = metrics[k]
        s = to_score(k, v)
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        median[k] = med
        # Robust sigma with a std-based floor: when most of the corpus sits on
        # one exact value, MAD collapses to 0 and any deviation becomes an
        # infinite z-score.
        sigma[k] = max(1.4826 * mad, 0.25 * float(s.std()), 1e-3)
        values[k] = np.sort(v)
        p5[k] = float(np.percentile(v, 5))
        p95[k] = float(np.percentile(v, 95))

    X = np.stack([
        np.clip((to_score(k, metrics[k]) - median[k]) / sigma[k], -6.0, 6.0)
        for k in keys
    ], axis=1)
    mean_vec = X.mean(axis=0)
    cov = np.cov(X, rowvar=False) + 1e-3 * np.eye(len(keys))
    cov_inv = np.linalg.inv(cov)
    d = X - mean_vec
    mahal_ref = np.sort(np.einsum("ij,jk,ik->i", d, cov_inv, d))

    return Reference(
        values=values, median=median, sigma=sigma, p5=p5, p95=p95,
        mean_vec=mean_vec, cov=cov, cov_inv=cov_inv, mahal_ref=mahal_ref,
        n_files=n_files, keys=keys, log_keys=log_keys,
    )


# ---------------------------------------------------------------------------
# Scoring & verdict translation
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    key: str
    symbol: str      # "✓" | "⚠" | "✗"
    text: str
    value: float
    percentile: float
    z: float


@dataclass
class SampleReport:
    path: str
    score: float
    grade: str
    likeness_pct: float          # % of corpus more atypical than this sample
    verdicts: list[Verdict]
    metrics: dict[str, float]


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def grade(score: float, profile: InstrumentProfile) -> str:
    """One-line summary of a score, in the instrument's own words."""
    noun = profile.noun
    if score >= 85:
        return f"sounds like a real {noun}"
    if score >= PASS_SCORE:
        return f"{noun}-like, minor artefacts"
    if score >= 50:
        return f"recognizably a {noun}, audible problems"
    return f"does not pass as a {noun}"


def score_sample(
    path: str,
    metrics: dict[str, float],
    ref: Reference,
    profile: InstrumentProfile,
) -> SampleReport:
    """Score one sample against the corpus and phrase the verdicts."""
    verdicts: list[Verdict] = []
    total_w, total = 0.0, 0.0

    for spec in profile.scored_metrics:
        v = metrics[spec.key]
        sv = ref.to_score_space(spec.key, v)
        z = (sv - ref.median[spec.key]) / ref.sigma[spec.key]
        corpus = ref.values[spec.key]
        pct = 100.0 * np.searchsorted(corpus, v) / len(corpus)

        # One-sided leniency: an empty `low`/`high` hint means that direction is
        # not an artefact (a quieter noise floor than the corpus, say).
        z_eff = z
        if z < 0 and not spec.low:
            z_eff = 0.0
        if z > 0 and not spec.high:
            z_eff = 0.0

        # Extreme percentiles escalate even when the corpus spread is wide:
        # sitting past the 98th percentile of the real corpus is never a clean pass.
        if z_eff > 0 and pct >= 98.0 and spec.high:
            z_eff = max(z_eff, 2.5)
        if z_eff < 0 and pct <= 2.0 and spec.low:
            z_eff = min(z_eff, -2.5)

        sub = float(np.clip(1.0 - max(0.0, abs(z_eff) - 2.0) / 3.0, 0.0, 1.0))
        total += spec.weight * sub
        total_w += spec.weight

        if abs(z_eff) <= 2.0:
            sym, text = "✓", spec.good
        else:
            sym = "⚠" if abs(z_eff) <= 3.5 else "✗"
            text = spec.low if z_eff < 0 else spec.high
        detail = (
            f"{spec.label}: {spec.fmt.format(v)} "
            f"(corpus median {spec.fmt.format(corpus[len(corpus) // 2])}, "
            f"{_ordinal(pct)} pct)"
        )
        verdicts.append(Verdict(spec.key, sym, f"{text} — {detail}", v, pct, float(z)))

    score = 100.0 * total / total_w if total_w else 0.0

    # Hard gate: multiple onsets is an artefact whatever the corpus says.
    if "n_onsets" in metrics:
        n_onsets = int(metrics["n_onsets"])
        if n_onsets > 1:
            spec = profile.metric("n_onsets")
            score -= 20.0 * (n_onsets - 1)
            verdicts.append(Verdict(
                "n_onsets", "✗", f"{n_onsets} {spec.high}", float(n_onsets), 100.0, 10.0,
            ))
    score = float(np.clip(score, 0.0, 100.0))

    # Overall likeness: how typical is this sample versus the corpus itself
    d = ref.transform(metrics) - ref.mean_vec
    d2 = float(d @ ref.cov_inv @ d)
    likeness = 100.0 * (1.0 - np.searchsorted(ref.mahal_ref, d2) / len(ref.mahal_ref))

    # Worst problems first, passes last
    order = {"✗": 0, "⚠": 1, "✓": 2}
    verdicts.sort(key=lambda v: (order[v.symbol], -abs(v.z)))

    return SampleReport(
        path=path, score=score, grade=grade(score, profile),
        likeness_pct=float(likeness), verdicts=verdicts, metrics=metrics,
    )


def frechet_distance(reports: list[SampleReport], ref: Reference) -> float:
    """Fréchet distance between the generated set and the corpus in metric space.

    Both sets are standardized by corpus statistics; lower is better, and ~0
    means the generated distribution is indistinguishable from the real one.
    """
    if len(reports) < 2:
        return float("nan")
    X = np.stack([ref.transform(r.metrics) for r in reports])
    mu_g, cov_g = X.mean(axis=0), np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1])
    diff = mu_g - ref.mean_vec
    covmean = scipy.linalg.sqrtm(cov_g @ ref.cov)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_g + ref.cov - 2.0 * covmean))


def score_waveform(
    x: np.ndarray, ref: Reference, profile: InstrumentProfile, label: str = "sample",
) -> SampleReport | None:
    """Analyze and score a waveform in one call. None if it has no usable transient."""
    m = analyze_hit(x, profile)
    return None if m is None else score_sample(label, m, ref, profile)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def run_eval(
    generated: str,
    reference: str | None = None,
    instrument: str | None = None,
    pattern: str = "*.wav",
    ref_samples: int = 400,
    refresh_ref: bool = False,
    json_out: str | None = None,
    verbose: bool = True,
) -> dict:
    """Evaluate generated hits against the reference corpus."""
    from rich.console import Console
    from rich.table import Table

    profile = get_profile(instrument)
    reference = reference or profile.paths.data_dir
    console = Console()

    console.print(f"[dim]Building {profile.name} reference from {reference} "
                  f"(up to {ref_samples} samples)...[/dim]")
    ref = build_reference(
        reference, profile, n_samples=ref_samples, refresh=refresh_ref,
        progress=lambda i, n: console.print(f"[dim]  {i}/{n}[/dim]"),
    )
    console.print(f"[dim]Reference: {ref.n_files} corpus {profile.plural} analyzed[/dim]\n")

    paths = sorted(glob.glob(os.path.join(generated, pattern)))
    if not paths:
        raise RuntimeError(f"No files matching {pattern} in {generated}")

    reports: list[SampleReport] = []
    for path in paths:
        x = load_audio(path)
        m = analyze_hit(x, profile) if x is not None else None
        if m is None:
            console.print(f"[red]✗ {os.path.basename(path)}: unreadable or silent[/red]")
            continue
        reports.append(score_sample(path, m, ref, profile))

    for r in reports:
        color = "green" if r.score >= PASS_SCORE else ("yellow" if r.score >= 50 else "red")
        console.print(
            f"[bold {color}]{os.path.basename(r.path)}  "
            f"{r.score:.0f}/100 — {r.grade}[/bold {color}]  "
            f"[dim](more typical than {r.likeness_pct:.0f}% of corpus)[/dim]"
        )
        if verbose:
            for v in r.verdicts:
                style = {"✓": "green", "⚠": "yellow", "✗": "red"}[v.symbol]
                console.print(f"  [{style}]{v.symbol}[/{style}] {v.text}")
        console.print()

    fd = frechet_distance(reports, ref)
    mean_score = float(np.mean([r.score for r in reports])) if reports else 0.0
    n_pass = sum(1 for r in reports if r.score >= PASS_SCORE)

    table = Table(title=f"Set summary ({profile.display_name})", show_header=False)
    table.add_row("Files evaluated", str(len(reports)))
    table.add_row("Mean score", f"{mean_score:.1f}/100")
    table.add_row(f"Pass rate (≥{PASS_SCORE:.0f})", f"{n_pass}/{len(reports)}")
    table.add_row("Fréchet distance to corpus",
                  f"{fd:.2f}  (lower = more {profile.noun}-like)")
    console.print(table)

    result = {
        "instrument": profile.name,
        "mean_score": mean_score,
        "pass_rate": n_pass / max(1, len(reports)),
        "frechet_distance": fd,
        "files": [
            {
                "path": r.path,
                "score": r.score,
                "grade": r.grade,
                "likeness_pct": r.likeness_pct,
                "metrics": r.metrics,
                "verdicts": [
                    {"metric": v.key, "symbol": v.symbol, "text": v.text,
                     "percentile": v.percentile, "z": v.z}
                    for v in r.verdicts
                ],
            }
            for r in reports
        ],
    }
    if json_out:
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        console.print(f"[dim]Wrote {json_out}[/dim]")
    return result
