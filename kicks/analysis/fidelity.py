"""Waveform fidelity report: matched references against rendered audio.

Stage 4 of ``docs/high-fidelity-generation.md``. The dashboard's live detail
KPIs measure the decoded *log-mel* before the vocoder; this module measures
what a listener gets — the vocoded waveform — against the same hit rendered two
other ways:

* **reference** — the corpus hit after the model's own preprocessing
  (``io.load_waveform``: mono, resampled, fitted, LUFS-normalised);
* **vocoder only** — the reference's *real* mel through the vocoder, which is
  the ceiling a perfect VAE could reach on this backend;
* **vae** — the VAE's posterior-mean reconstruction through the vocoder.

The difference between the last two is the VAE's own cost. Every candidate is
level-matched to the reference (equal RMS, one shared anti-clip gain) before
measurement, and the same level-matched files are written out as randomised
blind A/B pairs, whole-hit and HF-band, so listening happens on exactly what
was measured. Fresh generations from the latent prior get the reference-free
texture checks plus the corpus realism score.

The metric functions are numpy/scipy only and importable without torch; only
:func:`run_fidelity` renders audio and imports the model stack lazily.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.signal

from ..audio.constants import LOG_MEL_MAX, LOG_MEL_MIN, SAMPLE_RATE
from ..audio.waveform import rms_envelope
from ..instruments import InstrumentProfile

METHOD = ("512-point Hann STFT, 75% overlap, dBFS re a full-scale sine; errors over reference "
          "bins above -70 dBFS inside the profile's attack and body windows; candidates RMS-matched "
          "to the reference with one shared anti-clip gain; in-corpus reconstruction diagnostic, "
          "held out from training when the run's split is reproducible")

#: Frequency bands the acceptance criteria ask for separately, in Hz.
BANDS = {"2_8k": (2000.0, 8000.0), "8_16k": (8000.0, 16000.0)}
#: Onset is the first time the envelope reaches this level relative to its peak.
ONSET_DB = -20.0
#: The reference has "decayed" once its envelope stays below this, relative to peak.
DECAY_DB = -60.0
#: A gated, digitally silent tail is not "quieter" than this for the late-energy comparison.
LATE_FLOOR_DB = -80.0
#: dB per normalised-mel unit, for the pre-vocoder cross-check against the dashboard KPI.
MEL_DB_PER_UNIT = (LOG_MEL_MAX - LOG_MEL_MIN) * 20 / math.log(10)


# ---------------------------------------------------------------------------
# Metrics (numpy only)
# ---------------------------------------------------------------------------

def magnitude_db(x: np.ndarray, sr: int = SAMPLE_RATE, n_fft: int = 512, hop: int = 128):
    """(freqs_hz, times_ms, dBFS magnitude) with a full-scale sine at 0 dBFS."""
    f, t, z = scipy.signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                                boundary=None, padded=False)
    return f, t * 1000.0, 20 * np.log10(2 * np.abs(z) + 1e-10)


def band_window_error(ref_db, cand_db, f, t_ms, band, window, floor_db: float = -70.0):
    """(MAE dB, signed bias dB) over reference bins above ``floor_db`` in band × window."""
    mask = ((f >= band[0]) & (f < band[1]))[:, None] & ((t_ms >= window[0]) & (t_ms < window[1]))[None, :]
    mask &= ref_db > floor_db
    if not mask.any():
        return None, None
    diff = cand_db[mask] - ref_db[mask]
    return float(np.abs(diff).mean()), float(diff.mean())


def _silent(x: np.ndarray) -> bool:
    return x.size == 0 or float(np.abs(x).max()) <= 1e-6


def onset_ms(x: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = ONSET_DB) -> float | None:
    """First time the RMS envelope reaches ``threshold_db`` relative to its peak."""
    if _silent(x):
        return None
    env = rms_envelope(x, win=256)
    peak = env.max()
    hits = np.flatnonzero(env >= peak * 10 ** (threshold_db / 20))
    return float(hits[0] / sr * 1000.0) if hits.size else None


def decay_point_ms(x: np.ndarray, sr: int = SAMPLE_RATE, floor_db: float = DECAY_DB) -> float | None:
    """Last time the envelope is above ``floor_db`` relative to its peak."""
    if _silent(x):
        return None
    env = rms_envelope(x, win=512)
    peak = env.max()
    above = np.flatnonzero(env >= peak * 10 ** (floor_db / 20))
    return float(above[-1] / sr * 1000.0) if above.size else None


def envelope_error_db(ref: np.ndarray, cand: np.ndarray, floor_db: float = DECAY_DB) -> float | None:
    """Mean |level difference| of the RMS envelopes where the reference is audible."""
    if _silent(ref):
        return None
    ref_env, cand_env = rms_envelope(ref, win=512), rms_envelope(cand, win=512)
    peak = ref_env.max()
    active = ref_env >= peak * 10 ** (floor_db / 20)
    if not active.any():
        return None
    return float(np.abs(20 * np.log10((cand_env[active] + 1e-10) / (ref_env[active] + 1e-10))).mean())


def spectral_flatness(x: np.ndarray, sr: int = SAMPLE_RATE, band=(2000.0, 16000.0),
                      floor_db: float = DECAY_DB) -> float | None:
    """Mean per-frame flatness (geometric / arithmetic mean power) in ``band``, 0..1.

    Noise sits near 1, a ringing partial near 0. Averaged over frames where the
    hit is still audible so the silent tail does not vote.
    """
    f, t, z = scipy.signal.stft(x, fs=sr, nperseg=512, noverlap=384, boundary=None, padded=False)
    power = np.abs(z[(f >= band[0]) & (f < band[1])]) ** 2 + 1e-14
    frame_level = 10 * np.log10(power.sum(axis=0))
    active = frame_level >= frame_level.max() + floor_db
    if not active.any():
        return None
    flatness = np.exp(np.log(power).mean(axis=0)) / power.mean(axis=0)
    return float(flatness[active].mean())


def late_energy_db(x: np.ndarray, after_ms: float, sr: int = SAMPLE_RATE) -> float:
    """Energy after ``after_ms`` relative to the whole hit's energy, in dB (≤ 0)."""
    start = min(len(x), int(after_ms / 1000.0 * sr))
    total = float(np.sum(x ** 2)) + 1e-12
    return float(10 * np.log10((np.sum(x[start:] ** 2) + 1e-12) / total))


def fit_to(x: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad ``x`` to ``length`` samples (vocoders round to whole frames)."""
    if len(x) >= length:
        return x[:length]
    return np.pad(x, (0, length - len(x)))


def level_match(reference: np.ndarray, *candidates: np.ndarray, peak: float = 0.9):
    """Reference at ``peak``, candidates at the reference's RMS and length, one shared anti-clip gain.

    Returns the reference followed by the matched candidates. One common gain
    keeps the relative levels intact, which is the point of level matching.
    """
    ref = reference / (np.abs(reference).max() + 1e-12) * peak
    ref_rms = math.sqrt(float(np.mean(ref ** 2)) + 1e-18)
    matched = []
    for cand in candidates:
        cand = fit_to(cand, len(ref))
        rms = math.sqrt(float(np.mean(cand ** 2)) + 1e-18)
        matched.append(cand * (ref_rms / rms))
    highest = max(np.abs(ref).max(), *(np.abs(c).max() for c in matched)) if matched else np.abs(ref).max()
    gain = min(1.0, 0.99 / (highest + 1e-12))
    return [ref * gain, *(c * gain for c in matched)]


def compare_waveforms(reference: np.ndarray, candidate: np.ndarray, profile: InstrumentProfile,
                      sr: int = SAMPLE_RATE) -> dict[str, float | None]:
    """The acceptance-criteria metrics for one level-matched (reference, candidate) pair."""
    windows = profile.eval_windows
    attack = (0.0, windows.click_ms)
    body = (windows.click_ms, windows.body_ms[1])
    candidate = fit_to(candidate, len(reference))
    f, t_ms, ref_db = magnitude_db(reference, sr)
    _, _, cand_db = magnitude_db(candidate, sr)
    row: dict[str, float | None] = {}
    for band_name, band in BANDS.items():
        for window_name, window in (("attack", attack), ("body", body)):
            mae, bias = band_window_error(ref_db, cand_db, f, t_ms, band, window)
            row[f"{band_name}_{window_name}_mae_db"] = mae
            row[f"{band_name}_{window_name}_bias_db"] = bias
    ref_onset, cand_onset = onset_ms(reference, sr), onset_ms(candidate, sr)
    row["onset_error_ms"] = None if ref_onset is None or cand_onset is None else cand_onset - ref_onset
    row["envelope_mae_db"] = envelope_error_db(reference, candidate)
    ref_flat, cand_flat = spectral_flatness(reference, sr), spectral_flatness(candidate, sr)
    row["flatness_delta"] = None if ref_flat is None or cand_flat is None else cand_flat - ref_flat
    decayed = decay_point_ms(reference, sr)
    row["late_excess_db"] = (None if decayed is None
                             else max(late_energy_db(candidate, decayed, sr), LATE_FLOOR_DB)
                             - max(late_energy_db(reference, decayed, sr), LATE_FLOOR_DB))
    return row


def texture_metrics(x: np.ndarray, profile: InstrumentProfile, sr: int = SAMPLE_RATE) -> dict:
    """Reference-free checks for a fresh generation."""
    decayed = decay_point_ms(x, sr)
    return {"onset_ms": onset_ms(x, sr), "decay_ms": decayed,
            "flatness": spectral_flatness(x, sr),
            "late_energy_db": late_energy_db(x, profile.eval_windows.noise_floor_ms, sr)}


def summarise(rows: list[dict], keys: tuple[str, ...]) -> dict[str, float | None]:
    """Mean of each key over rows where it was measurable."""
    out = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Rendering (imports torch lazily)
# ---------------------------------------------------------------------------

def _validation_indices(dataset, seed: int | None, val_split: float | None,
                        fingerprint: str | None = None) -> list[int] | None:
    """Reproduce the trainer's validation split (same seed, ratio and sorted listing).

    With ``fingerprint`` (from a run record) the corpus must still match it;
    without one the caller vouches that the corpus is the one that was trained on.
    """
    import torch
    from torch.utils.data import random_split

    from ..training.tracking import split_fingerprint

    if seed is None or val_split is None:
        return None
    if fingerprint is not None and split_fingerprint(dataset, seed, val_split) != fingerprint:
        return None
    n_val = max(1, min(len(dataset) - 1, int(len(dataset) * val_split)))
    _, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                              generator=torch.Generator().manual_seed(seed))
    return sorted(val_set.indices)


def _write_listening_page(out: Path, pairs: list[dict]) -> None:
    rows = "\n".join(
        f'<li><b>Pair {p["pair"]:02d}</b>'
        f' <label>A <audio controls preload="none" src="{p["a"]}"></audio></label>'
        f' <label>B <audio controls preload="none" src="{p["b"]}"></audio></label>'
        f' <label>HF · A <audio controls preload="none" src="{p["a_hf"]}"></audio></label>'
        f' <label>HF · B <audio controls preload="none" src="{p["b_hf"]}"></audio></label>'
        f' <label>reject <input type="checkbox" data-pair="{p["pair"]}"></label>'
        f' <span class="key" hidden>A = {p["a_is"]}, B = {p["b_is"]}</span></li>'
        for p in pairs)
    page = f"""<!doctype html><meta charset="utf-8"><title>Blind A/B · listening</title>
<style>body{{font:14px/1.5 system-ui;margin:24px;max-width:960px}}li{{margin:12px 0;display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}}audio{{height:30px;width:190px}}.key{{color:#a33}}</style>
<h1>Level-matched blind pairs</h1>
<p>One of each pair is the reference recording, the other the VAE reconstruction through the vocoder, at equal RMS.
Whole hit and 2 kHz+ band. Tick <em>reject</em> where the candidate is audibly worse, then reveal.
Higher brightness alone is not an improvement.</p>
<button id="reveal">Reveal key</button> <span id="count"></span>
<ol>{rows}</ol>
<script>
const boxes=[...document.querySelectorAll('input[data-pair]')];const count=()=>document.getElementById('count').textContent=`${{boxes.filter(b=>b.checked).length}} / ${{boxes.length}} rejected`;boxes.forEach(b=>b.onchange=count);count();
document.getElementById('reveal').onclick=()=>document.querySelectorAll('.key').forEach(k=>k.hidden=false);
</script>"""
    (out / "listening.html").write_text(page)


def run_fidelity(
    instrument: str | None = None,
    checkpoint: str | None = None,
    data: str | None = None,
    out_dir: str | None = None,
    count: int = 16,
    generate: int = 8,
    vocoder: str | None = None,
    run_id: str | None = None,
    runs_dir: str | None = None,
    seed: int = 20260916,
    note: str = "",
    split_seed: int | None = None,
    val_split: float = 0.1,
) -> dict:
    """Render, measure and write the fidelity report; attach it to a run when asked.

    Hits come from the run's held-out validation split when ``run_id`` is given
    and the corpus still matches its fingerprint; from the split ``split_seed`` /
    ``val_split`` would have produced for a model trained before tracking
    existed; otherwise from the whole corpus.
    """
    import soundfile as sf
    import torch

    from ..analysis.evaluation import analyze_hit, build_reference, score_sample
    from ..audio import io
    from ..audio.vocoder import load_vocoder, resolve_vocoder_type, spec_to_audio
    from ..audio.waveform import bandpass
    from ..config import get_device, load_vae_from_checkpoint
    from ..data import DrumDataset
    from ..instruments import get_profile
    from ..synthesis.generator import fit_latent_prior
    from ..training.tracking import attach_report, find_run, read_run, runs_root

    profile = get_profile(instrument)
    checkpoint = checkpoint or profile.paths.checkpoint
    out = Path(out_dir or os.path.join(profile.paths.output_dir, "fidelity",
                                       datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")))
    for sub in ("reconstruction", "listening", "generation"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, ckpt = load_vae_from_checkpoint(checkpoint, device)
    data = data or profile.paths.data_dir
    dataset = DrumDataset(data, profile, n_frames=model.n_frames)

    run_dir = run = None
    if run_id:
        run_dir = find_run(runs_root(runs_dir, profile.paths.output_root), run_id)
        run = read_run(run_dir)
    if run:
        held_out = _validation_indices(dataset, run["config"].get("seed"), run["config"].get("val_split"),
                                       run["config"].get("split_fingerprint") or "missing")
        if held_out is None:
            print("Run split not reproducible on this corpus (data changed?) — sampling the whole corpus instead", flush=True)
    else:
        held_out = _validation_indices(dataset, split_seed, val_split)
    pool = held_out if held_out is not None else list(range(len(dataset)))
    rng = random.Random(seed)
    indices = sorted(rng.sample(pool, min(count, len(pool))))

    vocoder_type = resolve_vocoder_type(profile, vocoder)
    voc = load_vocoder(device, vocoder_type, weights_dir=profile.paths.vocoder_dir)
    meter = io.make_meter()
    hf_band = (2000.0, None)

    origin = ("the run’s validation split" if run and held_out is not None
              else f"the seed-{split_seed} validation split" if held_out is not None else "the whole corpus")
    print(f"Rendering {len(indices)} {profile.plural} with {vocoder_type} from {origin}...", flush=True)
    samples, pairs = [], []
    started = time.monotonic()
    for pair_index, idx in enumerate(indices, 1):
        path = dataset.paths[idx]
        reference = io.load_waveform(path, meter, length=model.n_frames * 256)
        if reference is None:
            continue
        spec = dataset[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = model.encode(spec)
            decoded = model.decode(mu)
            vocoder_only = spec_to_audio(spec, voc, device)[0].numpy()
            vae = spec_to_audio(decoded, voc, device)[0].numpy()
        # Pre-vocoder KPI on the same hit, for cross-reference with the dashboard.
        active = (spec > spec.amax() - 60 / MEL_DB_PER_UNIT) & (spec > 1e-6)
        mel_mae_db = float(((decoded - spec).abs() * MEL_DB_PER_UNIT)[active].mean()) if active.any() else None

        ref, voc_m, vae_m = level_match(reference[0].numpy(), vocoder_only, vae)
        row = {"index": idx, "pair": pair_index, "duration_ms": float(len(ref) / SAMPLE_RATE * 1000),
               "mel_mae_db": mel_mae_db,
               "vocoder_only": compare_waveforms(ref, voc_m, profile),
               "vae": compare_waveforms(ref, vae_m, profile)}
        samples.append(row)

        stem = out / "reconstruction" / f"{pair_index:02d}"
        for name, audio in (("reference", ref), ("vocoder", voc_m), ("vae", vae_m)):
            sf.write(f"{stem}_{name}.wav", audio, SAMPLE_RATE, subtype="PCM_24")
        # Blind pair: reference vs VAE, order randomised per pair; HF band alongside.
        a_is, b_is = rng.sample(["reference", "vae"], 2)
        chosen = {"reference": ref, "vae": vae_m}
        for label, kind in (("A", a_is), ("B", b_is)):
            sf.write(out / "listening" / f"pair_{pair_index:02d}_{label}.wav", chosen[kind], SAMPLE_RATE, subtype="PCM_24")
            sf.write(out / "listening" / f"pair_{pair_index:02d}_{label}_hf.wav",
                     bandpass(chosen[kind], hf_band), SAMPLE_RATE, subtype="PCM_24")
        pairs.append({"pair": pair_index, "a_is": a_is, "b_is": b_is,
                      "a": f"pair_{pair_index:02d}_A.wav", "b": f"pair_{pair_index:02d}_B.wav",
                      "a_hf": f"pair_{pair_index:02d}_A_hf.wav", "b_hf": f"pair_{pair_index:02d}_B_hf.wav"})
        print(f"  {pair_index:02d}: 8–16 kHz body MAE vocoder {row['vocoder_only']['8_16k_body_mae_db'] or float('nan'):.2f} dB"
              f" · vae {row['vae']['8_16k_body_mae_db'] or float('nan'):.2f} dB", flush=True)
    (out / "listening" / "key.json").write_text(json.dumps(pairs, indent=2))
    _write_listening_page(out / "listening", pairs)

    generations = []
    if generate > 0:
        print(f"Generating {generate} fresh {profile.plural} from the latent prior...", flush=True)
        prior = fit_latent_prior(model, device, data, profile.paths.latent_prior)
        prior.random_state = np.random.RandomState(seed)
        latents, _ = prior.sample(generate)
        corpus_reference = build_reference(data, profile)
        for i, z in enumerate(latents, 1):
            with torch.no_grad():
                spec = model.decode(torch.tensor(z[None], dtype=torch.float32, device=device))
                audio = spec_to_audio(spec, voc, device)[0].numpy()
            name = f"gen_{i:02d}.wav"
            sf.write(out / "generation" / name, audio, SAMPLE_RATE, subtype="PCM_24")
            metrics = analyze_hit(audio, profile)
            scored = score_sample(name, metrics, corpus_reference, profile) if metrics else None
            generations.append({"file": name, "score": scored.score if scored else None,
                                **texture_metrics(audio, profile)})

    keys = tuple(compare_waveforms(np.zeros(1024), np.zeros(1024), profile).keys())
    vocoder_summary = summarise([s["vocoder_only"] for s in samples], keys)
    vae_summary = summarise([s["vae"] for s in samples], keys)
    penalty = {k: (None if vae_summary[k] is None or vocoder_summary[k] is None
                   else vae_summary[k] - vocoder_summary[k]) for k in keys if k.endswith("_mae_db")}
    scores = [g["score"] for g in generations if g["score"] is not None]
    summary = {
        "samples": len(samples), "held_out": held_out is not None, "vocoder": vocoder_type,
        "mel_mae_db": summarise(samples, ("mel_mae_db",))["mel_mae_db"],
        "vae_8_16k_body_mae_db": vae_summary["8_16k_body_mae_db"],
        "vae_2_8k_body_mae_db": vae_summary["2_8k_body_mae_db"],
        "vae_8_16k_attack_mae_db": vae_summary["8_16k_attack_mae_db"],
        "vocoder_8_16k_body_mae_db": vocoder_summary["8_16k_body_mae_db"],
        "vae_penalty_8_16k_body_db": penalty["8_16k_body_mae_db"],
        "vae_onset_error_ms": vae_summary["onset_error_ms"],
        "vae_envelope_mae_db": vae_summary["envelope_mae_db"],
        "vae_late_excess_db": vae_summary["late_excess_db"],
        "generation_mean_score": float(np.mean(scores)) if scores else None,
        "generation_pass_rate": float(np.mean([s >= 70 for s in scores])) if scores else None,
        "seconds": time.monotonic() - started,
    }
    report = {
        "instrument": profile.name, "corpus": str(Path(data).resolve()), "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _sha256(checkpoint), "epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss"),
        "architecture": getattr(model, "architecture", None), "vocoder": vocoder_type,
        "run_id": run["id"] if run else None, "held_out": held_out is not None, "hit_origin": origin,
        "hits": [os.path.basename(dataset.paths[i]) for i in indices],
        "windows": {"attack_ms": [0.0, profile.eval_windows.click_ms],
                    "body_ms": [profile.eval_windows.click_ms, profile.eval_windows.body_ms[1]]},
        "method": METHOD, "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "vocoder_only": vocoder_summary, "vae": vae_summary, "vae_penalty": penalty,
        "samples": samples, "generation": generations,
        "listening": {"page": str((out / "listening" / "listening.html").resolve()), "pairs": len(pairs)},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Report: {out / 'report.json'}\nListening: {out / 'listening' / 'listening.html'}", flush=True)

    if run_dir is not None:
        attach_report(run_dir, "fidelity", out / "report.json", summary,
                      note or f"{vocoder_type} · {len(samples)} {'held-out' if held_out is not None else 'corpus'} hits"
                              f" · epoch {ckpt.get('epoch')}")
        print(f"Attached to run {run['id']}", flush=True)
    return report
