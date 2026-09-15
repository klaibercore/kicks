"""Reconstruct identical corpus hits with each vocoder and write a local A/B page.

Metrics compare final audio to the same bandlimited, peak-normalized reference.
They measure reconstruction fidelity, independently of corpus realism scores.
Optional DisCoder input gains allow calibration without changing VAE training.
"""
from __future__ import annotations

import argparse
import gc
import html
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

from kicks.audio.constants import LOG_MEL_MAX, LOG_MEL_MIN, SAMPLE_RATE
from kicks.audio.io import list_wavs, load_waveform, make_meter
from kicks.audio.mel import log_mel, spectrogram
from kicks.audio.vocoder import _post_process, load_vocoder, spec_to_audio
from kicks.config import get_device, load_vae_from_checkpoint
from kicks.instruments import get_profile


def fidelity(reference, reconstructed):
    """Magnitude/envelope metrics: no penalty for an arbitrary carrier phase."""
    spectral = []
    for fft in (512, 1024, 2048):
        window = torch.hann_window(fft)
        target = torch.stft(reference, fft, fft // 4, window=window, return_complex=True).abs()
        actual = torch.stft(reconstructed, fft, fft // 4, window=window, return_complex=True).abs()
        spectral.append(float((target - actual).norm() / target.norm().clamp_min(1e-8)))
    target_mel = log_mel(reference[None])[0]
    actual_mel = log_mel(reconstructed[None])[0]
    active = target_mel > target_mel.max() + np.log(0.001)
    active_db = float((target_mel[active] - actual_mel[active]).abs().mean() * (20 / np.log(10)))
    def envelope(x):
        return torch.nn.functional.avg_pool1d(x[None, None].square(), 256, 128).sqrt()[0, 0]
    target_env, actual_env = envelope(reference), envelope(reconstructed)
    envelope_error = float((target_env - actual_env).norm() / target_env.norm().clamp_min(1e-8))
    attack = int(0.03 * SAMPLE_RATE)
    attack_energy_db = float(10 * torch.log10(
        reconstructed[:attack].square().sum().clamp_min(1e-12)
        / reference[:attack].square().sum().clamp_min(1e-12)))
    return {"spectral_convergence": float(np.mean(spectral)), "active_mel_mae_db": active_db,
            "envelope_relative_error": envelope_error, "attack_energy_error_db": attack_energy_db}


def write_report(out, rows, sources, args):
    groups = sorted({row["variant"] for row in rows})
    keys = ("spectral_convergence", "active_mel_mae_db", "envelope_relative_error",
            "attack_energy_error_db", "seconds")
    summary = {group: {key: float(np.mean([abs(r[key]) if key == "attack_energy_error_db" else r[key]
                                        for r in rows if r["variant"] == group]))
                       for key in keys} for group in groups}
    payload = {"instrument": args.instrument, "seed": args.seed, "checkpoint": args.checkpoint,
               "sources": sources, "summary": summary, "results": rows}
    (out / "report.json").write_text(json.dumps(payload, indent=2))
    cards = []
    for source in sources:
        index = source["index"]
        players = [f'<div><b>Original</b><audio controls src="{index:03d}_original.wav"></audio></div>']
        for row in rows:
            if row["index"] == index:
                players.append(f'<div><b>{html.escape(row["variant"])}</b>'
                               f'<audio controls src="{row["file"]}"></audio>'
                               f'<small>Mel error {row["active_mel_mae_db"]:.2f} dB · '
                               f'Spectral error {row["spectral_convergence"]:.3f}</small></div>')
        cards.append(f'<section><h2>Sample {index + 1}</h2><div class="players">{"".join(players)}</div></section>')
    (out / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Drum reconstruction comparison</title>
<style>body{font:16px system-ui;background:#141414;color:#eee;max-width:1200px;margin:auto;padding:28px}
h1{font-size:32px}p,small{color:#aaa}section{border-top:1px solid #444;padding:18px 0}
.players{display:flex;flex-wrap:wrap;gap:20px}.players>div{display:grid;gap:10px}audio{width:270px}</style>
<h1>Drum reconstruction comparison</h1><p>Same input hits, peak matched, 24-bit / 44.1 kHz.
Lower reconstruction errors mean closer magnitude and envelope recovery; listening remains essential.
“VAE” variants also include the generative model’s reconstruction error.</p>'''
        + ''.join(cards) + '''<script>document.addEventListener('play',e=>{document.querySelectorAll('audio')
.forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>''')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="kick")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocoders", nargs="+", choices=("bigvgan", "discoder", "griffinlim"),
                        default=["bigvgan", "discoder"])
    parser.add_argument("--discoder-gains", nargs="+", type=float, default=[0])
    parser.add_argument("--checkpoint", help="Also compare VAE posterior-mean reconstructions")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    profile, device = get_profile(args.instrument), get_device()
    files = list_wavs(profile.paths.data_dir)
    selected = np.random.default_rng(args.seed).choice(len(files), size=min(args.count, len(files)), replace=False)
    sources, inputs, references = [], [], []
    meter = make_meter()
    for index, selected_index in enumerate(selected):
        path = Path(profile.paths.data_dir) / files[selected_index]
        waveform = load_waveform(str(path), meter)
        if waveform is None:
            raise RuntimeError(f"Cannot read {path}")
        inputs.append(spectrogram(waveform)[None])
        reference = _post_process(waveform)[0]
        references.append(reference)
        sf.write(args.out / f"{index:03d}_original.wav", reference.numpy(), SAMPLE_RATE, subtype="PCM_24")
        sources.append({"index": index, "path": str(path)})
    vae_inputs = []
    if args.checkpoint:
        vae, _ = load_vae_from_checkpoint(args.checkpoint, device)
        with torch.inference_mode():
            for spec in inputs:
                mean, _ = vae.encode(spec.to(device))
                vae_inputs.append(vae.decode(mean).cpu())
        del vae
    rows = []
    for backend in args.vocoders:
        vocoder = load_vocoder(device, backend, weights_dir=profile.paths.vocoder_dir)
        gains = args.discoder_gains if backend == "discoder" else [0]
        for gain in gains:
            for mode, specs in (("mel", inputs), ("vae", vae_inputs)):
                for index, (spec, reference) in enumerate(zip(specs, references)):
                    # A dB amplitude shift adds a constant in natural-log mel space.
                    shifted = (spec + gain * np.log(10) / 20 / (LOG_MEL_MAX - LOG_MEL_MIN)).clamp(0, 1)
                    start = time.monotonic()
                    audio = spec_to_audio(shifted, vocoder, device)[0]
                    elapsed = time.monotonic() - start
                    variant = f"{backend}_{mode}_{gain:g}dB"
                    filename = f"{index:03d}_{variant}.wav"
                    sf.write(args.out / filename, audio.numpy(), SAMPLE_RATE, subtype="PCM_24")
                    row = {"index": index, "variant": variant, "file": filename,
                           **fidelity(reference, audio), "seconds": elapsed}
                    rows.append(row)
                    print(index, variant, f'mel={row["active_mel_mae_db"]:.3f}dB', flush=True)
                    summary = write_report(args.out, rows, sources, args)
        del vocoder
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
