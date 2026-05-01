"""Strip non-kick content from drum loop WAVs.

Detects the kick hit via low-frequency envelope analysis, finds its natural
decay endpoint, then fades out and zeros everything after — leaving only
the isolated kick.
"""

import os
import shutil

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

TARGET_SR = 44100


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load WAV as mono float32 tensor at TARGET_SR. Returns (audio_1d, sr)."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        audio = torch.from_numpy(data).unsqueeze(0)
    else:
        audio = torch.from_numpy(data.T)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != TARGET_SR:
        audio = torchaudio.transforms.Resample(sr, TARGET_SR)(audio)
    return audio.squeeze(0), TARGET_SR


def _smooth_envelope(
    audio: torch.Tensor, sr: int, cutoff: float, order: int = 2, smooth_ms: float = 5.0
) -> np.ndarray:
    """Cascaded biquad filter -> abs -> moving-average smoothing.

    Args:
        audio: 1-D tensor.
        cutoff: Filter cutoff in Hz.
        order: Number of cascaded biquad passes.
        smooth_ms: Moving-average window in milliseconds.
    """
    x = audio.unsqueeze(0)  # (1, T) for biquad
    for _ in range(order):
        x = F.lowpass_biquad(x, sr, cutoff_freq=cutoff)
    env = x.squeeze(0).abs().numpy()
    # Moving average smoothing
    win = max(1, int(sr * smooth_ms / 1000))
    kernel = np.ones(win) / win
    env = np.convolve(env, kernel, mode="same")
    return env


def _hf_envelope(audio: torch.Tensor, sr: int, cutoff: float = 2000.0, smooth_ms: float = 5.0) -> np.ndarray:
    """High-frequency envelope for detecting hi-hat/snare onsets."""
    x = audio.unsqueeze(0)
    for _ in range(2):
        x = F.highpass_biquad(x, sr, cutoff_freq=cutoff)
    env = x.squeeze(0).abs().numpy()
    win = max(1, int(sr * smooth_ms / 1000))
    kernel = np.ones(win) / win
    env = np.convolve(env, kernel, mode="same")
    return env


def _detect_kick_region(
    audio: torch.Tensor,
    sr: int,
    threshold: float = 0.01,
    min_duration_ms: float = 50.0,
    max_duration_ms: float = 1200.0,
) -> tuple[int, int] | None:
    """Detect kick onset and decay endpoint.

    Returns (start_sample, end_sample) or None if no kick found.
    """
    lf_env = _smooth_envelope(audio, sr, cutoff=200.0)
    peak_idx = int(np.argmax(lf_env))
    peak_val = lf_env[peak_idx]

    if peak_val < 1e-6:
        return None  # No meaningful LF content

    # --- Onset: walk backward from peak to 5% of peak ---
    onset_thresh = 0.05 * peak_val
    onset = peak_idx
    while onset > 0 and lf_env[onset] > onset_thresh:
        onset -= 1
    onset = max(0, onset - 64)  # 64-sample guard

    # --- Decay end (method 1): LF threshold ---
    lf_end = peak_idx
    while lf_end < len(lf_env) - 1 and lf_env[lf_end] > threshold * peak_val:
        lf_end += 1

    # --- Decay end (method 2): HF onset detection ---
    hf_env = _hf_envelope(audio, sr)
    # Skip the kick's own click transient (~30ms after peak)
    skip_samples = int(sr * 0.030)
    search_start = peak_idx + skip_samples
    hf_end = len(audio)

    if search_start < len(hf_env):
        # Look for a sudden HF spike (hi-hat/snare)
        hf_after = hf_env[search_start:]
        if len(hf_after) > 0:
            hf_baseline = np.median(hf_after[:max(1, len(hf_after) // 4)])
            if hf_baseline > 0:
                # Spike = 5x the baseline
                spike_indices = np.where(hf_after > 5.0 * hf_baseline)[0]
                if len(spike_indices) > 0:
                    hf_end = search_start + int(spike_indices[0])

    # Take the earlier of the two endpoints
    end = min(lf_end, hf_end)

    # Clamp duration
    min_samples = int(sr * min_duration_ms / 1000)
    max_samples = int(sr * max_duration_ms / 1000)
    duration = end - onset
    if duration < min_samples:
        end = onset + min_samples
    elif duration > max_samples:
        end = onset + max_samples

    end = min(end, len(audio))
    return (onset, end)


def _is_loop(
    audio: torch.Tensor,
    sr: int,
    min_lag_ms: float = 100.0,
    acf_threshold: float = 0.3,
    frame_ms: float = 10.0,
) -> bool:
    """Detect cyclical/looping content via energy-envelope autocorrelation.

    A single kick decays monotonically — its autocorrelation falls off and
    stays low.  A loop has periodic energy peaks, producing strong ACF peaks
    at the loop period.

    Args:
        min_lag_ms: Ignore lags shorter than this (skips the kick body).
        acf_threshold: Normalized ACF peak above this → loop.
        frame_ms: RMS frame length for the energy envelope.
    """
    x = audio.numpy()
    # Compute RMS energy envelope in short frames
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(x) // frame_len
    if n_frames < 4:
        return False
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    envelope = np.sqrt(np.mean(frames ** 2, axis=1))

    # Zero-center for autocorrelation
    envelope = envelope - envelope.mean()
    norm = np.dot(envelope, envelope)
    if norm < 1e-10:
        return False

    # Normalized autocorrelation via FFT (much faster than np.correlate for long signals)
    n_fft = 1
    while n_fft < 2 * n_frames:
        n_fft *= 2
    spec = np.fft.rfft(envelope, n=n_fft)
    acf = np.fft.irfft(spec * np.conj(spec))[:n_frames]
    acf = acf / norm

    # Only look at lags past the kick body
    min_lag_frames = max(1, int(min_lag_ms / frame_ms))
    if min_lag_frames >= n_frames:
        return False
    acf_tail = acf[min_lag_frames:]

    return bool(np.max(acf_tail) > acf_threshold)


def _apply_fade_and_zero(audio: np.ndarray, end: int, fade_samples: int) -> np.ndarray:
    """Apply cosine fade-out at `end` and zero everything after."""
    out = audio.copy()
    fade_start = max(0, end - fade_samples)
    fade_len = end - fade_start
    if fade_len > 0:
        fade = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        out[fade_start:end] *= fade
    out[end:] = 0.0
    return out


def strip_file(
    path: str,
    threshold: float = 0.01,
    min_duration_ms: float = 50.0,
    max_duration_ms: float = 1200.0,
    fade_ms: float = 10.0,
    dry_run: bool = False,
    exclude_loops: bool = True,
) -> dict:
    """Process a single WAV file. Returns status dict."""
    audio, sr = _load_audio(path)
    n_samples = len(audio)

    # Too short to process
    min_samples = int(sr * min_duration_ms / 1000)
    if n_samples <= min_samples:
        return {"path": path, "status": "skip_short", "duration_ms": n_samples / sr * 1000}

    # Detect cyclical/loop content
    if exclude_loops and _is_loop(audio, sr):
        return {"path": path, "status": "loop", "duration_ms": n_samples / sr * 1000}

    region = _detect_kick_region(audio, sr, threshold, min_duration_ms, max_duration_ms)
    if region is None:
        return {"path": path, "status": "skip_no_kick", "duration_ms": n_samples / sr * 1000}

    onset, end = region

    # Check if already clean: post-endpoint audio is near-silent
    if end < n_samples:
        tail_rms = np.sqrt(np.mean(audio.numpy()[end:] ** 2))
        peak_rms = np.sqrt(np.mean(audio.numpy()[onset:end] ** 2))
        if peak_rms > 0 and tail_rms / peak_rms < 0.01:
            return {
                "path": path,
                "status": "skip_clean",
                "duration_ms": (end - onset) / sr * 1000,
            }

    kick_duration_ms = (end - onset) / sr * 1000
    fade_samples = int(sr * fade_ms / 1000)

    if not dry_run:
        audio_np = audio.numpy()
        processed = _apply_fade_and_zero(audio_np, end, fade_samples)
        # Keep from onset onward (preserve any silence before onset as leading zeros)
        sf.write(path, processed, sr, subtype="FLOAT")

    return {
        "path": path,
        "status": "stripped",
        "duration_ms": kick_duration_ms,
        "onset": onset,
        "end": end,
    }


def run_strip(
    data: str = "data/kicks",
    dry_run: bool = False,
    max_duration: float = 1200.0,
    min_duration: float = 50.0,
    threshold: float = 0.01,
    fade_ms: float = 10.0,
    backup: bool = True,
    exclude_loops: bool = True,
    move_loops: bool = False,
) -> None:
    """Strip non-kick content from all WAVs in a directory."""
    if not os.path.isdir(data):
        print(f"Directory not found: {data}")
        return

    wav_files = sorted(f for f in os.listdir(data) if f.lower().endswith(".wav"))
    if not wav_files:
        print(f"No WAV files found in {data}")
        return

    if backup and not dry_run:
        backup_dir = os.path.join(os.path.dirname(data.rstrip("/")), os.path.basename(data.rstrip("/")) + "_backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            for f in wav_files:
                shutil.copy2(os.path.join(data, f), os.path.join(backup_dir, f))
            print(f"Backed up {len(wav_files)} files to {backup_dir}/")
        else:
            print(f"Backup directory already exists: {backup_dir}/ — skipping backup")

    mode = "DRY RUN" if dry_run else "STRIP"
    print(f"\n[{mode}] Processing {len(wav_files)} files in {data}/\n")

    loops_dir = data.rstrip("/") + "_loops"
    counts = {"stripped": 0, "skip_clean": 0, "skip_short": 0, "skip_no_kick": 0, "loop": 0}
    durations = []
    loop_files: list[str] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Stripping kicks", total=len(wav_files))

        for filename in wav_files:
            path = os.path.join(data, filename)
            result = strip_file(
                path,
                threshold=threshold,
                min_duration_ms=min_duration,
                max_duration_ms=max_duration,
                fade_ms=fade_ms,
                dry_run=dry_run,
                exclude_loops=exclude_loops,
            )
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if result["status"] == "loop":
                loop_files.append(filename)
            elif result["status"] in ("stripped", "skip_clean") and result.get("duration_ms"):
                durations.append(result["duration_ms"])
            progress.update(task, advance=1)

    # Copy detected loops to reject directory (move only with --move-loops)
    if loop_files and not dry_run:
        os.makedirs(loops_dir, exist_ok=True)
        for f in loop_files:
            if move_loops:
                shutil.move(os.path.join(data, f), os.path.join(loops_dir, f))
            else:
                shutil.copy2(os.path.join(data, f), os.path.join(loops_dir, f))
        action = "Moved" if move_loops else "Copied"
        print(f"\n{action} {len(loop_files)} loops to {loops_dir}/")

    # Summary
    print(f"\nResults:")
    print(f"  Stripped:       {counts['stripped']}")
    print(f"  Already clean:  {counts['skip_clean']}")
    print(f"  Loops excluded: {counts['loop']}")
    print(f"  Too short:      {counts['skip_short']}")
    print(f"  No kick:        {counts['skip_no_kick']}")

    if durations:
        durations_arr = np.array(durations)
        print(f"\nKick durations (ms):")
        print(f"  Min:    {durations_arr.min():.0f}")
        print(f"  Median: {np.median(durations_arr):.0f}")
        print(f"  Max:    {durations_arr.max():.0f}")
        print(f"  Mean:   {durations_arr.mean():.0f}")
