"""Isolate a single hit inside a longer file.

Sample packs are full of one-shots with a hi-hat bleeding in 200 ms later, or
whole loop bars filed as one-shots. Both teach the VAE the wrong thing — a model
trained on two-hit files faithfully generates two hits. This detects the hit via
an envelope in the instrument's own band, finds its natural decay endpoint,
fades out and zeros everything after.

Which band to listen to, how long a hit may last, and where a *different*
instrument would show up all come from the profile: a kick is found under
200 Hz and interrupted by a hat above 2 kHz, a hat is found above 3 kHz and
interrupted by a kick underneath.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import soundfile as sf
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from ..audio.constants import SAMPLE_RATE
from ..audio.waveform import apply_fade_out, band_envelope, is_loop, load_audio
from ..instruments import InstrumentProfile, StripSpec, get_profile


def detect_hit_region(
    x: np.ndarray,
    strip: StripSpec,
    sr: int = SAMPLE_RATE,
) -> tuple[int, int] | None:
    """Find (start_sample, end_sample) of the hit, or None if there isn't one.

    The endpoint is the earlier of two answers: where the instrument's own
    envelope decays below threshold, and where another instrument enters. Taking
    the earlier one is deliberate — a truncated tail is a much smaller training
    error than a second instrument in the sample.
    """
    env = band_envelope(x, strip.onset_band, sr)
    peak_idx = int(np.argmax(env))
    peak_val = env[peak_idx]
    if peak_val < 1e-6:
        return None

    # Onset: walk backward from the peak to 5% of it, plus a small guard.
    onset_thresh = 0.05 * peak_val
    onset = peak_idx
    while onset > 0 and env[onset] > onset_thresh:
        onset -= 1
    onset = max(0, onset - 64)

    # Endpoint 1: the instrument's own decay.
    end = peak_idx
    while end < len(env) - 1 and env[end] > strip.threshold * peak_val:
        end += 1

    # Endpoint 2: another instrument entering.
    if strip.interrupt_band is not None:
        other = band_envelope(x, strip.interrupt_band, sr)
        # Skip the hit's own transient, which is broadband and would trip this.
        search_start = peak_idx + int(sr * strip.transient_skip_ms / 1000)
        if search_start < len(other):
            after = other[search_start:]
            baseline = np.median(after[: max(1, len(after) // 4)])
            if baseline > 0:
                spikes = np.flatnonzero(after > strip.interrupt_ratio * baseline)
                if len(spikes):
                    end = min(end, search_start + int(spikes[0]))

    min_samples = int(sr * strip.min_duration_ms / 1000)
    max_samples = int(sr * strip.max_duration_ms / 1000)
    duration = end - onset
    if duration < min_samples:
        end = onset + min_samples
    elif duration > max_samples:
        end = onset + max_samples

    return onset, min(end, len(x))


def strip_file(
    path: str,
    profile: InstrumentProfile,
    dry_run: bool = False,
    exclude_loops: bool = True,
) -> dict:
    """Process one WAV in place. Returns a status dict."""
    strip = profile.strip
    x = load_audio(path, length=0, normalize=False)
    if x is None:
        return {"path": path, "status": "unreadable", "duration_ms": 0.0}
    n_samples = len(x)
    duration_ms = n_samples / SAMPLE_RATE * 1000

    if n_samples <= int(SAMPLE_RATE * strip.min_duration_ms / 1000):
        return {"path": path, "status": "skip_short", "duration_ms": duration_ms}

    if exclude_loops and is_loop(
        x, SAMPLE_RATE,
        min_lag_ms=strip.loop_min_lag_ms,
        acf_threshold=strip.loop_acf_threshold,
    ):
        return {"path": path, "status": "loop", "duration_ms": duration_ms}

    region = detect_hit_region(x, strip)
    if region is None:
        return {"path": path, "status": "skip_no_hit", "duration_ms": duration_ms}
    onset, end = region

    # Already clean: whatever follows the endpoint is effectively silent.
    if end < n_samples:
        tail_rms = np.sqrt(np.mean(x[end:] ** 2))
        body_rms = np.sqrt(np.mean(x[onset:end] ** 2))
        if body_rms > 0 and tail_rms / body_rms < 0.01:
            return {"path": path, "status": "skip_clean",
                    "duration_ms": (end - onset) / SAMPLE_RATE * 1000}

    if not dry_run:
        fade_samples = int(SAMPLE_RATE * strip.fade_ms / 1000)
        # Keep everything before the onset as leading silence rather than
        # trimming it — the dataset pads to a fixed length anyway, and shifting
        # the onset would change the transient's frame alignment.
        sf.write(path, apply_fade_out(x, end, fade_samples), SAMPLE_RATE, subtype="FLOAT")

    return {
        "path": path,
        "status": "stripped",
        "duration_ms": (end - onset) / SAMPLE_RATE * 1000,
        "onset": onset,
        "end": end,
    }


def run_strip(
    data: str | None = None,
    instrument: str | None = None,
    dry_run: bool = False,
    backup: bool = True,
    exclude_loops: bool = True,
    move_loops: bool = False,
) -> dict:
    """Strip every WAV in a directory down to a single hit."""
    profile: InstrumentProfile = get_profile(instrument)
    data = data or profile.paths.data_dir
    if not os.path.isdir(data):
        print(f"Directory not found: {data}")
        return {}

    wav_files = sorted(f for f in os.listdir(data) if f.lower().endswith(".wav"))
    if not wav_files:
        print(f"No WAV files found in {data}")
        return {}

    if backup and not dry_run:
        backup_dir = data.rstrip("/") + "_backup"
        if os.path.exists(backup_dir):
            print(f"Backup directory already exists: {backup_dir}/ — skipping backup")
        else:
            os.makedirs(backup_dir)
            for f in wav_files:
                shutil.copy2(os.path.join(data, f), os.path.join(backup_dir, f))
            print(f"Backed up {len(wav_files)} files to {backup_dir}/")

    mode = "DRY RUN" if dry_run else "STRIP"
    print(f"\n[{mode}] Processing {len(wav_files)} files in {data}/ "
          f"as {profile.plural}\n")

    counts: dict[str, int] = {}
    durations: list[float] = []
    loop_files: list[str] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Stripping {profile.plural}", total=len(wav_files))
        for filename in wav_files:
            result = strip_file(
                os.path.join(data, filename), profile,
                dry_run=dry_run, exclude_loops=exclude_loops,
            )
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if result["status"] == "loop":
                loop_files.append(filename)
            elif result["status"] in ("stripped", "skip_clean"):
                durations.append(result["duration_ms"])
            progress.update(task, advance=1)

    if loop_files and not dry_run:
        loops_dir = data.rstrip("/") + "_loops"
        os.makedirs(loops_dir, exist_ok=True)
        for f in loop_files:
            src, dst = os.path.join(data, f), os.path.join(loops_dir, f)
            shutil.move(src, dst) if move_loops else shutil.copy2(src, dst)
        print(f"\n{'Moved' if move_loops else 'Copied'} {len(loop_files)} "
              f"loops to {loops_dir}/")

    print("\nResults:")
    for status in ("stripped", "skip_clean", "loop", "skip_short",
                   "skip_no_hit", "unreadable"):
        if counts.get(status):
            print(f"  {status:14s} {counts[status]:5d}")

    if durations:
        arr = np.array(durations)
        print(f"\n{profile.display_name} durations (ms):")
        print(f"  Min: {arr.min():.0f}   Median: {np.median(arr):.0f}   "
              f"Max: {arr.max():.0f}   Mean: {arr.mean():.0f}")

    return {"counts": counts, "durations": durations}
