"""Torch-side audio loading: the exact pre-processing the model was trained on.

Kept separate from :mod:`kicks.audio.waveform` (numpy only) so that analysis
tools do not drag torch and BigVGAN into their import graph.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio

from .constants import AUDIO_LENGTH, SAMPLE_RATE, TARGET_LUFS


def list_wavs(directory: str) -> list[str]:
    """Sorted .wav filenames in a directory (names only, not paths)."""
    if not os.path.isdir(directory):
        raise RuntimeError(f"Directory not found: {directory}")
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".wav"))


def make_meter(sample_rate: int = SAMPLE_RATE) -> pyln.Meter:
    """An ITU-R BS.1770 loudness meter. Reuse one across a corpus — it is not cheap."""
    return pyln.Meter(sample_rate)


def read_mono(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor | None:
    """Read a wav as a mono (1, T) float tensor at ``sample_rate``."""
    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    # sf.read returns (samples,) for mono or (samples, channels) for stereo
    audio = (
        torch.from_numpy(data).unsqueeze(0) if data.ndim == 1
        else torch.from_numpy(data.T)
    )
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    return audio


def fit_length(audio: torch.Tensor, length: int = AUDIO_LENGTH) -> torch.Tensor:
    """Truncate or zero-pad the last dimension to exactly ``length``."""
    n = audio.shape[-1]
    if n > length:
        return audio[..., :length]
    if n < length:
        return torch.nn.functional.pad(audio, (0, length - n))
    return audio


def lufs_normalize(
    audio: torch.Tensor,
    meter: pyln.Meter | None = None,
    target_lufs: float = TARGET_LUFS,
) -> torch.Tensor:
    """Normalize integrated loudness to ``target_lufs``.

    Silent or non-finite-loudness input is returned untouched — normalizing it
    would amplify a noise floor into the corpus.
    """
    if meter is None:
        meter = make_meter()
    audio_np = audio.squeeze(0).numpy()
    loudness = meter.integrated_loudness(audio_np)
    if not np.isfinite(loudness):
        return audio
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Possible clipped samples", module="pyloudnorm",
        )
        audio_np = pyln.normalize.loudness(audio_np, loudness, target_lufs)
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return torch.from_numpy(audio_np).unsqueeze(0).float()


def load_waveform(
    path: str,
    meter: pyln.Meter | None = None,
    length: int = AUDIO_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float | None = TARGET_LUFS,
) -> torch.Tensor | None:
    """Full training pre-processing: mono -> resample -> fit length -> LUFS.

    Returns a (1, ``length``) tensor, or None if the file cannot be read. This
    is the single definition of "how audio enters the model" — the dataset and
    the latent prior both go through it, so they cannot drift apart.
    """
    audio = read_mono(path, sample_rate)
    if audio is None:
        return None
    audio = fit_length(audio, length)
    if target_lufs is not None:
        audio = lufs_normalize(audio, meter, target_lufs)
    return audio
