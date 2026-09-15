"""Log-mel spectrogram computation and [0, 1] normalization.

Uses BigVGAN's own ``mel_spectrogram()`` so the representation the VAE learns is
exactly the one the vocoder was trained to invert (n_fft=1024, magnitude
spectrum, ``ln(clamp(mel, min=1e-5))``).
"""

from __future__ import annotations

import torch
from bigvgan import mel_spectrogram as _bigvgan_mel

from .constants import (
    FMAX,
    FMIN,
    HOP_LENGTH,
    LOG_MEL_MAX,
    LOG_MEL_MIN,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    WIN_SIZE,
)


def log_mel(audio: torch.Tensor, n_frames: int = N_FRAMES) -> torch.Tensor:
    """(1, T) waveform -> (1, N_MELS, n_frames) log-mel spectrogram."""
    spec = _bigvgan_mel(
        audio, N_FFT, N_MELS, SAMPLE_RATE, HOP_LENGTH, WIN_SIZE,
        FMIN, FMAX, center=False,
    )
    if spec.shape[-1] > n_frames:
        return spec[..., :n_frames]
    if spec.shape[-1] < n_frames:
        return torch.nn.functional.pad(
            spec, (0, n_frames - spec.shape[-1]), value=LOG_MEL_MIN,
        )
    return spec


def normalize(spec: torch.Tensor) -> torch.Tensor:
    """BigVGAN log-mel -> [0, 1] using the fixed bounds in ``constants``."""
    spec = torch.clamp(spec, min=LOG_MEL_MIN, max=LOG_MEL_MAX)
    return (spec - LOG_MEL_MIN) / (LOG_MEL_MAX - LOG_MEL_MIN)


def denormalize(spec: torch.Tensor) -> torch.Tensor:
    """[0, 1] normalized spectrogram -> BigVGAN log-mel scale."""
    return spec * (LOG_MEL_MAX - LOG_MEL_MIN) + LOG_MEL_MIN


def spectrogram(audio: torch.Tensor, n_frames: int = N_FRAMES) -> torch.Tensor:
    """(1, T) waveform -> normalized (1, N_MELS, n_frames) spectrogram in [0, 1]."""
    return normalize(log_mel(audio, n_frames))
