"""Post-vocoder shaping applied by the REST API's optional query parameters.

Instrument-agnostic: these operate on a finished waveform and mean the same
thing for a kick, a snare or a hat.
"""

from __future__ import annotations

import torch
import torchaudio

from .constants import SAMPLE_RATE


def apply_envelope(
    waveform: torch.Tensor,
    attack_ms: float | None = None,
    decay_ms: float | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Linear attack ramp at the head and/or decay ramp at the tail.

    ``waveform`` is (T,); returns (T,).
    """
    if attack_ms is None and decay_ms is None:
        return waveform
    n = waveform.shape[-1]
    env = torch.ones(n, dtype=waveform.dtype)
    if attack_ms is not None:
        a = min(n, int(sample_rate * max(0.0, attack_ms) / 1000))
        if a > 0:
            env[:a] = torch.linspace(0, 1, a)
    if decay_ms is not None:
        d = min(n, int(sample_rate * max(0.0, decay_ms) / 1000))
        if d > 0:
            env[-d:] = torch.linspace(1, 0, d)
    return waveform * env


def apply_drive(waveform: torch.Tensor, drive: float) -> torch.Tensor:
    """Tanh saturation with gain compensation. ``drive`` in [0, 1]."""
    drive = max(0.0, min(1.0, drive))
    return torch.tanh(waveform * (1.0 + drive * 8.0)) * (1.0 / (1.0 + drive * 0.3))


def apply_lowpass(
    waveform: torch.Tensor,
    cutoff_hz: float,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Second-order lowpass. ``waveform`` is (T,); returns (T,)."""
    cutoff = max(40.0, min(sample_rate / 2.0 - 100.0, cutoff_hz))
    return torchaudio.functional.lowpass_biquad(
        waveform.unsqueeze(0), sample_rate, cutoff_freq=cutoff, Q=0.707,
    ).squeeze(0)
