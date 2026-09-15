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
    """Attack ramp followed by an exponential decay to -60 dB.

    ``decay_ms`` is measured from the end of the attack, so it shapes the
    audible hit even when the waveform has a long zero-padded tail.
    ``waveform`` is (T,); returns (T,).
    """
    if attack_ms is None and decay_ms is None:
        return waveform
    n = waveform.shape[-1]
    env = torch.ones_like(waveform)
    a = 0
    if attack_ms is not None:
        a = min(n, int(sample_rate * max(0.0, attack_ms) / 1000))
        if a > 0:
            env[:a] = torch.linspace(0, 1, a, device=waveform.device)
    if decay_ms is not None:
        d = max(1.0, sample_rate * max(0.0, decay_ms) / 1000)
        elapsed = (torch.arange(n, device=waveform.device) - a).clamp(min=0)
        env *= torch.exp(-6.907755278982137 * elapsed / d)
    return waveform * env


def apply_drive(waveform: torch.Tensor, drive: float) -> torch.Tensor:
    """Tanh saturation with gain compensation. ``drive`` in [0, 1]."""
    drive = max(0.0, min(1.0, drive))
    if drive == 0:
        return waveform
    gain = 1.0 + drive * 8.0
    saturated = torch.tanh(waveform * gain) / torch.tanh(waveform.new_tensor(gain))
    return waveform.lerp(saturated, drive)


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
