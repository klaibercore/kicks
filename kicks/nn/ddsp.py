"""Small original PyTorch percussion processors inspired by DDSP/NoiseBandNet.

No TensorFlow or upstream NoiseBandNet code is used. The noise branch learns
band-energy envelopes, never a phase-dependent recording-minus-DSP residual.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from ..audio.constants import AUDIO_LENGTH, SAMPLE_RATE
from ..instruments import get_profile


def finish_audio(x: torch.Tensor) -> torch.Tensor:
    """Bounded peak, DC removal and file-boundary fades, preserving silence."""
    x = x - x.mean(dim=-1, keepdim=True)
    fade = min(220, x.shape[-1] // 4)
    window = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    window[:min(8, fade)] = torch.linspace(0, 1, min(8, fade), device=x.device)
    window[-fade:] = torch.linspace(1, 0, fade, device=x.device)
    x = x * window
    return x * (.9 / x.abs().amax(dim=-1, keepdim=True).clamp_min(.9))


def physical_values(instrument, positions):
    spec = get_profile(instrument).procedural
    values = {}
    for i, p in enumerate(spec.parameters):
        v = positions[:, i:i+1].clamp(0, 1)
        values[p.key] = p.low * (p.high/p.low)**v if p.logarithmic else p.low+(p.high-p.low)*v
    return values


def modal_body(instrument, positions, length=AUDIO_LENGTH):
    """Differentiable integrated-pitch and inharmonic modes at 44.1 kHz.

    This is a separate renderer from the NumPy fourfold-oversampled voice.
    Frequencies are bounded below Nyquist; no square-wave source is used.
    """
    p = physical_values(instrument, positions)
    t = torch.arange(length, device=positions.device, dtype=positions.dtype)[None]/SAMPLE_RATE
    envelope = lambda ms: torch.exp(-math.log(1000)*t/(ms/1000))
    if instrument == 'kick':
        f = p['tune_hz'] + p['pitch_drop_hz']*torch.exp(-t/(p['pitch_ms']/1000))
        phase = 2*math.pi*(f.cumsum(-1)-f)/SAMPLE_RATE
        x = (phase.sin()+.2*p['tone']*(phase*2.17).sin())*envelope(p['decay_ms'])
    elif instrument == 'snare':
        x = torch.zeros_like(t).expand(len(positions), -1)
        for i, ratio in enumerate((1., 1.59, 2.14, 2.79, 3.52)):
            harmonic = i+1
            r = harmonic+(ratio-harmonic)*p['inharmonic']
            x = x + torch.sin(2*math.pi*p['tune_hz']*r*t)*envelope(p['body_ms']/(1+.25*i))/(i+1)
        x = x*(1-p['wires'])*.7
    else:
        x = torch.zeros_like(t).expand(len(positions), -1)
        for ratio in (1., 1.304, 1.466, 1.787, 1.932, 2.536):
            f = p['metal_hz']*ratio
            # A bounded sinusoidal mode bank; the learned branch carries air.
            x = x + torch.sin(2*math.pi*f*t + p['fm']*torch.sin(2*math.pi*f*.73*t))/6
        x = x*envelope(p['decay_ms'])*(1-p['noise'])
    return x[:, None]


class NoiseBands(nn.Module):
    """Seeded, band-limited stochastic excitation with learned temporal gains."""
    def __init__(self, bands=16, length=AUDIO_LENGTH, low_hz=300, high_hz=20000):
        super().__init__()
        if not 2 <= bands <= 64 or length < 512 or not 0 < low_hz < high_hz < SAMPLE_RATE/2:
            raise ValueError('invalid noise-bank geometry')
        self.bands, self.length = bands, length
        frequencies = torch.fft.rfftfreq(length, 1/SAMPLE_RATE)
        centers = torch.logspace(math.log10(low_hz), math.log10(high_hz), bands)
        log_f = frequencies.clamp_min(1).log()
        width = (centers[1].log()-centers[0].log())*1.5
        masks = (1-(log_f[None]-centers.log()[:, None]).abs()/width).clamp_min(0)
        masks[:, (frequencies < low_hz/2) | (frequencies > high_hz)] = 0
        self.register_buffer('masks', masks)

    def forward(self, gains, *, seed=0):
        if gains.ndim != 3 or gains.shape[1] != self.bands:
            raise ValueError('gains must be batch × bands × envelope frames')
        rng = torch.Generator(device='cpu').manual_seed(seed)
        white = torch.randn(gains.shape[0], self.bands, self.length, generator=rng).to(gains)
        noise = torch.fft.irfft(torch.fft.rfft(white)*self.masks[None], n=self.length)
        noise = noise/noise.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        env = F.interpolate(gains, size=self.length, mode='linear', align_corners=False)
        return (noise*env).sum(1, keepdim=True)/math.sqrt(self.bands)


def spectral_loss(audio, reference):
    """True multi-window magnitude and temporal-envelope loss; no phase target."""
    total = audio.new_tensor(0.)
    for size in (128, 512, 2048):
        window = torch.hann_window(size, device=audio.device)
        a = torch.stft(audio[:, 0], size, size//4, window=window, return_complex=True).abs()
        b = torch.stft(reference[:, 0], size, size//4, window=window, return_complex=True).abs()
        total = total + (a.clamp_min(1e-4).log()-b.clamp_min(1e-4).log()).abs().mean()/3
        total = total + (a-b).abs().mean()/(b.mean().clamp_min(1e-4)*3)
    env = lambda x: F.avg_pool1d(x.square(), 256, 128).clamp_min(1e-8).sqrt()
    return total + 2*F.l1_loss(env(audio), env(reference))
