"""Close the control loop on the actual vocoded waveform.

Smooth, bounded STFT gains preserve the vocoder's phase while correcting its
spectral/envelope drift. Every iteration measures the reconstructed waveform,
so cancellation of cross-talk includes overlap-add and the mel analysis.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import least_squares

from ..analysis.descriptors import descriptor_tensor
from ..instruments import InstrumentProfile
from .constants import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE
from .mel import spectrogram


@lru_cache(maxsize=16)
def _control_masks(profile: InstrumentProfile, n_frames: int) -> torch.Tensor:
    # Slaney mel centres, matching BigVGAN (linear below 1 kHz, log above).
    log_step = np.log(6.4) / 27
    top = 15 + np.log((SAMPLE_RATE / 2) / 1000) / log_step
    mels = np.linspace(0, top, N_MELS + 2)[1:-1]
    centres = np.where(mels < 15, mels * 200 / 3, 1000 * np.exp(log_step * (mels - 15)))
    frequencies = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    times = np.arange(n_frames) * HOP_LENGTH / SAMPLE_RATE

    def window(x, low, high, width):
        left = 1 if low == 0 else 0.5 * (1 + np.tanh((x - low) / width))
        right = 1 if high is None else 0.5 * (1 - np.tanh((x - high) / width))
        return left * right

    def region_mask(region):
        f0 = 0 if region.bands[0] == 0 else centres[region.bands[0]]
        f1 = None if region.bands[1] is None else centres[min(N_MELS - 1, region.bands[1])]
        f = window(frequencies, f0, f1, max(SAMPLE_RATE / N_FFT, (f1 or f0) * 0.05))
        t = window(times, region.frames[0] * HOP_LENGTH / SAMPLE_RATE,
                   None if region.frames[1] is None else region.frames[1] * HOP_LENGTH / SAMPLE_RATE,
                   HOP_LENGTH / SAMPLE_RATE)
        return f[:, None] * t

    masks, seen = [], set()
    for descriptor in profile.descriptors:
        if descriptor.kind == "centroid_ms":
            # Tilt the low-end envelope on the instrument's onset timescale.
            # Cap the tilt so silence far beyond the hit cannot be amplified.
            timescale = profile.onsets.min_distance_ms / 1000
            masks.append(region_mask(descriptor.region) * np.minimum(times / timescale, 5))
        else:
            for region in (descriptor.region, descriptor.reference):
                if region is not None and region not in seen:
                    masks.append(region_mask(region))
                    seen.add(region)
    return torch.tensor(np.stack(masks), dtype=torch.float32)


def correct_waveform(
    waveform: torch.Tensor,
    targets: np.ndarray,
    spans: np.ndarray,
    profile: InstrumentProfile,
    *,
    max_gain_db: float = 40.0,
    regularization: float = 0.0,
) -> torch.Tensor:
    """Return a mono waveform matching the calibrated, gain-invariant controls.

    The VAE supplies the sound; this bounded correction removes the vocoder's
    drift. Call before explicit user effects, which intentionally alter it.
    """
    if not profile.waveform_controls:
        return waveform
    if not np.isfinite(max_gain_db) or not 0 < max_gain_db <= 40 or not np.isfinite(regularization) or regularization < 0:
        raise ValueError("correction gain must be in (0, 40] dB and regularization nonnegative")
    if waveform.ndim != 1 or not torch.isfinite(waveform).all():
        raise ValueError("waveform correction expects finite mono audio")
    waveform = waveform.detach().cpu().float()
    if waveform.abs().max() < 1e-8:
        return waveform
    target = np.asarray(targets, dtype=np.float64)
    span = np.maximum(np.asarray(spans, dtype=np.float64), 1e-6)
    if target.shape != (len(profile.descriptors),) or span.shape != target.shape or not np.isfinite(target).all() or not np.isfinite(span).all():
        raise ValueError("targets and spans must be finite and match the descriptor count")
    max_log_gain = max_gain_db * np.log(10) / 20
    window = torch.hann_window(N_FFT)
    original = torch.stft(waveform, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    masks = _control_masks(profile, original.shape[-1])
    count = len(masks)

    def render(parameters):
        parameters = torch.as_tensor(np.atleast_2d(parameters), dtype=torch.float32)
        gain = torch.einsum("bd,dft->bft", parameters, masks)
        spectrum = original[None] * gain.clamp(-max_log_gain, max_log_gain).exp()
        audio = torch.istft(spectrum, N_FFT, HOP_LENGTH, window=window, length=len(waveform))
        return audio / audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)

    def measure(parameters):
        return descriptor_tensor(spectrogram(render(parameters)), profile).numpy().astype(np.float64)

    def residual(parameters):
        error = (measure(parameters)[0] - target) / span
        return np.r_[error, regularization * parameters] if regularization else error

    def jacobian(parameters):
        step = 0.005
        delta = np.eye(count) * step
        measured = measure(np.vstack([parameters + delta, parameters - delta]))
        jac = ((measured[:count] - measured[count:]) / (2 * step) / span).T
        return np.vstack([jac, regularization * np.eye(count)]) if regularization else jac

    result = least_squares(
        residual, np.zeros(count), jac=jacobian, bounds=(-3, 3), tr_solver="lsmr",
        max_nfev=48, ftol=1e-6, xtol=1e-6, gtol=1e-6,
    )
    return render(result.x)[0]
