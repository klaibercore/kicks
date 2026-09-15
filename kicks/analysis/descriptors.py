"""Perceptual descriptors read off a spectrogram.

Descriptors are the human-facing axes of the latent space — what a slider is
labelled and what it is supposed to do. Which ones exist, and which region of
the spectrogram each one measures, comes entirely from the instrument profile:
a kick reports sub / punch / click / bright / decay, a hi-hat reports
attack / body / sizzle / bright / decay, and this module does not know or care
which it is handed.
"""

from __future__ import annotations

import numpy as np

from ..instruments import InstrumentProfile


def _as_2d(spec) -> np.ndarray:
    """Accept a torch tensor or ndarray of shape (1, F, T), (F, T) or (1, 1, F, T)."""
    if hasattr(spec, "detach"):
        spec = spec.detach().cpu().numpy()
    spec = np.asarray(spec)
    while spec.ndim > 2:
        spec = spec[0]
    if spec.ndim != 2:
        raise ValueError(f"expected a 2-D spectrogram, got shape {spec.shape}")
    return spec


def compute_descriptors(spec, profile: InstrumentProfile) -> dict[str, float]:
    """Evaluate every descriptor in ``profile`` on one normalized spectrogram."""
    arr = _as_2d(spec)
    return {d.key: d.compute(arr) for d in profile.descriptors}


def descriptor_vector(spec, profile: InstrumentProfile) -> np.ndarray:
    """Descriptors as a vector, in profile order."""
    arr = _as_2d(spec)
    return np.array([d.compute(arr) for d in profile.descriptors], dtype=np.float64)


def descriptor_matrix(specs, profile: InstrumentProfile) -> np.ndarray:
    """(n_samples, n_descriptors) matrix over an iterable of spectrograms."""
    return np.stack([descriptor_vector(s, profile) for s in specs])


def descriptor_tensor(specs, profile: InstrumentProfile):
    """Differentiable, batched equivalent of the profile's numpy descriptors.

    Torch is imported here so corpus-only analysis remains lightweight.
    """
    import torch
    from ..audio.constants import LOG_MEL_MAX, LOG_MEL_MIN, MS_PER_FRAME

    arr = specs[:, 0] if specs.ndim == 4 else specs
    energy = None
    values = []
    def region_mean(data, region):
        return data[:, region.bands[0]:region.bands[1],
                    region.frames[0]:region.frames[1]].mean(dim=(-2, -1))
    for d in profile.descriptors:
        if d.kind in ("power_db_ratio", "centroid_ms"):
            if energy is None:
                magnitude = (torch.exp(arr * (LOG_MEL_MAX - LOG_MEL_MIN) + LOG_MEL_MIN)
                             - np.exp(LOG_MEL_MIN)).clamp_min(0)
                energy = magnitude.square()
            if d.kind == "centroid_ms":
                a = energy[:, d.region.bands[0]:d.region.bands[1],
                           d.region.frames[0]:d.region.frames[1]].sum(dim=1)
                times = (torch.arange(a.shape[-1], device=arr.device, dtype=arr.dtype)
                         + d.region.frames[0]) * MS_PER_FRAME
                value = (a * times).sum(dim=-1) / (a.sum(dim=-1) + 1e-20)
            else:
                value = 10 * torch.log10((region_mean(energy, d.region) + 1e-20)
                                        / (region_mean(energy, d.reference) + 1e-20))
        else:
            a = region_mean(arr, d.region)
            if d.kind == "mean":
                value = a
            else:
                b = region_mean(arr, d.reference)
                if d.kind == "log_ratio":
                    value = (torch.log((a / (b + 1e-8)).clamp_min(1e-8)) / d.scale).clamp(0, 1)
                elif d.kind == "fraction":
                    value = a / (a + b + 1e-8)
                elif d.kind == "inverse_ratio":
                    value = 1 - (a / (b + 1e-8)).clamp(0, 1)
                else:
                    raise ValueError(f"unknown descriptor kind {d.kind!r}")
        values.append(value)
    return torch.stack(values, dim=-1)


class DecoderResponse:
    """Measurements and exact local derivatives of the decoder's controls."""

    def __init__(self, model, profile: InstrumentProfile):
        self.model = model
        self.profile = profile

    def __call__(self, z: np.ndarray) -> np.ndarray:
        import torch
        device = next(self.model.parameters()).device
        with torch.no_grad():
            spec = self.model.decode(torch.as_tensor(z, dtype=torch.float32, device=device))
            return descriptor_tensor(spec, self.profile).cpu().numpy().astype(np.float64)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        import torch
        device = next(self.model.parameters()).device
        with torch.enable_grad():
            latent = torch.tensor(z, dtype=torch.float32, device=device, requires_grad=True)
            d = descriptor_tensor(self.model.decode(latent), self.profile)
            rows = [torch.autograd.grad(d[0, j], latent, retain_graph=j < d.shape[1] - 1)[0][0]
                    for j in range(d.shape[1])]
        return torch.stack(rows).detach().cpu().numpy().astype(np.float64)

    def quality(self, z: np.ndarray, jacobian: bool = False):
        """Excess low-end envelope rises after the transient (zero is one decay).

        This constraint occupies unused latent dimensions instead of forcing
        the user-facing descriptors to absorb a second, delayed body peak.
        """
        import torch
        import torch.nn.functional as F
        from ..audio.constants import HOP_LENGTH, LOG_MEL_MAX, LOG_MEL_MIN

        if not (self.profile.waveform_controls and self.profile.envelope_gate):
            return None
        device = next(self.model.parameters()).device
        with torch.enable_grad():
            latent = torch.tensor(z, dtype=torch.float32, device=device, requires_grad=jacobian)
            spec = self.model.decode(latent)[:, 0]
            region = next(d.region for d in self.profile.descriptors if d.kind == "centroid_ms")
            magnitude = torch.exp(spec[:, region.bands[0]:region.bands[1]]
                                  * (LOG_MEL_MAX - LOG_MEL_MIN) + LOG_MEL_MIN)
            energy = magnitude.square().sum(dim=1)
            width = max(3, self.profile.onsets.envelope_win // HOP_LENGTH | 1)
            smooth = F.avg_pool1d(energy[:, None], width, stride=1, padding=width // 2)[:, 0]
            start = self.profile.transient_loss.tail_start
            rises = (smooth[:, start:] - smooth[:, start-1:-1]).clamp_min(0).sum(dim=-1)
            value = rises / smooth.amax(dim=-1).clamp_min(1e-12)
            if jacobian:
                return torch.autograd.grad(value.sum(), latent)[0][0].detach().cpu().numpy()[None]
        return value.detach().cpu().numpy().astype(np.float64)


def descriptor_stats(matrix: np.ndarray, profile: InstrumentProfile) -> dict[str, dict[str, float]]:
    """Per-descriptor mean/std/min/max, keyed by descriptor."""
    return {
        key: {
            "mean": float(matrix[:, i].mean()),
            "std": float(matrix[:, i].std()),
            "min": float(matrix[:, i].min()),
            "max": float(matrix[:, i].max()),
        }
        for i, key in enumerate(profile.descriptor_keys)
    }


def correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between every column of ``a`` and every column of ``b``.

    Columns with zero variance correlate to 0 rather than NaN, so a collapsed
    descriptor degrades gracefully instead of poisoning downstream sorting.
    """
    def z(x: np.ndarray) -> np.ndarray:
        std = x.std(axis=0)
        safe = np.where(std > 0, std, 1.0)
        return np.where(std > 0, (x - x.mean(axis=0)) / safe, 0.0)

    return z(a).T @ z(b) / len(a)
