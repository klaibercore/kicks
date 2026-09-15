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
