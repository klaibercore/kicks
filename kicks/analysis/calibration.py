"""Persist descriptor calibration, tied to weights, preprocessing and corpus.

NPZ contains only arrays and JSON strings; loading never enables pickle.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .basis import DescriptorBasis, SliderBasis, analyze_latent_space
from .latents import extract_latents

CALIBRATION_VERSION = 3
_ARRAYS = ("mean_z", "d_means", "transform", "transform_inverse", "lower", "upper",
           "d_scale", "axes", "W", "anchor", "r2")


def calibration_fingerprint(checkpoint: str, dataset, profile) -> str:
    h = hashlib.sha256()
    with open(checkpoint, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    h.update(repr((CALIBRATION_VERSION, "peak_safe_lufs_v1", profile.descriptors)).encode())
    for path in dataset.paths:
        stat = os.stat(path)
        h.update(repr((path, stat.st_size, stat.st_mtime_ns)).encode())
    return h.hexdigest()


def fit_or_load_basis(model, dataset, profile, device, *, checkpoint=None, mode="descriptor", refresh=False):
    """Fit once per model/corpus/profile version; reuse the verified range on restart."""
    from torch.utils.data import DataLoader

    checkpoint = checkpoint or profile.paths.checkpoint
    path = Path(checkpoint).with_suffix(".controls.npz")
    fingerprint = calibration_fingerprint(checkpoint, dataset, profile)
    if mode == "descriptor" and path.exists() and not refresh:
        try:
            with np.load(path, allow_pickle=False) as cache:
                if str(cache["fingerprint"]) == fingerprint:
                    basis = DescriptorBasis.__new__(DescriptorBasis)
                    for key in _ARRAYS:
                        setattr(basis, key, cache[key])
                    basis.radius = float(cache["radius"])
                    return SliderBasis(
                        basis=basis, projected=cache["projected"], names=cache["names"].tolist(),
                        mins=cache["mins"].tolist(), maxs=cache["maxs"].tolist(),
                        calibration=json.loads(str(cache["calibration"])),
                    )
        except (ValueError, KeyError, OSError):
            pass  # incomplete or obsolete cache: rebuild from the source
    latents, specs = extract_latents(model, DataLoader(dataset, batch_size=64), device)
    fitted = analyze_latent_space(latents, specs, profile, basis=mode, model=model)
    if mode == "descriptor":
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, fingerprint=fingerprint, projected=fitted.projected,
                            names=fitted.names, mins=fitted.mins, maxs=fitted.maxs,
                            calibration=json.dumps(fitted.calibration), radius=fitted.basis.radius,
                            **{key: getattr(fitted.basis, key) for key in _ARRAYS})
        os.replace(temp, path)
    return fitted
