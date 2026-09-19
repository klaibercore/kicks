"""Corpus-anchored controls: change character without discarding texture.

This addresses the studio's inverse problem, not the waveform reconstruction
ceiling. Five descriptor targets cannot determine a complete drum sound. The
remaining latent degrees of freedom are supplied by a real corpus encoding.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..analysis.basis import DescriptorBasis, SliderBasis, _decoded_descriptors
from ..analysis.calibration import _ARRAYS, calibration_fingerprint
from ..analysis.latents import extract_latents

IDENTITY_VERSION = 1
IDENTITY_RADIUS = 4.0
CORRECTION_LIMIT_DB = 6.0


class IdentityBasis(DescriptorBasis):
    """Retain an encoded anchor's locally descriptor-neutral directions."""

    def __init__(self, latents: np.ndarray, descriptors: np.ndarray):
        super().__init__(latents, descriptors)
        self.support_z = np.asarray(latents, dtype=np.float64)
        self.support_d = np.asarray(descriptors, dtype=np.float64)
        self.quantiles = np.percentile(descriptors, [5, 25, 50, 75, 95], axis=0)

    def anchor_index(self, seed: int = 0) -> int:
        """Seed zero is the nearest central sound; other seeds vary that texture.

        Selection is independent of slider position, so moving a slider does
        not select a different recording halfway through its travel.
        """
        if not isinstance(seed, (int, np.integer)) or seed < 0 or seed > 2**32 - 1:
            raise ValueError("texture seed must be an integer in [0, 4294967295]")
        scale = np.maximum(self.quantiles[3] - self.quantiles[1], 1e-6)
        distance = np.linalg.norm((self.support_d - self.quantiles[2]) / scale, axis=1)
        order = np.argsort(distance, kind="stable")
        if seed == 0:
            return int(order[0])
        pool = order[:min(64, len(order))]
        return int(np.random.default_rng(seed).choice(pool))

    def local_directions(self, measure_fn, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        anchor = self.support_z[self.anchor_index(seed)][None]
        if hasattr(measure_fn, "jacobian"):
            jac = measure_fn.jacobian(anchor) @ self.transform / self.d_scale[:, None]
        else:
            eps = .01
            jac = np.column_stack([
                (np.asarray(measure_fn(anchor + eps * axis)).reshape(-1)
                 - np.asarray(measure_fn(anchor - eps * axis)).reshape(-1)) / (2 * eps * self.d_scale)
                for axis in self.transform.T
            ])
        _, singular, vt = np.linalg.svd(jac, full_matrices=False)
        rank = int((singular > max(1e-8, singular[0] * 1e-5)).sum())
        return anchor, vt[:rank].T

    def solve(self, targets, measure_fn=None, n_iter: int = 24, *, seed: int = 0) -> np.ndarray:
        target = np.asarray(targets, dtype=np.float64)
        if target.shape != self.d_means.shape or not np.isfinite(target).all():
            raise ValueError("descriptor targets must be finite and match the descriptor count")
        if measure_fn is None:
            return self.support_z[self.anchor_index(seed)][None].copy()
        anchor, directions = self.local_directions(measure_fn, seed)
        if not directions.shape[1]:
            return anchor.copy()
        latent_directions = self.transform @ directions
        dimensions = directions.shape[1]
        # Box inscribed in a trust-region ball: even all controls at their limit
        # cannot leave the local standardized latent trust region.
        bound = IDENTITY_RADIUS / np.sqrt(dimensions)
        regularization = .025

        def latent(delta):
            return anchor + (latent_directions @ delta)[None]

        def residual(delta):
            measured = np.asarray(measure_fn(latent(delta))).reshape(-1)
            return np.r_[(measured - target) / self.d_scale, regularization * delta]

        def jacobian(delta):
            if hasattr(measure_fn, "jacobian"):
                jac = measure_fn.jacobian(latent(delta)) @ latent_directions / self.d_scale[:, None]
            else:
                eps = .01
                jac = np.column_stack([
                    (np.asarray(measure_fn(latent(delta + eps * e))).reshape(-1)
                     - np.asarray(measure_fn(latent(delta - eps * e))).reshape(-1)) / (2 * eps * self.d_scale)
                    for e in np.eye(dimensions)
                ])
            return np.vstack([jac, np.eye(dimensions) * regularization])

        result = least_squares(residual, np.zeros(dimensions), jac=jacobian,
                               bounds=(-bound, bound), max_nfev=n_iter,
                               ftol=1e-5, xtol=1e-5, gtol=1e-5)
        return latent(result.x)


def fit_identity_basis(model, dataset, profile, device, *, checkpoint=None, refresh=False) -> SliderBasis:
    """Cache support encodings separately from legacy exact-control calibration."""
    from torch.utils.data import DataLoader

    checkpoint = checkpoint or profile.paths.checkpoint
    path = Path(checkpoint).with_suffix(".identity.npz")
    fingerprint = hashlib.sha256(
        f"{IDENTITY_VERSION}:{calibration_fingerprint(checkpoint, dataset, profile)}".encode()
    ).hexdigest()
    basis = None
    if path.exists() and not refresh:
        try:
            with np.load(path, allow_pickle=False) as cache:
                if str(cache["fingerprint"]) == fingerprint:
                    basis = IdentityBasis.__new__(IdentityBasis)
                    for key in (*_ARRAYS, "support_z", "support_d", "quantiles"):
                        setattr(basis, key, cache[key])
                    basis.radius = float(cache["radius"])
        except (ValueError, KeyError, OSError):
            basis = None
    if basis is None:
        latents, _ = extract_latents(model, DataLoader(dataset, batch_size=64), device)
        z, descriptors = _decoded_descriptors(model, latents, profile)
        basis = IdentityBasis(z, descriptors)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, fingerprint=fingerprint, radius=basis.radius,
                            **{key: getattr(basis, key) for key in (*_ARRAYS, "support_z", "support_d", "quantiles")})
        os.replace(temp, path)
    return SliderBasis(
        basis=basis, projected=basis.support_d, names=profile.descriptor_labels,
        mins=basis.quantiles[0].tolist(), maxs=basis.quantiles[-1].tolist(),
        control_points=basis.quantiles,
        calibration={"version": IDENTITY_VERSION, "scope": "corpus_anchored",
                     "mapping": "5_25_50_75_95_percentiles", "texture_anchors": len(basis.support_z),
                     "max_latent_step": IDENTITY_RADIUS, "max_correction_db": CORRECTION_LIMIT_DB,
                     "exact_targets_guaranteed": False},
    )
