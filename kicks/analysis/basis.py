"""Slider bases: how a handful of knobs address a 32-dimensional latent space.

Two strategies, both exposing ``inverse_transform`` so consumers do not care
which is in use:

* ``"pca"`` (default) — unsupervised principal components of the corpus latents,
  auto-named by whichever perceptual descriptor they correlate with most. Honest
  about the data's own structure, but each component drags several descriptors
  at once, so one axis gets explicit cross-talk compensation.
* ``"descriptor"`` — supervised axes where slider *j* targets descriptor *j*
  directly, with first-order cross-talk cancelled by construction and a
  closed-loop Newton correction on top.

Both are built from the instrument profile's descriptor list, so the slider
count and labels follow the instrument rather than a hardcoded five.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from ..instruments import InstrumentProfile
from .descriptors import correlation_matrix, descriptor_matrix

#: Minimum |r| before a principal component is named after a descriptor.
NAMING_THRESHOLD = 0.15


@dataclass
class SliderBasis:
    """Everything the server and generators need to drive sliders."""

    basis: "PCA | DescriptorBasis"   # anything with inverse_transform()
    projected: np.ndarray            # (n_samples, n_sliders) corpus positions
    names: list[str]                 # slider labels, e.g. ["Sub", "Punch", ...]
    mins: list[float]                # 2nd percentile per axis
    maxs: list[float]                # 98th percentile per axis
    decorrelate_idx: int | None = None        # axis that absorbs cross-talk
    decorrelate_ratios: np.ndarray | None = None   # per-axis compensation ratios

    @property
    def n_sliders(self) -> int:
        return len(self.names)

    @property
    def is_descriptor_basis(self) -> bool:
        return isinstance(self.basis, DescriptorBasis)


class DescriptorBasis:
    """Slider axes that target perceptual descriptors directly.

    Fits a linear map ``descriptors ≈ (z - mean_z) @ W`` over the corpus, then
    uses the minimum-norm pseudo-inverse directions ``S = W (WᵀW)⁻¹`` as slider
    axes: moving slider *j* changes descriptor *j* by the requested amount while
    (to first order) leaving the others untouched — unlike raw PCA components,
    which each move several descriptors at once.

    Duck-types PCA's ``inverse_transform`` so consumers work unchanged; "axis
    values" are simply target descriptor values.
    """

    def __init__(self, latents: np.ndarray, desc_matrix: np.ndarray):
        self.mean_z = latents.mean(axis=0)
        self.d_means = desc_matrix.mean(axis=0)
        Zc = latents - self.mean_z
        Dc = desc_matrix - self.d_means
        W, *_ = np.linalg.lstsq(Zc, Dc, rcond=None)     # (latent_dim, n_desc)
        self.W = W
        self.axes = W @ np.linalg.inv(W.T @ W + 1e-6 * np.eye(W.shape[1]))
        # Fit quality per descriptor (R²) — how linearly controllable each is.
        pred = Zc @ W
        ss_res = ((Dc - pred) ** 2).sum(axis=0)
        ss_tot = (Dc ** 2).sum(axis=0) + 1e-12
        self.r2 = 1.0 - ss_res / ss_tot

    def inverse_transform(self, X) -> np.ndarray:
        """Descriptor targets (B, n_desc) -> latent vectors (B, latent_dim)."""
        X = np.asarray(X, dtype=np.float64)
        return self.mean_z + (X - self.d_means) @ self.axes.T

    def solve(self, targets, measure_fn=None, n_iter: int = 2) -> np.ndarray:
        """Latent (1, latent_dim) hitting the descriptor targets.

        With a ``measure_fn`` (latents (1, dim) -> measured descriptors of the
        *decoded* spectrogram), runs closed-loop Newton correction: decode,
        measure, step along the axes toward the residual. Measured on the kick
        model this roughly doubles slider authority versus the open-loop map
        (~30% -> ~64% of the corpus descriptor span), because the decoder's
        response is nonlinear away from the corpus mean.
        """
        t = np.asarray(targets, dtype=np.float64)
        z = self.inverse_transform(t[None])
        if measure_fn is None:
            return z
        for _ in range(n_iter):
            d = np.asarray(measure_fn(z), dtype=np.float64)
            z = z + ((t - d) @ self.axes.T)[None, :]
        return z


def _decoded_descriptors(
    model, latents: np.ndarray, profile: InstrumentProfile, n_fit: int = 1500,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample corpus latents, decode them, and measure the result.

    Fitting the basis on decoded output rather than corpus spectrograms gives
    the decoder's *actual* response — the correct Jacobian for the closed-loop
    solve, which otherwise over- or under-shoots.
    """
    import torch

    rng = np.random.default_rng(0)
    idx = rng.choice(len(latents), min(n_fit, len(latents)), replace=False)
    Z = latents[idx]
    device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for i in range(0, len(Z), 64):
            batch = torch.tensor(Z[i: i + 64], dtype=torch.float32).to(device)
            decoded = model.decode(batch).cpu()
            rows.append(descriptor_matrix(decoded, profile))
    return Z, np.concatenate(rows)


def _percentile_range(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-column 2nd/98th percentiles — the usable slider travel."""
    return (
        [float(np.percentile(matrix[:, i], 2)) for i in range(matrix.shape[1])],
        [float(np.percentile(matrix[:, i], 98)) for i in range(matrix.shape[1])],
    )


def _fit_descriptor_basis(
    latents: np.ndarray,
    spectrograms,
    profile: InstrumentProfile,
    model=None,
    verbose: bool = True,
) -> SliderBasis:
    if model is not None:
        Z, D = _decoded_descriptors(model, latents, profile)
    else:
        Z, D = latents, descriptor_matrix(spectrograms, profile)

    basis = DescriptorBasis(Z, D)
    if verbose:
        r2s = ", ".join(f"{k}={r:.2f}" for k, r in zip(profile.descriptor_keys, basis.r2))
        print(f"Descriptor basis fit R²: {r2s}")

    mins, maxs = _percentile_range(D)
    return SliderBasis(
        basis=basis,
        projected=D,
        names=profile.descriptor_labels,
        mins=mins,
        maxs=maxs,
    )


def _name_components(
    projected: np.ndarray,
    desc_matrix: np.ndarray,
    pca: PCA,
    profile: InstrumentProfile,
    verbose: bool,
) -> list[str]:
    """Name each PC after its strongest unclaimed descriptor, flipping sign as needed.

    A component that correlates negatively with, say, brightness is negated in
    place so that every slider reads left-to-right as "less" to "more".
    """
    corr = correlation_matrix(projected, desc_matrix)   # (n_pcs, n_desc)
    keys = profile.descriptor_keys
    names: list[str] = []
    used: set[int] = set()

    for i in range(projected.shape[1]):
        candidates = [(abs(corr[i, j]), j) for j in range(len(keys)) if j not in used]
        best_abs, best_j = max(candidates) if candidates else (0.0, -1)
        if best_j >= 0 and best_abs >= NAMING_THRESHOLD:
            used.add(best_j)
            r = corr[i, best_j]
            names.append(profile.label_for(keys[best_j]))
            if r < 0:
                pca.components_[i] *= -1
                projected[:, i] *= -1
            if verbose:
                flipped = ", flipped" if r < 0 else ""
                print(f"  PC{i + 1} -> {names[-1]} (r={r:.2f}{flipped})")
        else:
            names.append(f"PC{i + 1}")
            if verbose:
                print(f"  PC{i + 1} -> PC{i + 1} (no strong correlation)")
    return names


def _decorrelation_ratios(
    projected: np.ndarray, target: np.ndarray, target_idx: int,
) -> np.ndarray | None:
    """Per-axis regression slopes normalized by the target axis's own slope.

    Subtracting ``ratio[i] * (value[i] - centre[i])`` from the target axis
    cancels, to first order, the descriptor drift the other sliders cause.
    Returns None when the target axis barely moves the descriptor at all, in
    which case the compensation would divide by ~0 and blow up.
    """
    d_std = target.std()
    if d_std <= 0:
        return None
    betas = np.zeros(projected.shape[1])
    for i in range(projected.shape[1]):
        pc = projected[:, i]
        if pc.std() > 0:
            betas[i] = float(np.corrcoef(pc, target)[0, 1]) * d_std / pc.std()
    if abs(betas[target_idx]) <= 1e-8:
        return None
    ratios = betas / betas[target_idx]
    ratios[target_idx] = 0.0
    return ratios


def analyze_latent_space(
    latents: np.ndarray,
    spectrograms,
    profile: InstrumentProfile,
    basis: str = "pca",
    model=None,
    n_sliders: int | None = None,
    verbose: bool = True,
) -> SliderBasis:
    """Fit the slider basis for one instrument.

    Args:
        latents: (n_samples, latent_dim) corpus mu vectors.
        spectrograms: matching corpus spectrograms, (n_samples, 1, F, T).
        profile: instrument profile — supplies the descriptors and slider count.
        basis: ``"pca"`` or ``"descriptor"``.
        model: the VAE. Only used by the descriptor basis, which fits on decoded
            output when it is available.
        n_sliders: override the axis count (defaults to one per descriptor).
        verbose: print the fit summary.
    """
    if basis == "descriptor":
        return _fit_descriptor_basis(latents, spectrograms, profile, model, verbose)

    n = n_sliders or profile.n_sliders
    desc_matrix = descriptor_matrix(spectrograms, profile)

    pca = PCA(n_components=n)
    projected = pca.fit_transform(latents)
    names = _name_components(projected, desc_matrix, pca, profile, verbose)

    mins, maxs = _percentile_range(projected)

    decorrelate_idx = None
    ratios = None
    target_key = profile.decorrelated_descriptor
    if target_key in profile.descriptor_keys:
        target_label = profile.label_for(target_key)
        if target_label in names:
            decorrelate_idx = names.index(target_label)
            col = profile.descriptor_keys.index(target_key)
            ratios = _decorrelation_ratios(projected, desc_matrix[:, col], decorrelate_idx)
            if verbose:
                if ratios is None:
                    print(f"{target_label} decorrelation skipped "
                          f"(that PC barely moves {target_key})")
                else:
                    print(f"{target_label} decorrelation enabled (ratios: {ratios.round(3)})")
            if ratios is None:
                decorrelate_idx = None

    if verbose:
        print(f"PCA variance explained: {pca.explained_variance_ratio_}")

    return SliderBasis(
        basis=pca,
        projected=projected,
        names=names,
        mins=mins,
        maxs=maxs,
        decorrelate_idx=decorrelate_idx,
        decorrelate_ratios=ratios,
    )


def slider_positions_to_axis_values(
    positions: list[float], basis: SliderBasis,
) -> list[float]:
    """Map slider positions in [0, 1] to basis-space values, with decorrelation.

    Each position is scaled into its axis's 2nd-98th percentile range; the
    decorrelated axis is then nudged to cancel the drift the other sliders
    introduced by moving away from their midpoints.
    """
    if len(positions) != basis.n_sliders:
        raise ValueError(
            f"expected {basis.n_sliders} slider positions, got {len(positions)}"
        )

    values = []
    for i, raw in enumerate(positions):
        lo, hi = basis.mins[i], basis.maxs[i]
        values.append(max(lo, min(hi, lo + raw * (hi - lo))))

    if basis.decorrelate_ratios is not None and basis.decorrelate_idx is not None:
        di = basis.decorrelate_idx
        for i in range(len(values)):
            if i == di:
                continue
            centre = (basis.mins[i] + basis.maxs[i]) / 2.0
            values[di] -= basis.decorrelate_ratios[i] * (values[i] - centre)
    return values
