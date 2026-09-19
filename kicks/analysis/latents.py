"""Latent-space extraction and Gaussian mixture fitting."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def extract_latents(model, dataloader, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    """Encode a whole corpus. Returns (mu vectors, the spectrograms that produced them)."""
    model.eval()
    latents, spectrograms = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latents.append(mu.cpu().numpy())
            spectrograms.append(batch.cpu())
    return np.concatenate(latents), torch.cat(spectrograms)


def select_n_clusters(latents: np.ndarray, max_k: int = 10) -> tuple[int, list[float]]:
    """Choose a GMM component count by BIC (lower is better)."""
    if len(latents) < 2:
        raise ValueError("Clustering requires at least two samples")
    max_k = min(max_k, max(1, len(latents) // 3))
    k_range = range(1, max_k + 1)
    latents = latents.astype(np.float64)
    bics = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=42, n_init=3, reg_covar=1e-4,
        )
        gmm.fit(latents)
        bics.append(gmm.bic(latents))
    return list(k_range)[int(np.argmin(bics))], list(bics)


def fit_gmm(
    latents: np.ndarray, n_clusters: int,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Fit a GMM and return (model, hard labels, soft probabilities)."""
    latents = latents.astype(np.float64)
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type="full",
        random_state=42, n_init=3, reg_covar=1e-4,
    )
    gmm.fit(latents)
    return gmm, gmm.predict(latents), gmm.predict_proba(latents)


def analyze_clusters(latents: np.ndarray, max_k: int = 16) -> dict:
    """Select a reproducible, regularized mixture and retain its evidence.

    Remove constant dimensions, standardize, then retain orthogonal
    components covering at least 95% of variance. This limits covariance
    complexity without whitening low-energy noise. Compare full and diagonal
    covariance models, including the one-component null model, on the same data.
    Only converged fits are eligible; use the winning fit without re-fitting it.
    """
    x = np.asarray(latents, dtype=np.float64)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Clustering requires at least three finite latent vectors")
    std = x.std(axis=0)
    active = std > 1e-8
    if not active.any():
        raise ValueError("Latent vectors have no variation to cluster")
    normalized = (x[:, active] - x[:, active].mean(axis=0)) / std[active]
    pca = PCA(n_components=min(normalized.shape[1], len(x) - 1), svd_solver="full")
    projected = pca.fit_transform(normalized)
    dimensions = min(projected.shape[1], int(np.searchsorted(pca.explained_variance_ratio_.cumsum(), .95)) + 1)
    features = projected[:, :dimensions]
    limit = min(max_k, max(1, len(x) // max(10, dimensions * 2)))
    if limit < 1:
        raise ValueError("max_k must be positive")
    candidates = []
    best = None
    best_bic = float("inf")
    for k in range(1, limit + 1):
        for covariance in ("diag", "full"):
            model = GaussianMixture(
                n_components=k, covariance_type=covariance, random_state=42,
                n_init=3, max_iter=300, reg_covar=1e-3,
            ).fit(features)
            bic = float(model.bic(features))
            candidates.append({"k": k, "covariance": covariance, "bic": bic, "converged": bool(model.converged_)})
            if model.converged_ and np.isfinite(bic) and bic < best_bic:
                best, best_bic = model, bic
    if best is None:
        raise RuntimeError("No mixture candidate converged; inspect the input corpus")
    probabilities = best.predict_proba(features)
    # Population order makes IDs useful within a report, without implying that
    # cluster IDs are stable across different corpora or checkpoint versions.
    order = np.argsort(-np.bincount(probabilities.argmax(axis=1), minlength=best.n_components), kind="stable")
    probabilities = probabilities[:, order]
    labels = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-15))).sum(axis=1)
    entropy /= np.log(best.n_components) if best.n_components > 1 else 1
    rng = np.random.default_rng(42)
    selection = rng.choice(len(x), min(1500, len(x)), replace=False)
    unique = np.unique(labels[selection])
    silhouette = None
    if 1 < len(unique) < len(selection):
        silhouette = float(silhouette_score(features[selection], labels[selection]))
    projection = np.zeros((len(x), 3))
    projection[:, :min(3, projected.shape[1])] = projected[:, :3]
    eligible = sorted(c["bic"] for c in candidates if c["converged"] and np.isfinite(c["bic"]))
    diagnostics = {
        "method": "pca_gmm_bic_v2", "space": "standardized_latent_pca",
        "dimensions": dimensions, "retained_variance": float(pca.explained_variance_ratio_[:dimensions].sum()),
        "covariance": best.covariance_type, "selected_k": best.n_components,
        "max_k": limit, "at_search_boundary": best.n_components == limit,
        "bic": best_bic, "bic_gap": eligible[1] - eligible[0] if len(eligible) > 1 else None,
        "candidates": candidates, "silhouette": silhouette,
        "silhouette_samples": len(selection) if silhouette is not None else 0,
        "mean_confidence": float(confidence.mean()),
        "ambiguous_count": int((confidence < .8).sum()), "confidence_threshold": .8,
        "mean_entropy": float(entropy.mean()), "seed": 42,
    }
    return {"labels": labels, "probabilities": probabilities, "projection": projection,
            "projection_variance": pca.explained_variance_ratio_[:3].tolist(),
            "diagnostics": diagnostics, "features": features, "entropy": entropy}
