"""Latent-space extraction and Gaussian mixture fitting."""

from __future__ import annotations

import numpy as np
import torch
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
    max_k = min(max_k, max(2, len(latents) // 3))
    k_range = range(2, max_k + 1)
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
