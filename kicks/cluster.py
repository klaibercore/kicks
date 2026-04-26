"""Latent space clustering utilities for kick VAE."""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


def extract_latents(
    model, dataloader, device: torch.device
):
    """Extract latent mu vectors and spectrograms from the trained model."""
    model.eval()
    latents = []
    spectrograms = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latents.append(mu.cpu().numpy())
            spectrograms.append(batch.cpu())
    return np.concatenate(latents), torch.cat(spectrograms)


def select_n_clusters(latents: np.ndarray, max_k: int = 10) -> tuple[int, list[float]]:
    """Select optimal GMM component count via BIC (lower is better)."""
    max_k = min(max_k, max(2, len(latents) // 3))
    k_range = range(2, max_k + 1)
    bics = []
    latents = latents.astype(np.float64)
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=3, reg_covar=1e-4)
        gmm.fit(latents)
        bics.append(gmm.bic(latents))
    best_k = list(k_range)[int(np.argmin(bics))]
    return best_k, list(bics)


def fit_gmm(
    latents: np.ndarray, n_clusters: int
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Fit GMM and return model, hard labels, and soft probabilities."""
    latents = latents.astype(np.float64)
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type="full", random_state=42, n_init=3, reg_covar=1e-4
    )
    gmm.fit(latents)
    labels = gmm.predict(latents)
    probs = gmm.predict_proba(latents)
    return gmm, labels, probs


def compute_descriptors(spec_tensor: torch.Tensor) -> dict[str, float]:
    """Derive perceptual descriptors from a normalized log-mel spectrogram.

    Spectrogram shape: (1, 128, 256) - 128 mel bands, 256 time frames
    Normalized to [0, 1] (BigVGAN log-mel scale)
    
    Mel band frequency mapping (approximate):
    - Bands 0-12:  ~20-80 Hz (sub-bass)
    - Bands 12-30: ~80-250 Hz (bass)
    - Bands 30-50: ~250-1k Hz (low-mids)
    - Bands 50-128: ~1k-8k Hz (highs)

    Time frame mapping (at 44.1kHz/256hop = ~5.8ms/frame, 256 frames = ~1.49s):
    - Frames 0-3:    Transient/attack (~0-17ms) - EXCLUDED for stability
    - Frames 3-30:   Body (~17-174ms)
    - Frames 30+:    Decay tail
    
    Excludes transient (first 3 frames) from most calculations for stability.
    Decay uses a broadband (bands 0-40) early-to-late energy ratio for a
    robust measure of sustain that spreads naturally across the full range.
    """
    import numpy as np
    
    spec = spec_tensor.squeeze().numpy()  # (128, 256)
    
    # Use frames after transient for more stable measurements
    body = spec[:, 3:]  # Exclude first 3 frames (transient)
    
    # Sub-bass energy: lowest frequency bands (20-80Hz) in body
    # This measures the sustained sub content, not transient
    sub = float(spec[:12, 3:].mean())
    
    # Punch: ratio of attack transient to body energy
    # Compare transient (frames 0-3) to body (frames 3-30)
    transient_energy = spec[10:30, :3].mean()
    body_energy = spec[10:30, 3:30].mean()
    ratio = transient_energy / (body_energy + 1e-8)
    punch = float(np.clip(np.log(max(ratio, 1e-8)) / 2.0, 0.0, 1.0))
    
    # Click/transient: high frequency energy in first few frames
    # This is specifically what we want to measure - the initial transient
    click = float(spec[40:100, :3].mean())
    
    # Brightness: high frequency content in body (excluding transient)
    high_energy = spec[50:, 3:].mean()
    low_energy = spec[:30, 3:].mean()
    bright = float(high_energy / (low_energy + high_energy + 1e-8))
    
    # Decay: broadband early-to-late energy ratio.
    # Compares energy in the body (frames ~17-174ms, bands ~20-350Hz)
    # to the tail (frames ~174-700ms). Higher = faster decay (acoustic
    # thump), lower = longer sustain (808-style). The 1 - ratio
    # formulation gives a natural spread across [0, 1].
    early = spec[:40, 3:30].mean()   # bands 0-40, frames 3-30
    late = spec[:40, 30:120].mean()  # bands 0-40, frames 30-120
    ratio = np.clip(late / (early + 1e-8), 0, 1)
    decay = float(1.0 - ratio)

    return {
        "sub": float(sub),
        "punch": float(punch),
        "click": float(click),
        "bright": float(bright),
        "decay": decay,
    }
