"""VAE loss: multi-resolution frequency-weighted reconstruction + KL divergence."""

import torch
import torch.nn.functional as F

# Cached frequency weight tensors per (n_mels, device) to avoid re-creation.
_freq_weight_cache: dict[tuple[int, str], torch.Tensor] = {}


def _frequency_weights(n_mels: int, device: torch.device, alpha: float = 0.5) -> torch.Tensor:
    """Linearly decaying frequency weights: 1.0 at bin 0, (1-alpha) at top bin.

    Returns shape (1, 1, n_mels, 1) for broadcasting over (B, C, F, T).
    """
    key = (n_mels, str(device))
    if key not in _freq_weight_cache:
        w = torch.linspace(1.0, 1.0 - alpha, n_mels, device=device)
        _freq_weight_cache[key] = w.view(1, 1, n_mels, 1)
    return _freq_weight_cache[key]


def spectral_convergence(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral convergence loss: Frobenius norm ratio."""
    return torch.norm(target - recon, p="fro") / (torch.norm(target, p="fro") + 1e-8)


def multi_resolution_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
    freq_alpha: float = 0.5,
) -> torch.Tensor:
    """Multi-resolution frequency-weighted reconstruction loss.

    At each scale, computes frequency-weighted L1 + spectral convergence on
    avg-pooled spectrograms. Catches both fine transient detail (scale 1) and
    global spectral envelope (coarser scales).
    """
    total = torch.tensor(0.0, device=recon.device)
    for s in scales:
        if s > 1:
            r = F.avg_pool2d(recon, kernel_size=s)
            t = F.avg_pool2d(target, kernel_size=s)
        else:
            r, t = recon, target
        n_mels = t.shape[2]
        weights = _frequency_weights(n_mels, t.device, alpha=freq_alpha)
        weighted_l1 = (weights * (r - t).abs()).mean()
        sc = spectral_convergence(r, t)
        total = total + weighted_l1 + sc
    return total


def loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
    free_bits: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Beta-VAE loss: multi-resolution reconstruction + beta * KL with free bits.

    Per-dimension KL is clamped to `free_bits` nats before summing, preventing
    individual latent dimensions from collapsing to zero (posterior collapse).

    Returns (total_loss, recon_loss, kl) for separate logging.
    """
    batch_size = x.size(0)
    recon_loss = multi_resolution_loss(recon, x)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(0)
    kl = kl_per_dim.clamp(min=free_bits).sum()
    return recon_loss + beta * kl, recon_loss, kl
