"""VAE loss: multi-resolution frequency-weighted reconstruction + KL divergence.

The reconstruction term is instrument-agnostic. The transient term is not — it
guards a specific mel band over a specific stretch of frames, and where the
transient lives is exactly what differs between a kick's beater click, a snare's
wire crack and a hi-hat's stick attack. Those windows arrive as a
:class:`~kicks.instruments.TransientLossSpec` from the instrument profile.

Two optional terms exist for the high-fidelity experiments
(``docs/high-fidelity-generation.md``, stage 2) and are off by default so the
shipped objective is unchanged: :func:`hf_detail_loss` is a *symmetric*,
reference-weighted error over the instrument's HF band across the whole audible
hit — the transient term only penalises excess HF in the tail, so a decoder
that drops a real HF tail pays nothing there — and :func:`attack_change_loss`
matches frame-to-frame change over the attack, where smearing is heard first.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..audio.constants import LOG_MEL_MAX, LOG_MEL_MIN
from ..instruments import TransientLossSpec

#: Normalised-mel units per dB: the spectrogram maps ``[LOG_MEL_MIN, LOG_MEL_MAX]``
#: natural-log magnitudes onto [0, 1], so this converts a dB span to that scale.
_UNITS_PER_DB = math.log(10) / 20 / (LOG_MEL_MAX - LOG_MEL_MIN)

# Cached frequency weight tensors per (n_mels, device, alpha) to avoid re-creation.
_freq_weight_cache: dict[tuple[int, str, float], torch.Tensor] = {}


def _frequency_weights(
    n_mels: int, device: torch.device, alpha: float = 0.0,
) -> float | torch.Tensor:
    """Linearly decaying frequency weights: 1.0 at bin 0, (1-alpha) at the top bin.

    With alpha=0.0 (default) weights are flat, so high-frequency detail is not
    penalized less than low-frequency detail — important for crisp,
    non-metallic transients. Returns a scalar 1.0 in that case to skip the
    broadcast entirely; otherwise shape (1, 1, n_mels, 1).
    """
    if alpha == 0.0:
        return 1.0
    key = (n_mels, str(device), alpha)
    if key not in _freq_weight_cache:
        w = torch.linspace(1.0, 1.0 - alpha, n_mels, device=device)
        _freq_weight_cache[key] = w.view(1, 1, n_mels, 1)
    return _freq_weight_cache[key]


def spectral_convergence(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral convergence loss: Frobenius norm ratio."""
    return torch.norm(target - recon, p="fro") / (torch.norm(target, p="fro") + 1e-8)


def _temporal_weights(target: torch.Tensor, floor: float = 0.1) -> torch.Tensor:
    """Per-frame weights from the target's own energy, in a [floor, 1] range.

    A one-shot occupies only a fraction of the 256 frames; the rest is padded
    silence. Weighting by per-frame energy focuses the loss on the actual hit
    instead of averaging it away over the silent tail. Shape (B, 1, 1, T).
    """
    frame_energy = target.mean(dim=2, keepdim=True)
    w = frame_energy / (frame_energy.amax(dim=-1, keepdim=True) + 1e-8)
    return w.clamp(min=floor)


def multi_resolution_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
    freq_alpha: float = 0.0,
    temporal_floor: float = 0.1,
) -> torch.Tensor:
    """Multi-resolution frequency-weighted reconstruction loss.

    At each scale, computes frequency-weighted, energy-weighted L1 plus spectral
    convergence on avg-pooled spectrograms — catching both fine transient detail
    (scale 1) and the global spectral envelope (coarser scales). Set
    ``temporal_floor=1.0`` to disable the energy weighting.
    """
    total = torch.tensor(0.0, device=recon.device)
    temporal_w = _temporal_weights(target, floor=temporal_floor)
    for s in scales:
        if s > 1:
            r = F.avg_pool2d(recon, kernel_size=s)
            t = F.avg_pool2d(target, kernel_size=s)
            tw = F.avg_pool2d(temporal_w, kernel_size=(1, s))
        else:
            r, t, tw = recon, target, temporal_w
        n_mels = t.shape[2]
        freq_w = _frequency_weights(n_mels, t.device, alpha=freq_alpha)
        # Weighted mean: normalize by the active weight mass, not the frame count.
        weighted_l1 = (freq_w * tw * (r - t).abs()).sum() / (tw.sum() * n_mels + 1e-8)
        total = total + weighted_l1 + spectral_convergence(r, t)
    return total


def transient_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    spec: TransientLossSpec,
) -> torch.Tensor:
    """High-frequency transient fidelity term.

    Targets the two perceptual failures the evaluator keeps finding: a missing
    high-frequency click (the VAE over-smooths and averages the bright transient
    away) and high-frequency energy smeared into the tail.

    The click term is a plain L1 — reproduce the transient exactly. The tail term
    is one-sided, penalizing only *excess* high end the target does not have, so
    genuinely long-tailed samples are not pushed toward silence.
    """
    click = (
        recon[..., spec.band:, :spec.click_frames]
        - target[..., spec.band:, :spec.click_frames]
    ).abs().mean()
    tail_excess = (
        recon[..., spec.band:, spec.tail_start:]
        - target[..., spec.band:, spec.tail_start:]
    ).clamp(min=0).mean()
    return click + tail_excess


def hf_detail_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    spec: TransientLossSpec,
    floor_db: float = 60.0,
) -> torch.Tensor:
    """Symmetric, reference-weighted high-frequency detail error over the whole hit.

    Every bin at or above ``spec.band`` is weighted by how audible it is in the
    *reference*: a linear ramp from 0 at ``floor_db`` below the hit's peak to 1
    at the peak. Missing and excess energy cost the same, so a decoder cannot
    drop a real HF tail for free (the transient term's one-sided tail penalty
    lets it). Bins the reference does not excite have weight 0: the term asks
    for the detail that is there, not for a raised noise floor — the transient
    term keeps guarding the tail against that.

    Normalised by the weight mass, so a bright hit and a dull one contribute on
    the same scale.
    """
    hf_recon = recon[..., spec.band:, :]
    hf_target = target[..., spec.band:, :]
    peak = target.amax(dim=(1, 2, 3), keepdim=True)
    weight = ((hf_target - (peak - floor_db * _UNITS_PER_DB)) / (floor_db * _UNITS_PER_DB)).clamp(0, 1)
    return (weight * (hf_recon - hf_target).abs()).sum() / (weight.sum() + 1e-8)


def attack_change_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    spec: TransientLossSpec,
) -> torch.Tensor:
    """Temporal-change matching over the attack window.

    Compares first differences along time (frame ``t+1`` minus frame ``t``) for
    the first ``spec.click_frames`` transitions, across all mel bins. A smeared
    onset has the right average level but the wrong slope; this term sees the
    slope directly, which a per-bin L1 over the same window does not.
    """
    n = spec.click_frames + 1
    d_recon = recon[..., 1:n] - recon[..., :n - 1]
    d_target = target[..., 1:n] - target[..., :n - 1]
    return (d_recon - d_target).abs().mean()


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
    free_bits: float = 0.5,
    transient: TransientLossSpec | None = None,
    transient_weight: float | None = None,
    hf_detail_weight: float = 0.0,
    attack_change_weight: float = 0.0,
    terms: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Beta-VAE loss: multi-resolution reconstruction + beta * KL with free bits.

    Per-dimension KL is clamped to ``free_bits`` nats before summing, preventing
    individual latent dimensions from collapsing to the prior (posterior
    collapse). ``transient_weight`` overrides the profile's own weight; 0
    disables the transient term. ``hf_detail_weight`` and
    ``attack_change_weight`` switch on the stage-2 experiment terms; both need
    ``transient`` for their windows.

    Returns (total, reconstruction, kl) for separate logging; the transient,
    HF-detail and attack-change terms are folded into the reconstruction figure.
    Pass ``terms`` to also receive each *unweighted* component as a float, for
    the training record.
    """
    recon_loss = multi_resolution_loss(recon, x)
    components = {"multi_resolution": recon_loss}
    if transient is not None:
        weight = transient.weight if transient_weight is None else transient_weight
        if weight > 0:
            components["transient"] = transient_loss(recon, x, transient)
            recon_loss = recon_loss + weight * components["transient"]
        if hf_detail_weight > 0:
            components["hf_detail"] = hf_detail_loss(recon, x, transient)
            recon_loss = recon_loss + hf_detail_weight * components["hf_detail"]
        if attack_change_weight > 0:
            components["attack_change"] = attack_change_loss(recon, x, transient)
            recon_loss = recon_loss + attack_change_weight * components["attack_change"]
    elif hf_detail_weight > 0 or attack_change_weight > 0:
        raise ValueError("hf_detail_weight and attack_change_weight need a TransientLossSpec")
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(0)
    kl = kl_per_dim.clamp(min=free_bits).sum()
    if terms is not None:
        terms.update({name: float(value.detach()) for name, value in components.items()})
    return recon_loss + beta * kl, recon_loss, kl
