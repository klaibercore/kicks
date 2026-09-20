"""Descriptor-conditioned waveform diffusion: the denoiser and its schedule.

This is the alternative backend proposed in issue #3. A 1-D U-Net denoises a raw
waveform directly, so neither the VAE nor a vocoder sits in the generation path.
The representation is the project's own audio contract (mono, 44.1 kHz,
``AUDIO_LENGTH`` samples) rather than the reference implementation's 0.74 s
stereo window, so the corpus, the profile's descriptor windows and every audit
tool carry over unchanged. Nothing here has been trained: the module defines the
experiment, it does not report a result.

Parameterisation is v-prediction on an angular schedule (Salimans & Ho). For a
noise level ``sigma`` in [0, 1]::

    alpha = cos(sigma * pi / 2)      beta = sin(sigma * pi / 2)
    x_noisy = alpha * x + beta * noise
    v       = alpha * noise - beta * x

``sigma = 0`` is clean audio and ``sigma = 1`` is pure noise. ``alpha**2 +
beta**2 == 1`` at every level, which is what makes :func:`v_sample` a
deterministic walk down the schedule instead of a stochastic one — two hits that
share a texture seed and differ only in their sliders stay comparable.

Conditioning is FiLM. The noise level and the instrument's descriptor vector are
embedded, summed, and turned into a per-channel scale and shift inside every
residual block. Descriptors enter in *raw* profile units and are standardised by
buffers written during training, so a caller never has to remember which
normalisation a checkpoint was trained with. A learned null embedding replaces
the descriptors when conditioning is dropped, which is what makes
classifier-free guidance available at sampling time.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..audio.constants import AUDIO_LENGTH, SAMPLE_RATE

#: Marks a checkpoint as this backend's, so a VAE loader never silently accepts one.
MODEL_KIND = "waveform_diffusion"

#: The reference configuration's feature widths. Eight stages of stride 2 take
#: the project's 65,536-sample window down to 256 frames at the bottleneck —
#: the same bottleneck length the reference reaches from its shorter window,
#: which is the minimal adaptation to a window twice as long.
DEFAULT_CHANNELS = (32, 32, 64, 64, 128, 128, 256, 256)
DEFAULT_FACTORS = (2, 2, 2, 2, 2, 2, 2, 2)
DEFAULT_ATTENTION_SCALES = 3


# ---------------------------------------------------------------------------
# Angular v-diffusion schedule
# ---------------------------------------------------------------------------

def alpha_beta(sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Signal and noise weights for a noise level in [0, 1]."""
    angle = sigmas * (math.pi / 2)
    return torch.cos(angle), torch.sin(angle)


def _expand(sigmas: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Shape a per-sample noise level so it broadcasts over ``like``."""
    return sigmas.reshape(-1, *([1] * (like.dim() - 1))).to(like.dtype)


def diffuse(x: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Mix clean audio with noise at per-sample levels ``sigmas``."""
    alpha, beta = alpha_beta(_expand(sigmas, x))
    return alpha * x + beta * noise


def v_target(x: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """The velocity the denoiser is trained to predict."""
    alpha, beta = alpha_beta(_expand(sigmas, x))
    return alpha * noise - beta * x


def guided_v(model, x: torch.Tensor, sigmas: torch.Tensor, labels, guidance: float):
    """One denoiser call, blending the conditional and unconditional velocities.

    ``guidance == 1`` is ordinary conditional prediction and costs one forward
    pass; anything else costs two.
    """
    if labels is None:
        return model(x, sigmas, labels=None)
    conditional = model(x, sigmas, labels=labels)
    if guidance == 1.0:
        return conditional
    unconditional = model(x, sigmas, labels=None)
    return unconditional + guidance * (conditional - unconditional)


@torch.no_grad()
def v_sample(
    model,
    *,
    labels: torch.Tensor | None = None,
    batch: int | None = None,
    steps: int = 50,
    guidance: float = 1.0,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Deterministic sampler: walk ``sigma`` from 1 to 0 in ``steps`` hops.

    Each step reads the predicted clean signal and the predicted noise out of
    one velocity, then re-mixes them at the next noise level. No fresh noise is
    drawn along the way, so the output is a function of the starting ``noise``
    (the texture seed) and the labels alone.

    ``noise`` is drawn on the CPU from ``generator`` and then moved, so a seed
    reproduces the same hit on CPU, CUDA and MPS alike.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    device = device or next(model.parameters()).device
    if noise is None:
        if batch is None:
            batch = 1 if labels is None else int(labels.shape[0])
        noise = torch.randn(batch, 1, model.length, generator=generator)
    x = noise.to(device)
    if labels is not None:
        labels = labels.to(device=device, dtype=x.dtype)
        if labels.shape[0] != x.shape[0]:
            raise ValueError(
                f"{labels.shape[0]} label rows for {x.shape[0]} samples",
            )

    was_training = model.training
    model.eval()
    try:
        levels = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=x.dtype)
        for step in range(steps):
            alpha, beta = alpha_beta(levels[step])
            next_alpha, next_beta = alpha_beta(levels[step + 1])
            velocity = guided_v(
                model, x, levels[step].expand(x.shape[0]), labels, guidance,
            )
            predicted_x = alpha * x - beta * velocity
            predicted_noise = beta * x + alpha * velocity
            x = next_alpha * predicted_x + next_beta * predicted_noise
    finally:
        model.train(was_training)
    return x


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _groups(channels: int) -> int:
    """GroupNorm groups that divide any width, including the tiny test ones."""
    return math.gcd(8, channels)


def _resample(in_channels: int, out_channels: int, factor: int, up: bool) -> nn.Module:
    """Change width, and length by ``factor``, in one convolution.

    A kernel of ``2 * factor`` with padding ``factor // 2`` divides (or
    multiplies) the length exactly, which is why factors must be 1 or even.
    """
    if factor == 1:
        return nn.Conv1d(in_channels, out_channels, 3, padding=1)
    kernel, padding = 2 * factor, factor // 2
    layer = nn.ConvTranspose1d if up else nn.Conv1d
    return layer(in_channels, out_channels, kernel, stride=factor, padding=padding)


class FourierFeatures(nn.Module):
    """Noise level -> embedding, through learned random Fourier features."""

    def __init__(self, dim: int, n_features: int = 16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features))
        self.project = nn.Linear(2 * n_features + 1, dim)

    def forward(self, sigmas: torch.Tensor) -> torch.Tensor:
        level = sigmas.reshape(-1, 1)
        angles = level * self.weight[None] * (2 * math.pi)
        return self.project(torch.cat([level, angles.sin(), angles.cos()], dim=-1))


class ResBlock(nn.Module):
    """Residual convolution block, modulated by the conditioning embedding.

    The output convolution and the FiLM projection start at zero, so a freshly
    built block is the identity and adding depth cannot make a new model worse
    at initialization.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels
            else nn.Identity()
        )
        for zeroed in (self.film, self.conv2):
            nn.init.zeros_(zeroed.weight)
            nn.init.zeros_(zeroed.bias)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(embedding).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.conv2(F.silu(h)) + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over time, at the coarse scales only."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.heads = heads if channels % heads == 0 else 1
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.project = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        head_dim = channels // self.heads
        qkv = self.qkv(self.norm(x)).reshape(batch, 3, self.heads, head_dim, length)
        query, key, value = (part.transpose(-2, -1) for part in qkv.unbind(1))
        attended = F.scaled_dot_product_attention(query, key, value)
        merged = attended.transpose(-2, -1).reshape(batch, channels, length)
        return x + self.project(merged)


class Stage(nn.Module):
    """Residual blocks at one width, optionally followed by self-attention."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 blocks: int, attention: bool, heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResBlock(in_channels if i == 0 else out_channels, out_channels, cond_dim)
            for i in range(blocks)
        )
        self.attention = SelfAttention(out_channels, heads) if attention else None

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, embedding)
        return x if self.attention is None else self.attention(x)


class WaveformUNet(nn.Module):
    """1-D U-Net denoiser conditioned on noise level and descriptor values.

    Args:
        n_descriptors: length of the profile's descriptor vector.
        channels: feature width per scale.
        factors: length reduction per scale; 1 or even.
        cond_dim: width of the conditioning embedding.
        blocks_per_scale: residual blocks in each encoder and decoder stage.
        attention_scales: how many of the deepest scales get self-attention.
        attention_heads: heads in each attention block.
        length: waveform length in samples; must divide by ``prod(factors)``.
    """

    def __init__(
        self,
        n_descriptors: int,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        factors: tuple[int, ...] = DEFAULT_FACTORS,
        cond_dim: int = 128,
        blocks_per_scale: int = 2,
        attention_scales: int = DEFAULT_ATTENTION_SCALES,
        attention_heads: int = 4,
        length: int = AUDIO_LENGTH,
    ) -> None:
        super().__init__()
        channels, factors = tuple(channels), tuple(factors)
        if not channels or len(channels) != len(factors):
            raise ValueError("channels and factors must be non-empty and the same length")
        if any(f != 1 and f % 2 for f in factors):
            raise ValueError("every resampling factor must be 1 or even")
        total = math.prod(factors)
        if length % total:
            raise ValueError(f"length {length} is not divisible by the total factor {total}")
        if blocks_per_scale < 1:
            raise ValueError("blocks_per_scale must be positive")

        self.n_descriptors = n_descriptors
        self.channels = channels
        self.factors = factors
        self.cond_dim = cond_dim
        self.blocks_per_scale = blocks_per_scale
        self.attention_scales = max(0, min(attention_scales, len(channels)))
        self.attention_heads = attention_heads
        self.length = length

        self.noise_embedding = FourierFeatures(cond_dim)
        self.label_embedding = nn.Sequential(
            nn.Linear(n_descriptors, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim),
        )
        self.null_embedding = nn.Parameter(torch.zeros(cond_dim))
        self.conditioning = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        # Raw profile units in, standardised units inside. Written once from the
        # training split so inference cannot drift from what was learned.
        self.register_buffer("label_mean", torch.zeros(n_descriptors))
        self.register_buffer("label_std", torch.ones(n_descriptors))
        # The training split's descriptor rows, in raw units, for drawing
        # generation targets that lie on the corpus rather than around its
        # moments. Not a buffer: its length is the split's, so it travels in the
        # checkpoint beside the weights rather than inside the state dict.
        self.label_bank: torch.Tensor | None = None

        depth = len(channels)
        deepest = depth - self.attention_scales
        self.stem = nn.Conv1d(1, channels[0], 5, padding=2)
        self.downsample = nn.ModuleList()
        self.down = nn.ModuleList()
        previous = channels[0]
        for scale, (width, factor) in enumerate(zip(channels, factors)):
            self.downsample.append(_resample(previous, width, factor, up=False))
            self.down.append(Stage(width, width, cond_dim, blocks_per_scale,
                                   scale >= deepest, attention_heads))
            previous = width

        self.mid_in = ResBlock(channels[-1], channels[-1], cond_dim)
        self.mid_attention = SelfAttention(channels[-1], attention_heads)
        self.mid_out = ResBlock(channels[-1], channels[-1], cond_dim)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for scale in reversed(range(depth)):
            below = channels[0] if scale == 0 else channels[scale - 1]
            self.up.append(Stage(2 * channels[scale], channels[scale], cond_dim,
                                 blocks_per_scale, scale >= deepest, attention_heads))
            self.upsample.append(_resample(channels[scale], below, factors[scale], up=True))

        self.final = Stage(2 * channels[0], channels[0], cond_dim,
                           blocks_per_scale, False, attention_heads)
        self.head_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.head = nn.Conv1d(channels[0], 1, 5, padding=2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # -- conditioning --------------------------------------------------------

    def set_label_stats(self, mean, std) -> None:
        """Record the training split's descriptor mean and standard deviation."""
        mean = torch.as_tensor(mean, dtype=self.label_mean.dtype).reshape(-1)
        std = torch.as_tensor(std, dtype=self.label_std.dtype).reshape(-1)
        if mean.numel() != self.n_descriptors or std.numel() != self.n_descriptors:
            raise ValueError("label statistics must have one entry per descriptor")
        self.label_mean.copy_(mean.to(self.label_mean.device))
        # A descriptor that never varies would otherwise divide by zero.
        self.label_std.copy_(std.clamp_min(1e-6).to(self.label_std.device))

    def set_label_bank(self, bank) -> None:
        """Record the training split's descriptor rows; ``None`` clears them."""
        if bank is None:
            self.label_bank = None
            return
        bank = torch.as_tensor(bank, dtype=torch.float32).detach().cpu().reshape(-1, self.n_descriptors)
        if bank.shape[0] == 0:
            raise ValueError("a label bank needs at least one row")
        self.label_bank = bank.clone()

    def embed(self, sigmas: torch.Tensor, labels, cond_mask, batch: int) -> torch.Tensor:
        noise = self.noise_embedding(sigmas)
        if labels is None:
            label = self.null_embedding.expand(batch, -1)
        else:
            standardized = (labels - self.label_mean) / self.label_std
            label = self.label_embedding(standardized)
            if cond_mask is not None:
                keep = cond_mask.reshape(-1, 1).to(torch.bool)
                label = torch.where(keep, label, self.null_embedding.expand_as(label))
        return self.conditioning(noise + label)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        sigmas,
        labels: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the velocity of ``x`` at noise level ``sigmas``.

        ``labels`` are raw descriptor values, or None for the unconditional
        branch. ``cond_mask`` keeps conditioning per sample (False drops it),
        which is how conditioning dropout is applied during training.
        """
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(f"expected a (batch, 1, samples) waveform, got {tuple(x.shape)}")
        if x.shape[-1] != self.length:
            raise ValueError(f"expected {self.length} samples, got {x.shape[-1]}")
        batch = x.shape[0]
        if not torch.is_tensor(sigmas):
            sigmas = torch.full((batch,), float(sigmas))
        sigmas = sigmas.to(device=x.device, dtype=x.dtype).reshape(-1)
        if sigmas.numel() == 1:
            sigmas = sigmas.expand(batch)

        embedding = self.embed(sigmas, labels, cond_mask, batch)
        h = self.stem(x)
        skips = [h]
        for downsample, stage in zip(self.downsample, self.down):
            h = stage(downsample(h), embedding)
            skips.append(h)
        h = self.mid_out(self.mid_attention(self.mid_in(h, embedding)), embedding)
        for stage, upsample in zip(self.up, self.upsample):
            h = upsample(stage(torch.cat([h, skips.pop()], dim=1), embedding))
        h = self.final(torch.cat([h, skips.pop()], dim=1), embedding)
        return self.head(F.silu(self.head_norm(h)))

    # -- persistence ---------------------------------------------------------

    @property
    def architecture(self) -> dict:
        return {
            "channels": list(self.channels),
            "factors": list(self.factors),
            "cond_dim": self.cond_dim,
            "blocks_per_scale": self.blocks_per_scale,
            "attention_scales": self.attention_scales,
            "attention_heads": self.attention_heads,
        }

    def checkpoint_meta(self) -> dict:
        """Everything needed to rebuild this network from its weights alone."""
        return {
            "model_kind": MODEL_KIND,
            "architecture": self.architecture,
            "n_descriptors": self.n_descriptors,
            "length": self.length,
            "sample_rate": SAMPLE_RATE,
        }
