"""2D convolutional VAE over normalized log-mel spectrograms.

The architecture is instrument-agnostic — a snare spectrogram is the same shape
as a kick spectrogram, and what differs between drum types lives in the corpus
and the profile, not the network. The spectrogram size is a constructor
argument rather than a constant so a future short-tail instrument (a closed hat
resolves in under 100 ms) can train on fewer frames without a second model class.

Two structural options exist for the high-fidelity experiments
(``docs/high-fidelity-generation.md``, stage 3). Both default to off, and with
both off the module tree — and therefore every ``state_dict`` key — is exactly
the shipped one, so existing checkpoints load unchanged:

- ``residual`` adds a zero-initialised residual block after every encoder and
  decoder stage, so intermediate detail has a path around each stride-2 conv.
- ``latent_skips`` injects the latent code at every decoder scale as a FiLM
  modulation (per-channel scale and shift), giving the fine-resolution layers
  direct access to the sound's identity instead of only what survives the
  bottleneck's upsampling chain. It is a *generative* skip: no encoder features
  are needed, so sampling a fresh latent works the same way as reconstruction.

Both start as the identity, so a run with a flag on begins from the same
function as one with it off and any difference is learned, not initialised.
The chosen options are written into checkpoints as ``architecture`` and read
back by ``config.load_vae_from_checkpoint``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.constants import N_FRAMES, N_MELS

#: Channel widths of the four stride-2 encoder stages.
_CHANNELS = (32, 64, 128, 256)
#: Each stage halves both axes, so the bottleneck is input / 2**4.
_DOWNSAMPLE = 2 ** len(_CHANNELS)


class ResidualBlock(nn.Module):
    """``relu(x + BN(conv(relu(BN(conv(x))))))`` with the last BN scale zeroed.

    Zero-initialising the final normalisation makes the block an identity at
    the start of training (the "zero-init residual" recipe), so switching it on
    changes nothing until the optimiser decides it should.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.body[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x))


class LatentFiLM(nn.Module):
    """Per-channel ``h * (1 + scale(z)) + shift(z)`` from the latent code.

    Both projections start at zero, so the modulation is the identity at
    initialisation and the decoder begins as the plain upsampling chain.
    """

    def __init__(self, latent_dim: int, channels: int) -> None:
        super().__init__()
        self.scale = nn.Linear(latent_dim, channels)
        self.shift = nn.Linear(latent_dim, channels)
        for linear in (self.scale, self.shift):
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale = self.scale(z)[:, :, None, None]
        shift = self.shift(z)[:, :, None, None]
        return h * (1 + scale) + shift


class VAE(nn.Module):
    """Conv VAE on (B, 1, n_mels, n_frames) spectrograms in [0, 1].

    ``latent_dim=32`` is small enough to resist posterior collapse while still
    holding an expressive drum synthesis space; it is recorded in the checkpoint
    and auto-detected on load, so changing it means retraining.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        n_mels: int = N_MELS,
        n_frames: int = N_FRAMES,
        residual: bool = False,
        latent_skips: bool = False,
    ) -> None:
        super().__init__()
        if n_mels % _DOWNSAMPLE or n_frames % _DOWNSAMPLE:
            raise ValueError(
                f"spectrogram dims must be divisible by {_DOWNSAMPLE}, got {n_mels}x{n_frames}"
            )
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.residual = residual
        self.latent_skips = latent_skips
        self._bottleneck = (_CHANNELS[-1], n_mels // _DOWNSAMPLE, n_frames // _DOWNSAMPLE)

        encoder: list[nn.Module] = []
        in_ch = 1
        for out_ch in _CHANNELS:
            encoder += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
            if residual:
                encoder.append(ResidualBlock(out_ch))
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder)

        self._enc_flat = math.prod(self._bottleneck)
        self.fc_mu = nn.Linear(self._enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self._enc_flat)

        # The decoder stays one Sequential so the shipped key layout survives;
        # latent injection points are remembered as indices into it.
        decoder: list[nn.Module] = []
        inject_after: list[int] = []
        film: list[nn.Module] = []
        rev = list(reversed(_CHANNELS))
        for i, in_ch in enumerate(rev):
            out_ch = rev[i + 1] if i + 1 < len(rev) else 1
            decoder.append(nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            if out_ch == 1:
                decoder.append(nn.Sigmoid())
            else:
                decoder += [nn.BatchNorm2d(out_ch), nn.ReLU()]
                if residual:
                    decoder.append(ResidualBlock(out_ch))
                if latent_skips:
                    inject_after.append(len(decoder) - 1)
                    film.append(LatentFiLM(latent_dim, out_ch))
        self.decoder = nn.Sequential(*decoder)
        self._inject_after = inject_after
        if latent_skips:
            self.film = nn.ModuleList(film)

    @property
    def architecture(self) -> dict[str, object]:
        """The structural choices, as stored in checkpoints and run records."""
        return {
            "residual": self.residual,
            "latent_skips": self.latent_skips,
            "channels": list(_CHANNELS),
        }

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-10, max=10)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(z.size(0), *self._bottleneck)
        if not self.latent_skips:
            return self.decoder(h)
        skip = 0
        for index, layer in enumerate(self.decoder):
            h = layer(h)
            if skip < len(self._inject_after) and index == self._inject_after[skip]:
                h = self.film[skip](h, z)
                skip += 1
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def checkpoint_meta(self) -> dict[str, object]:
        """Shape and architecture metadata to store alongside the weights."""
        return {
            "latent_dim": self.latent_dim,
            "n_mels": self.n_mels,
            "n_frames": self.n_frames,
            "architecture": self.architecture,
        }
