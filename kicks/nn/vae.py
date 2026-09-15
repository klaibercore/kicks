"""2D convolutional VAE over normalized log-mel spectrograms.

The architecture is instrument-agnostic — a snare spectrogram is the same shape
as a kick spectrogram, and what differs between drum types lives in the corpus
and the profile, not the network. The spectrogram size is a constructor
argument rather than a constant so a future short-tail instrument (a closed hat
resolves in under 100 ms) can train on fewer frames without a second model class.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..audio.constants import N_FRAMES, N_MELS

#: Channel widths of the four stride-2 encoder stages.
_CHANNELS = (32, 64, 128, 256)
#: Each stage halves both axes, so the bottleneck is input / 2**4.
_DOWNSAMPLE = 2 ** len(_CHANNELS)


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
    ) -> None:
        super().__init__()
        if n_mels % _DOWNSAMPLE or n_frames % _DOWNSAMPLE:
            raise ValueError(
                f"spectrogram dims must be divisible by {_DOWNSAMPLE}, got {n_mels}x{n_frames}"
            )
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.n_frames = n_frames
        self._bottleneck = (_CHANNELS[-1], n_mels // _DOWNSAMPLE, n_frames // _DOWNSAMPLE)

        encoder: list[nn.Module] = []
        in_ch = 1
        for out_ch in _CHANNELS:
            encoder += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder)

        self._enc_flat = math.prod(self._bottleneck)
        self.fc_mu = nn.Linear(self._enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self._enc_flat)

        decoder: list[nn.Module] = []
        rev = list(reversed(_CHANNELS))
        for i, in_ch in enumerate(rev):
            out_ch = rev[i + 1] if i + 1 < len(rev) else 1
            decoder.append(nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            if out_ch == 1:
                decoder.append(nn.Sigmoid())
            else:
                decoder += [nn.BatchNorm2d(out_ch), nn.ReLU()]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-10, max=10)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(z.size(0), *self._bottleneck)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def checkpoint_meta(self) -> dict[str, int]:
        """Shape metadata to store alongside the weights."""
        return {
            "latent_dim": self.latent_dim,
            "n_mels": self.n_mels,
            "n_frames": self.n_frames,
        }
