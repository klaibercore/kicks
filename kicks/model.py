"""2D Convolutional VAE for log-mel spectrogram kick drum synthesis."""

import torch
import torch.nn as nn

SAMPLE_RATE = 44100
AUDIO_LENGTH = 65536  # ~1.49s at 44100 Hz
N_FFT = 1024        # Must match BigVGAN (was 2048)
HOP_LENGTH = 256
WIN_SIZE = 1024     # Must match BigVGAN
N_MELS = 128
FMIN = 0
FMAX = None         # Nyquist
# Spectrogram shape: (1, 128, 256)


class VAE(nn.Module):
    """2D Conv VAE operating on log-mel spectrograms (1, 128, 256)."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (B, 1, 128, 256) → (B, 256, 8, 16) via 4x stride-2 downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self._enc_flat = 256 * 8 * 16  # 32768
        self.fc_mu = nn.Linear(self._enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat, latent_dim)

        # Decoder: z → (B, 256, 8, 16) then 4x upsample to (B, 1, 128, 256)
        self.fc_decode = nn.Linear(latent_dim, self._enc_flat)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-10, max=10)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 16)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
