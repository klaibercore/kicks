"""Neural network architectures."""

from .diffusion import WaveformUNet
from .vae import VAE

__all__ = ["VAE", "WaveformUNet"]
