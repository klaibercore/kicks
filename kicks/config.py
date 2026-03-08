"""Centralized configuration for the kicks project."""

import os

import torch


# Paths (overridable via environment variables)
DATA_DIR = os.environ.get("KICKS_DATA_DIR", "data/kicks")
MODEL_DIR = os.environ.get("KICKS_MODEL_DIR", "models")
OUTPUT_DIR = os.environ.get("KICKS_OUTPUT_DIR", "output")
BEST_CHECKPOINT = os.path.join(MODEL_DIR, "vae_best.pth")

# Model defaults
LATENT_DIM = 128
N_PCS = 4


def load_vae_from_checkpoint(
    checkpoint_path: str, device: torch.device,
) -> tuple["VAE", dict]:  # noqa: F821
    """Load a VAE from checkpoint, auto-detecting latent_dim."""
    from .model import VAE

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "latent_dim" in checkpoint:
        latent_dim = checkpoint["latent_dim"]
    elif "model" in checkpoint and "fc_mu.weight" in checkpoint["model"]:
        latent_dim = checkpoint["model"]["fc_mu.weight"].shape[0]
    else:
        latent_dim = LATENT_DIM
    model = VAE(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint


def get_device() -> torch.device:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
