"""Centralized configuration for the kicks project."""

import os

import torch


# Paths (overridable via environment variables)
DATA_DIR = os.environ.get("KICKS_DATA_DIR", "data/kicks")
MODEL_DIR = os.environ.get("KICKS_MODEL_DIR", "models")
OUTPUT_DIR = os.environ.get("KICKS_OUTPUT_DIR", "output")
BEST_CHECKPOINT = os.path.join(MODEL_DIR, "best.pth")

# Model defaults
LATENT_DIM = 32
N_PCS = 4


def get_device() -> torch.device:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
