"""Device selection and checkpoint loading.

Paths no longer live here — they belong to an instrument and are reached through
``get_profile(...).paths``. Only the root directories are still environment-
overridable (``KICKS_DATA_DIR``, ``KICKS_MODEL_DIR``, ``KICKS_OUTPUT_DIR``), and
the registry applies those when it hands out a profile.
"""

from __future__ import annotations

import torch

from .audio.constants import N_FRAMES, N_MELS
from .nn import VAE, WaveformUNet
from .nn.diffusion import MODEL_KIND as DIFFUSION_KIND

#: Fallback latent size for checkpoints predating shape metadata.
DEFAULT_LATENT_DIM = 32


def get_device() -> torch.device:
    """Detect the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vae_from_checkpoint(
    checkpoint_path: str, device: torch.device,
) -> tuple[VAE, dict]:
    """Load a VAE, auto-detecting latent size and spectrogram shape.

    Shape metadata is written by :meth:`VAE.checkpoint_meta`, but older
    checkpoints predate it, so both are also recoverable from the weights: the
    latent size from ``fc_mu``, the frame count from the decoder's input width.
    The ``architecture`` block (residual blocks, latent skips) is honoured when
    present and defaults to the shipped layout when absent.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)

    latent_dim = checkpoint.get("latent_dim")
    if latent_dim is None:
        latent_dim = (
            state["fc_mu.weight"].shape[0] if "fc_mu.weight" in state
            else DEFAULT_LATENT_DIM
        )

    n_mels = checkpoint.get("n_mels", N_MELS)
    n_frames = checkpoint.get("n_frames")
    if n_frames is None:
        # fc_decode maps latent -> 256 * (n_mels/16) * (n_frames/16)
        flat = state["fc_decode.weight"].shape[0] if "fc_decode.weight" in state else None
        n_frames = (
            16 * flat // (256 * (n_mels // 16)) if flat else N_FRAMES
        )

    # Structural options arrived with the high-fidelity experiments; checkpoints
    # written before them carry no ``architecture`` block and mean the defaults.
    architecture = checkpoint.get("architecture") or {}
    model = VAE(
        latent_dim=latent_dim, n_mels=n_mels, n_frames=n_frames,
        residual=bool(architecture.get("residual", False)),
        latent_skips=bool(architecture.get("latent_skips", False)),
        soft_logvar=bool(architecture.get("soft_logvar", False)),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint


def load_diffusion_from_checkpoint(
    checkpoint_path: str, device: torch.device, profile=None,
) -> tuple[WaveformUNet, dict]:
    """Load a waveform diffusion denoiser from its recorded architecture.

    Unlike the VAE loader there is no shape inference to fall back on: this
    backend is new, so every checkpoint carries its own ``architecture`` block
    and one that does not is not ours. When a ``profile`` is given, the stored
    descriptor keys must match it exactly — a checkpoint conditioned on kick
    sliders would otherwise accept snare values and quietly mean something else.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != DIFFUSION_KIND:
        raise ValueError(
            f"{checkpoint_path} is not a {DIFFUSION_KIND} checkpoint "
            f"(model_kind={checkpoint.get('model_kind')!r})",
        )
    if profile is not None:
        expected = [d.key for d in profile.descriptors]
        stored = list(checkpoint.get("descriptors") or [])
        if stored and stored != expected:
            raise ValueError(
                f"{checkpoint_path} is conditioned on {stored}, "
                f"but {profile.name} expects {expected}",
            )

    architecture = dict(checkpoint["architecture"])
    model = WaveformUNet(
        n_descriptors=checkpoint["n_descriptors"],
        length=checkpoint["length"],
        channels=tuple(architecture.pop("channels")),
        factors=tuple(architecture.pop("factors")),
        **architecture,
    )
    model.load_state_dict(checkpoint["model"])
    model.set_label_bank(checkpoint.get("label_bank"))
    model.to(device)
    model.eval()
    return model, checkpoint
