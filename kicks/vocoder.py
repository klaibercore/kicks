"""BigVGAN neural vocoder for mel spectrogram to audio conversion."""

import os

import torch
import torchaudio
import bigvgan

from .model import SAMPLE_RATE

BIGVGAN_MODEL = "nvidia/bigvgan_v2_44khz_128band_256x"
FINETUNED_PATH = "models/vocoder/best.pth"


def _patch_bigvgan_from_pretrained():
    """Patch BigVGAN._from_pretrained to work with huggingface_hub >= 1.0.

    Newer huggingface_hub versions removed the `proxies` and `resume_download`
    kwargs from the call to `_from_pretrained`, but bigvgan still declares them
    as required keyword-only arguments.
    """
    original = bigvgan.BigVGAN._from_pretrained.__func__

    @classmethod  # type: ignore[misc]
    def _patched(cls, **kwargs):
        kwargs.setdefault("proxies", None)
        kwargs.setdefault("resume_download", False)
        return original(cls, **kwargs)

    bigvgan.BigVGAN._from_pretrained = _patched


_patch_bigvgan_from_pretrained()


def load_vocoder(device: torch.device) -> bigvgan.BigVGAN:
    """Load BigVGAN vocoder. Uses fine-tuned weights if available, otherwise pretrained."""
    model = bigvgan.BigVGAN.from_pretrained(BIGVGAN_MODEL, use_cuda_kernel=False)

    if os.path.exists(FINETUNED_PATH):
        checkpoint = torch.load(FINETUNED_PATH, map_location=device)
        model.load_state_dict(checkpoint["generator"])
        print(f"Loaded fine-tuned vocoder (epoch {checkpoint['epoch']})")
    else:
        print("Using pretrained BigVGAN (run `kicks fine-tune` to improve)")

    model.remove_weight_norm()
    return model.eval().to(device)


def spec_to_audio(
    spec_normalized: torch.Tensor,
    dataset,
    vocoder: bigvgan.BigVGAN,
    device: torch.device,
) -> torch.Tensor:
    """Convert normalized spectrogram to audio waveform via BigVGAN.

    Args:
        spec_normalized: VAE output, shape (B, 1, 128, 256), values in [0, 1].
        dataset: KickDataset (kept for API compatibility, denormalize is now static).
        vocoder: Loaded BigVGAN model.
        device: Torch device.

    Returns:
        Waveform tensor, shape (B, T).
    """
    from .dataset import KickDataset, LOG_MEL_MIN
    log_mel = KickDataset.denormalize(spec_normalized.cpu())
    # Gate near-silence frames to the true silence floor.
    # The VAE's sigmoid output can't reach exact zero, so values meant to be
    # silent end up slightly above LOG_MEL_MIN. BigVGAN faithfully synthesizes
    # these as high-frequency artifacts. Snap them to the floor.
    silence_threshold = LOG_MEL_MIN + 2.0  # ~2 nats above silence floor
    log_mel = torch.where(log_mel < silence_threshold, LOG_MEL_MIN, log_mel)
    log_mel = log_mel.squeeze(1)  # (B, 128, 256)
    with torch.no_grad():
        waveform = vocoder(log_mel.to(device))  # (B, 1, T)
    waveform = waveform.squeeze(1).cpu()  # (B, T)
    waveform = torchaudio.functional.highpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=25.0)
    waveform = torchaudio.functional.lowpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=20000.0)
    # Fade-out to avoid abrupt cutoff artifacts at the end of the sample.
    fade_len = min(5000, waveform.shape[-1])
    fade = torch.linspace(1.0, 0.0, fade_len)
    waveform[..., -fade_len:] *= fade
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform
