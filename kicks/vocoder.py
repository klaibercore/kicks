"""Vocoder implementations for mel spectrogram to audio conversion.

Supports two backends:
- BigVGAN: High-quality neural vocoder (default)
- Griffin-LIM: Classical phase-estimation algorithm (lower quality, no GPU needed)
"""

import glob
import os

import torch
import torchaudio

from .model import SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_SIZE, N_MELS, FMIN, FMAX

BIGVGAN_MODEL = "nvidia/bigvgan_v2_44khz_128band_256x"
VOCODER_DIR = "models/vocoder"


# ---------------------------------------------------------------------------
# BigVGAN
# ---------------------------------------------------------------------------

_patched = False


def _patch_bigvgan_from_pretrained():
    """Patch BigVGAN._from_pretrained to work with huggingface_hub >= 1.0."""
    global _patched
    if _patched:
        return
    import bigvgan as _bigvgan

    original = _bigvgan.BigVGAN._from_pretrained.__func__

    @classmethod  # type: ignore[misc]
    def _patched_fn(cls, **kwargs):
        kwargs.setdefault("proxies", None)
        kwargs.setdefault("resume_download", False)
        return original(cls, **kwargs)

    _bigvgan.BigVGAN._from_pretrained = _patched_fn
    _patched = True


def load_bigvgan(device: torch.device):
    """Load BigVGAN vocoder. Uses fine-tuned weights if available, otherwise pretrained."""
    import bigvgan as _bigvgan

    _patch_bigvgan_from_pretrained()

    model = _bigvgan.BigVGAN.from_pretrained(BIGVGAN_MODEL, use_cuda_kernel=False)

    pth_files = sorted(glob.glob(os.path.join(VOCODER_DIR, "*.pth")))
    if pth_files:
        pth_path = pth_files[0]
        checkpoint = torch.load(pth_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["generator"])
        print(f"Loaded fine-tuned vocoder from {pth_path} (epoch {checkpoint['epoch']})")
    else:
        print("Using pretrained BigVGAN (run `kicks fine-tune` to improve)")

    model.remove_weight_norm()
    return model.eval().to(device)


def _bigvgan_spec_to_audio(
    spec_normalized: torch.Tensor,
    vocoder,
    device: torch.device,
) -> torch.Tensor:
    from .dataset import KickDataset

    log_mel = KickDataset.denormalize(spec_normalized.cpu())
    log_mel = log_mel.squeeze(1)  # (B, 128, 256)
    with torch.no_grad():
        waveform = vocoder(log_mel.to(device))  # (B, 1, T)
    waveform = waveform.squeeze(1).cpu()  # (B, T)
    waveform = torchaudio.functional.highpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=25.0)
    waveform = torchaudio.functional.lowpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=20000.0)
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform


# ---------------------------------------------------------------------------
# Griffin-LIM
# ---------------------------------------------------------------------------

class GriffinLimVocoder:
    """Mel-to-audio via pseudo-inverse mel filterbank + Griffin-LIM.

    Uses pinverse of the mel filterbank instead of torchaudio's InverseMelScale
    which relies on linalg_lstsq (unsupported on MPS and prone to rank errors).
    All computation runs on CPU.
    """

    def __init__(self, n_iter: int = 64):
        n_stft = N_FFT // 2 + 1
        # Build mel filterbank and compute its pseudo-inverse once
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_stft,
            f_min=FMIN,
            f_max=FMAX if FMAX is not None else SAMPLE_RATE / 2.0,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
        )  # (n_stft, n_mels)
        # fb is (n_stft, n_mels); forward op is mel = fb.T @ linear
        # To invert: linear ≈ pinv(fb.T) @ mel, pinv shape = (n_stft, n_mels)
        self.fb_pinv = torch.linalg.pinv(fb.T)  # (n_stft, n_mels)

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT,
            win_length=WIN_SIZE,
            hop_length=HOP_LENGTH,
            n_iter=n_iter,
        )

    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Convert log-mel spectrogram to audio. Input shape (B, n_mels, T)."""
        mel_linear = torch.exp(log_mel).cpu()
        # mel_linear: (B, n_mels, T) -> linear_spec: (B, n_stft, T)
        linear_spec = torch.clamp(self.fb_pinv @ mel_linear, min=0.0)
        # Add a small noise floor to help Griffin-LIM converge — bins that
        # are exactly zero produce phase-estimation artefacts.
        linear_spec = linear_spec + 1e-4
        waveform = self.griffin_lim(linear_spec)  # (B, T)
        return waveform


def load_griffin_lim(device: torch.device) -> GriffinLimVocoder:
    """Create a Griffin-LIM vocoder (no model weights needed)."""
    print("Using Griffin-LIM vocoder (lower quality, no neural model)")
    return GriffinLimVocoder()


def _griffinlim_spec_to_audio(
    spec_normalized: torch.Tensor,
    vocoder: GriffinLimVocoder,
    device: torch.device,
) -> torch.Tensor:
    from .dataset import KickDataset

    log_mel = KickDataset.denormalize(spec_normalized.cpu())
    log_mel = log_mel.squeeze(1)  # (B, 128, 256)
    with torch.no_grad():
        waveform = vocoder(log_mel)  # (B, T)
    waveform = waveform.cpu()
    waveform = torchaudio.functional.highpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=25.0)
    waveform = torchaudio.functional.lowpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=20000.0)
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_vocoder(device: torch.device, vocoder_type: str = "bigvgan"):
    """Load the requested vocoder backend.

    Args:
        device: Torch device.
        vocoder_type: ``"bigvgan"`` (default) or ``"griffinlim"``.
    """
    if vocoder_type == "griffinlim":
        return load_griffin_lim(device)
    return load_bigvgan(device)


def spec_to_audio(
    spec_normalized: torch.Tensor,
    dataset,
    vocoder,
    device: torch.device,
) -> torch.Tensor:
    """Convert normalized spectrogram to audio waveform.

    Dispatches to the correct backend based on the vocoder type.

    Args:
        spec_normalized: VAE output, shape (B, 1, 128, 256), values in [0, 1].
        dataset: KickDataset (kept for API compatibility).
        vocoder: Loaded vocoder (BigVGAN model or GriffinLimVocoder).
        device: Torch device.

    Returns:
        Waveform tensor, shape (B, T).
    """
    if isinstance(vocoder, GriffinLimVocoder):
        return _griffinlim_spec_to_audio(spec_normalized, vocoder, device)
    return _bigvgan_spec_to_audio(spec_normalized, vocoder, device)
