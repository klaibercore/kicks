"""Mel spectrogram -> audio.

Two backends:

* **BigVGAN** — neural, high quality, GPU recommended. Picks up instrument-
  specific fine-tuned weights when a checkpoint exists in the profile's
  ``vocoder_dir``.
* **Griffin-Lim** — classical phase estimation, CPU-only, no model download.

Both finish with the same post-chain: bandlimit, peak-normalize, then gate the
tail to true silence.
"""

from __future__ import annotations

import glob
import math
import os

import torch
import torch.nn.functional as F
import torchaudio

from .constants import (
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WIN_SIZE,
)
from .mel import denormalize

BIGVGAN_MODEL = "nvidia/bigvgan_v2_44khz_128band_256x"

HIGHPASS_HZ = 25.0
LOWPASS_HZ = 20000.0


def gate_tail(
    waveform: torch.Tensor,
    threshold_db: float = -70.0,
    fade_ms: float = 80.0,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Fade the tail to digital silence once the envelope stays below threshold.

    Neural vocoders leave a low-level noise floor (~-90 dBFS) where real drum
    samples decay into true silence; the exposed hiss after the hit is the
    single most audible artefact. Expects a peak-normalized (B, T) waveform.
    """
    win = 1024
    env = torch.sqrt(
        F.avg_pool1d((waveform ** 2).unsqueeze(1), win, stride=1, padding=win // 2)
        .squeeze(1)[:, : waveform.shape[-1]]
        + 1e-12
    )
    thr = 10 ** (threshold_db / 20)
    fade = int(sample_rate * fade_ms / 1000)
    out = waveform.clone()
    for i in range(out.shape[0]):
        above = (env[i] > thr).nonzero()
        if len(above) == 0:
            continue
        last = int(above[-1])
        end = min(last + fade, out.shape[-1])
        if end - last > 0:
            ramp = 0.5 * (1 + torch.cos(torch.linspace(
                0, math.pi, end - last, device=out.device, dtype=out.dtype,
            )))
            out[i, last:end] *= ramp
        out[i, end:] = 0.0
    return out


def _post_process(waveform: torch.Tensor) -> torch.Tensor:
    """Bandlimit, peak-normalize, gate the tail. Shared by both backends."""
    dtype = waveform.dtype
    # Low-cutoff biquads lose precision in float32; keep the CPU filter state
    # in float64 so quiet and loud versions of a hit receive the same filter.
    waveform = waveform.to(torch.float64)
    waveform = torchaudio.functional.highpass_biquad(
        waveform, SAMPLE_RATE, cutoff_freq=HIGHPASS_HZ)
    waveform = torchaudio.functional.lowpass_biquad(
        waveform, SAMPLE_RATE, cutoff_freq=LOWPASS_HZ)
    waveform = waveform / waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return gate_tail(waveform.to(dtype))


# ---------------------------------------------------------------------------
# BigVGAN
# ---------------------------------------------------------------------------

_patched = False


def patch_bigvgan_from_pretrained() -> None:
    """Patch BigVGAN._from_pretrained to work with huggingface_hub >= 1.0. Idempotent."""
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


def load_bigvgan(device: torch.device, weights_dir: str | None = None):
    """Load BigVGAN, preferring fine-tuned weights in ``weights_dir``."""
    import bigvgan as _bigvgan

    patch_bigvgan_from_pretrained()
    model = _bigvgan.BigVGAN.from_pretrained(BIGVGAN_MODEL, use_cuda_kernel=False)

    pth_files = sorted(glob.glob(os.path.join(weights_dir, "*.pth"))) if weights_dir else []
    if pth_files:
        pth_path = pth_files[0]
        checkpoint = torch.load(pth_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["generator"])
        print(f"Loaded fine-tuned vocoder from {pth_path} (epoch {checkpoint['epoch']})")
    else:
        print("Using pretrained BigVGAN (fine-tuned weights are picked up from the profile's vocoder dir)")

    model.remove_weight_norm()
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Griffin-Lim
# ---------------------------------------------------------------------------

class GriffinLimVocoder:
    """Mel-to-audio via pseudo-inverse mel filterbank + Griffin-Lim.

    Uses pinverse of the mel filterbank instead of torchaudio's InverseMelScale,
    which relies on linalg_lstsq (unsupported on MPS and prone to rank errors).
    All computation runs on CPU.
    """

    def __init__(self, n_iter: int = 64):
        n_stft = N_FFT // 2 + 1
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_stft,
            f_min=FMIN,
            f_max=FMAX if FMAX is not None else SAMPLE_RATE / 2.0,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
        )  # (n_stft, n_mels); forward op is mel = fb.T @ linear
        self.fb_pinv = torch.linalg.pinv(fb.T)  # (n_stft, n_mels)
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT, win_length=WIN_SIZE, hop_length=HOP_LENGTH, n_iter=n_iter,
        )

    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        """(B, n_mels, T) log-mel -> (B, T) waveform."""
        mel_linear = torch.exp(log_mel).cpu()
        linear_spec = torch.clamp(self.fb_pinv @ mel_linear, min=0.0)
        # Bins that are exactly zero produce phase-estimation artefacts.
        linear_spec = linear_spec + 1e-4
        return self.griffin_lim(linear_spec)


def load_griffin_lim(device: torch.device) -> GriffinLimVocoder:
    """Create a Griffin-Lim vocoder (no model weights needed)."""
    print("Using Griffin-Lim vocoder (lower quality, no neural model)")
    return GriffinLimVocoder()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_vocoder(
    device: torch.device,
    vocoder_type: str = "bigvgan",
    weights_dir: str | None = None,
):
    """Load the requested backend: ``"bigvgan"`` (default) or ``"griffinlim"``."""
    if vocoder_type == "griffinlim":
        return load_griffin_lim(device)
    return load_bigvgan(device, weights_dir)


def spec_to_audio(
    spec_normalized: torch.Tensor,
    vocoder,
    device: torch.device,
) -> torch.Tensor:
    """Normalized spectrogram -> waveform.

    Args:
        spec_normalized: (B, 1, n_mels, n_frames), values in [0, 1].
        vocoder: a BigVGAN model or a :class:`GriffinLimVocoder`.
        device: torch device (ignored by Griffin-Lim, which is CPU-only).

    Returns:
        (B, T) waveform tensor on the CPU.
    """
    log_mel = denormalize(spec_normalized.cpu()).squeeze(1)  # (B, n_mels, n_frames)
    with torch.no_grad():
        if isinstance(vocoder, GriffinLimVocoder):
            waveform = vocoder(log_mel)               # (B, T)
        else:
            waveform = vocoder(log_mel.to(device)).squeeze(1)
    return _post_process(waveform.cpu())
