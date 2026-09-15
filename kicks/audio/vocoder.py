"""Mel spectrogram -> audio.

Three backends, chosen per instrument by ``InstrumentProfile.vocoder`` (see
:func:`resolve_vocoder_type`):

* **DisCoder** — ETH DISCO's music vocoder, the official 44.1 kHz Z checkpoint
  with its fine-tuned DAC decoder. Same 128-band mel as BigVGAN, so no VAE
  retraining. Audits best on kicks and snares and renders ~2x faster on MPS.
* **BigVGAN** — neural, high quality, GPU recommended. Picks up instrument-
  specific fine-tuned weights when a checkpoint exists in the profile's
  ``vocoder_dir``. Audits best on hi-hats.
* **Griffin-Lim** — classical phase estimation, CPU-only, no model download.

All finish with the same post-chain: bandlimit, peak-normalize, then gate the
tail to true silence.
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

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
DISCODER_MODEL = "disco-eth/discoder"
DISCODER_REVISION = "6505384d8fd5f18338f171dd81dc10c9a0d34fe9"

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
    """Bandlimit, peak-normalize, gate the tail. Shared by all backends."""
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


def validate_discoder_config(config: dict) -> None:
    expected = {"n_fft": N_FFT, "win_length": WIN_SIZE, "hop_length": HOP_LENGTH,
                "n_mels": N_MELS, "f_min": FMIN, "f_max": FMAX}
    if config.get("sample_rate") != SAMPLE_RATE or any(
        config.get("mel", {}).get(key) != value for key, value in expected.items()
    ):
        raise ValueError("DisCoder checkpoint does not match the VAE's mel representation")


def load_discoder(device: torch.device):
    """Load the pinned official music checkpoint, including its DAC decoder."""
    from huggingface_hub import hf_hub_download
    from ..nn.discoder import DisCoder

    directory = Path(os.environ.get("KICKS_DISCODER_DIR", os.path.join(
        os.environ.get("KICKS_MODEL_DIR", "models"), "discoder",
    )))
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.pt"):
        if not (directory / name).exists():
            hf_hub_download(DISCODER_MODEL, name, revision=DISCODER_REVISION, local_dir=directory)
    config = json.loads((directory / "config.json").read_text())
    validate_discoder_config(config)
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    weights = {key.removeprefix("module."): value
               for key, value in checkpoint["model_state_dict"].items()}
    # Allocate directly from the mapped checkpoint; a second random 1.72 GB
    # parameter set would cause unnecessary swapping on smaller Macs.
    with torch.device("meta"):
        model = DisCoder(config)
    model.load_state_dict(weights, strict=True, assign=True)
    del weights, checkpoint
    model.remove_weight_norm()
    print(f"Loaded DisCoder from {directory} (official 44.1 kHz Z checkpoint)")
    return model.eval().requires_grad_(False).to(device)


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

VOCODERS = ("discoder", "bigvgan", "griffinlim")


def resolve_vocoder_type(profile=None, requested: str | None = None) -> str:
    """Which backend renders ``profile``: explicit request > ``KICKS_VOCODER`` > profile.

    The profile's own choice is the measured per-instrument default (DisCoder
    audits better on kicks and snares, BigVGAN on hi-hats); the two overrides
    exist to force one backend everywhere, e.g. Griffin-Lim on a CPU-only box.
    """
    chosen = requested or os.environ.get("KICKS_VOCODER") or (profile.vocoder if profile else "discoder")
    if chosen not in VOCODERS:
        raise ValueError(f"Unknown vocoder: {chosen!r} (expected one of {', '.join(VOCODERS)})")
    return chosen


def load_vocoder(
    device: torch.device,
    vocoder_type: str | None = None,
    weights_dir: str | None = None,
):
    """Load DisCoder, BigVGAN, or Griffin-Lim. ``None`` resolves via ``KICKS_VOCODER``."""
    vocoder_type = resolve_vocoder_type(None, vocoder_type)
    if vocoder_type == "discoder":
        return load_discoder(device)
    if vocoder_type == "griffinlim":
        return load_griffin_lim(device)
    if vocoder_type == "bigvgan":
        return load_bigvgan(device, weights_dir)
    raise ValueError(f"Unknown vocoder: {vocoder_type!r}")


def spec_to_audio(
    spec_normalized: torch.Tensor,
    vocoder,
    device: torch.device,
) -> torch.Tensor:
    """Normalized spectrogram -> waveform.

    Args:
        spec_normalized: (B, 1, n_mels, n_frames), values in [0, 1].
        vocoder: a DisCoder / BigVGAN model or a :class:`GriffinLimVocoder`.
        device: torch device (ignored by Griffin-Lim, which is CPU-only).

    Returns:
        (B, T) waveform tensor on the CPU.
    """
    log_mel = denormalize(spec_normalized.cpu()).squeeze(1)  # (B, n_mels, n_frames)
    with torch.no_grad():
        if isinstance(vocoder, GriffinLimVocoder):
            waveform = vocoder(log_mel)               # (B, T)
        else:
            batch_size = getattr(vocoder, "inference_batch_size", len(log_mel))
            waveform = torch.cat([
                vocoder(batch.to(device)).squeeze(1).cpu()
                for batch in log_mel.split(batch_size)
            ])
    return _post_process(waveform.cpu())
