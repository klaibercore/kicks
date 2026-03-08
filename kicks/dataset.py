"""Dataset for loading kick drum audio as normalized log-mel spectrograms.

Uses BigVGAN's mel_spectrogram() to ensure the representation matches
what the vocoder was trained on (n_fft=1024, magnitude spectrum, ln clamp).
"""

import os
import warnings

import numpy as np
import pyloudnorm as pyln
import torch
import soundfile as sf
import torchaudio
from bigvgan import mel_spectrogram as bigvgan_mel_spectrogram
from torch.utils.data import Dataset

from .model import SAMPLE_RATE, AUDIO_LENGTH, N_FFT, HOP_LENGTH, WIN_SIZE, N_MELS, FMIN, FMAX

# Fixed bounds for normalization — derived from BigVGAN's mel_spectrogram output.
# ln(1e-5) = -11.5129 is the silence floor (BigVGAN clamps magnitudes to 1e-5).
# 2.5 provides headroom above the observed dataset max (~2.23).
LOG_MEL_MIN = -11.5129
LOG_MEL_MAX = 2.5

# Target integrated loudness for LUFS normalization
TARGET_LUFS = -14.0


class KickDataset(Dataset):
    """Loads .wav kick samples and converts to normalized log-mel spectrograms.

    Pre-processing: LUFS loudness normalization of input audio.
    Mel computation: BigVGAN's mel_spectrogram (magnitude spectrum + ln clamp).
    Normalization: fixed bounds [-11.51, 2.5] mapped to [0, 1].
    Returns tensors of shape (1, 128, 256).
    """

    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.tensors: list[torch.Tensor] = []
        self.paths: list[str] = []

        self._lufs_meter = pyln.Meter(SAMPLE_RATE)

        for file in sorted(os.listdir(dir)):
            if not file.endswith(".wav"):
                continue
            path = os.path.join(dir, file)
            data, sr = sf.read(path, dtype="float32")
            # sf.read returns (samples,) for mono or (samples, channels) for stereo
            if data.ndim == 1:
                audio = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
            else:
                audio = torch.from_numpy(data.T)  # (channels, samples)

            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Resample to target sample rate
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                audio = resampler(audio)

            # Pad/truncate to fixed length
            length = audio.shape[-1]
            if length > AUDIO_LENGTH:
                audio = audio[:, :AUDIO_LENGTH]
            elif length < AUDIO_LENGTH:
                audio = torch.nn.functional.pad(audio, (0, AUDIO_LENGTH - length))

            # LUFS loudness normalization
            audio_np = audio.squeeze(0).numpy()
            loudness = self._lufs_meter.integrated_loudness(audio_np)
            if np.isfinite(loudness):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Possible clipped samples", module="pyloudnorm")
                    audio_np = pyln.normalize.loudness(audio_np, loudness, TARGET_LUFS)
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio = torch.from_numpy(audio_np).unsqueeze(0).float()

            # Compute log-mel spectrogram using BigVGAN's function.
            # This produces magnitude-based log mel with ln(clamp(mel, min=1e-5)).
            log_mel = bigvgan_mel_spectrogram(
                audio, N_FFT, N_MELS, SAMPLE_RATE, HOP_LENGTH, WIN_SIZE,
                FMIN, FMAX, center=False,
            )  # (1, N_MELS, T)

            # Pad/truncate to 256 frames
            if log_mel.shape[-1] > 256:
                log_mel = log_mel[:, :, :256]
            elif log_mel.shape[-1] < 256:
                log_mel = torch.nn.functional.pad(log_mel, (0, 256 - log_mel.shape[-1]))

            # Normalize to [0, 1] using fixed bounds
            log_mel = torch.clamp(log_mel, min=LOG_MEL_MIN, max=LOG_MEL_MAX)
            normalized = (log_mel - LOG_MEL_MIN) / (LOG_MEL_MAX - LOG_MEL_MIN)

            self.tensors.append(normalized)
            self.paths.append(path)

        if not self.tensors:
            raise RuntimeError(f"No .wav files found in {dir}")
        print(f"Loaded {len(self.tensors)} samples, spectrogram shape: {self.tensors[0].shape}")

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]

    @staticmethod
    def denormalize(spec: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] normalized spectrogram back to BigVGAN log-mel scale."""
        return spec * (LOG_MEL_MAX - LOG_MEL_MIN) + LOG_MEL_MIN
