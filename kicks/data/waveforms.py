"""Drum corpus as raw waveforms with descriptor labels.

The waveform diffusion backend learns audio directly, so it needs the corpus in
the time domain rather than as spectrograms. Loading still goes through
:mod:`kicks.audio.io`, so the mono / resample / fit-length / LUFS chain is
literally the one the VAE was trained on and the two corpora cannot drift apart.

One step is added on top: each hit is scaled to a fixed peak. Diffusion learns
an absolute signal distribution, and a corpus whose hits sit anywhere between
-30 and -6 dBFS spends capacity modelling gain. The kick's descriptors are
ratios and a time centroid, so they are gain-invariant and a fixed peak changes
none of them — but that is a property of the kick profile, not a guarantee, so
labels are measured on the *scaled* waveform the model is actually trained on.
Whatever the descriptor kinds, the label and the target then agree by
construction.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..analysis.descriptors import descriptor_vector
from ..audio import io, mel
from ..audio.constants import AUDIO_LENGTH, N_FRAMES, SAMPLE_RATE
from ..instruments import InstrumentProfile, get_profile

#: Peak every training hit is scaled to. Leaves headroom for the sampler's own
#: overshoot without normalising away the transient.
TARGET_PEAK = 0.9

#: Identity of the chain above, recorded in checkpoints and training records.
PREPROCESSING = "peak_safe_lufs_then_peak_0.9_v1"


def peak_normalize(audio: torch.Tensor, peak: float = TARGET_PEAK) -> torch.Tensor:
    """Scale to a fixed absolute peak. Silence is returned untouched."""
    highest = float(audio.abs().max())
    return audio if highest <= 0 else audio * (peak / highest)


def load_hit(
    path: str,
    meter=None,
    length: int = AUDIO_LENGTH,
    peak: float = TARGET_PEAK,
) -> torch.Tensor | None:
    """Load one .wav as a (1, ``length``) waveform at a fixed peak, or None."""
    audio = io.load_waveform(path, meter, length=length, sample_rate=SAMPLE_RATE)
    return None if audio is None else peak_normalize(audio, peak)


def label_waveform(
    audio: torch.Tensor, profile: InstrumentProfile, n_frames: int = N_FRAMES,
) -> np.ndarray:
    """Descriptor values of one waveform, in the profile's own units and order."""
    return descriptor_vector(mel.spectrogram(audio, n_frames), profile)


class WaveformDataset(Dataset):
    """Every .wav in a directory as ``(waveform, descriptor labels)`` pairs.

    Args:
        directory: corpus directory. Defaults to the profile's ``data_dir``.
        profile: instrument profile; supplies the corpus path and descriptors.
        length: waveform length in samples, matching the model being trained.
        peak: absolute peak every hit is scaled to.
        verbose: print a one-line summary after loading.
    """

    def __init__(
        self,
        directory: str | None = None,
        profile: InstrumentProfile | None = None,
        length: int = AUDIO_LENGTH,
        peak: float = TARGET_PEAK,
        verbose: bool = True,
    ) -> None:
        self.profile = profile or get_profile()
        self.dir = directory or self.profile.paths.data_dir
        self.length = length
        self.peak = peak
        self.waveforms: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.paths: list[str] = []

        meter = io.make_meter()
        for name in io.list_wavs(self.dir):
            path = os.path.join(self.dir, name)
            audio = load_hit(path, meter, length, peak)
            if audio is None:
                continue
            self.waveforms.append(audio)
            self.labels.append(
                torch.from_numpy(label_waveform(audio, self.profile)).float(),
            )
            self.paths.append(path)

        if not self.waveforms:
            raise RuntimeError(f"No readable .wav files found in {self.dir}")
        if verbose:
            print(f"Loaded {len(self.waveforms)} {self.profile.plural} from {self.dir}, "
                  f"{self.length} samples, "
                  f"{len(self.profile.descriptors)} descriptor labels per hit")

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.waveforms[idx], self.labels[idx]

    def label_matrix(self, indices=None) -> torch.Tensor:
        """Descriptor rows over ``indices`` (all by default), one hit per row."""
        chosen = range(len(self)) if indices is None else list(indices)
        if not chosen:
            raise ValueError("label statistics need at least one sample")
        return torch.stack([self.labels[i] for i in chosen])

    def label_stats(self, indices=None) -> tuple[np.ndarray, np.ndarray]:
        """Descriptor mean and standard deviation over ``indices`` (all by default).

        Pass the *training* split's indices: normalising with statistics that
        have seen the validation hits leaks them into the conditioning.
        """
        stacked = self.label_matrix(indices).numpy()
        return stacked.mean(axis=0), stacked.std(axis=0)
