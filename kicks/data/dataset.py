"""Drum corpus as normalized log-mel spectrograms.

One dataset class serves every instrument: what is loaded is decided by the
directory, and how it is interpreted is decided by the profile downstream. The
pre-processing chain (mono -> resample -> fit length -> LUFS -> BigVGAN log-mel
-> fixed-bounds [0, 1]) is shared with the latent prior via
:mod:`kicks.audio.io`, so the two cannot drift apart.
"""

from __future__ import annotations

import os

import pyloudnorm as pyln
import torch
from torch.utils.data import Dataset

from ..audio import io, mel
from ..audio.constants import AUDIO_LENGTH, N_FRAMES, SAMPLE_RATE
from ..instruments import InstrumentProfile, get_profile


def load_spectrogram(
    path: str,
    meter: pyln.Meter | None = None,
    n_frames: int = N_FRAMES,
) -> torch.Tensor | None:
    """Load one .wav as a normalized (1, n_mels, n_frames) spectrogram in [0, 1].

    Returns None for unreadable files. Pass a shared ``meter`` when looping over
    a corpus — constructing one per file dominates load time.
    """
    audio = io.load_waveform(path, meter, length=AUDIO_LENGTH, sample_rate=SAMPLE_RATE)
    if audio is None:
        return None
    return mel.spectrogram(audio, n_frames)


class DrumDataset(Dataset):
    """Every .wav in a directory, as spectrogram tensors held in memory.

    Args:
        directory: corpus directory. Defaults to the profile's ``data_dir``.
        profile: instrument profile; only used for its default corpus path.
        n_frames: spectrogram width, matching the model being trained.
        verbose: print a one-line summary after loading.
    """

    def __init__(
        self,
        directory: str | None = None,
        profile: InstrumentProfile | None = None,
        n_frames: int = N_FRAMES,
        verbose: bool = True,
    ) -> None:
        self.profile = profile or get_profile()
        self.dir = directory or self.profile.paths.data_dir
        self.n_frames = n_frames
        self.tensors: list[torch.Tensor] = []
        self.paths: list[str] = []

        meter = io.make_meter()
        for name in io.list_wavs(self.dir):
            path = os.path.join(self.dir, name)
            spec = load_spectrogram(path, meter, n_frames)
            if spec is None:
                continue
            self.tensors.append(spec)
            self.paths.append(path)

        if not self.tensors:
            raise RuntimeError(f"No readable .wav files found in {self.dir}")
        if verbose:
            print(f"Loaded {len(self.tensors)} {self.profile.plural} from {self.dir}, "
                  f"spectrogram shape: {tuple(self.tensors[0].shape)}")

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]
