"""Cheap validation detail measurements on the model's log-mel representation."""
from __future__ import annotations

import math

import librosa
import torch

from ..audio.constants import FMIN, FMAX, LOG_MEL_MAX, LOG_MEL_MIN, SAMPLE_RATE


class DetailMetrics:
    """Accumulate bin-weighted errors above the reference's -60 dB activity floor.

    These measure decoded mel detail, not post-vocoder waveform fidelity.
    Accumulating sums/counts makes the result independent of validation batching.
    """
    names = ("hf_mae_db", "air_mae_db", "attack_mae_db")

    def __init__(self, n_mels, transient):
        centres = torch.tensor(librosa.mel_frequencies(n_mels + 2, fmin=FMIN,
                                fmax=FMAX or SAMPLE_RATE / 2)[1:-1])
        self.bands = ((centres >= 2000) & (centres < 16000),
                      (centres >= 8000) & (centres < 16000),
                      torch.arange(n_mels) >= transient.band)
        self.attack_frames = transient.click_frames
        self.totals = torch.zeros(3, 2, dtype=torch.float64)

    def update(self, recon, target):
        scale_db = (LOG_MEL_MAX - LOG_MEL_MIN) * 20 / math.log(10)
        active = (target > target.amax(dim=(1, 2, 3), keepdim=True) - 60 / scale_db) & (target > 1e-6)
        errors = (recon - target).abs() * scale_db
        stats = []
        for index, band in enumerate(self.bands):
            mask = active & band.to(target.device)[None, None, :, None]
            if index == 2:
                mask = mask.clone()
                mask[..., self.attack_frames:] = False
            stats.append(torch.stack([(errors * mask).sum(), mask.sum().to(errors.dtype)]))
        self.totals += torch.stack(stats).detach().cpu().double()

    def compute(self):
        return {name: float(total / count) if count > 0 else None
                for name, (total, count) in zip(self.names, self.totals)}
