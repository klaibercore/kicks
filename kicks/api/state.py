"""Loaded synthesis state: one model, vocoder and slider basis per instrument.

Instruments are loaded lazily and kept, so a server started for kicks can serve
snares on request without a restart — at the cost of one model's memory each.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from ..analysis.basis import SliderBasis, analyze_latent_space
from ..analysis.descriptors import descriptor_vector
from ..analysis.evaluation import Reference, build_reference
from ..analysis.latents import extract_latents
from ..config import get_device, load_vae_from_checkpoint
from ..data import DrumDataset
from ..instruments import DEFAULT_INSTRUMENT, InstrumentProfile, get_profile
from ..nn import VAE
from ..audio.vocoder import load_vocoder


@dataclass
class InstrumentState:
    """Everything needed to synthesise one instrument."""

    profile: InstrumentProfile
    model: VAE
    basis: SliderBasis
    dataset: DrumDataset
    data_dir: str
    eval_ref: Reference | None = None

    def reference(self) -> Reference:
        """Corpus metric distributions, built on first use."""
        if self.eval_ref is None:
            self.eval_ref = build_reference(self.data_dir, self.profile)
        return self.eval_ref

    def measure(self, z: np.ndarray, device: torch.device) -> list[float]:
        """Descriptors of what the decoder actually produces for a latent.

        The closed-loop descriptor solver needs the decoder's real response, not
        the linear model's prediction of it.
        """
        zt = torch.tensor(z, dtype=torch.float32).to(device)
        with torch.no_grad():
            spec = self.model.decode(zt)
        return list(descriptor_vector(spec, self.profile))


class ServerState:
    """Process-wide synthesis state, populated during the app lifespan."""

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.vocoder: object = None
        self.vocoder_type: str = "bigvgan"
        self.control_basis: str = "pca"
        self.default_instrument: str = DEFAULT_INSTRUMENT
        self._instruments: dict[str, InstrumentState] = {}

    # -- lifecycle ----------------------------------------------------------

    def startup(
        self,
        instrument: str | None = None,
        vocoder_type: str | None = None,
        control_basis: str | None = None,
    ) -> InstrumentState:
        self.device = get_device()
        self.vocoder_type = vocoder_type or os.environ.get("KICKS_VOCODER", "bigvgan")
        self.control_basis = control_basis or os.environ.get("KICKS_CONTROL", "pca")
        profile = get_profile(instrument)
        self.default_instrument = profile.name
        self.vocoder = load_vocoder(
            self.device, self.vocoder_type, profile.paths.vocoder_dir,
        )
        return self.load(profile.name)

    @property
    def loaded(self) -> list[str]:
        return list(self._instruments)

    def get(self, name: str | None = None) -> InstrumentState:
        """Return an instrument's state, loading it on first request."""
        key = get_profile(name or self.default_instrument).name
        if key not in self._instruments:
            self._instruments[key] = self.load(key)
        return self._instruments[key]

    def load(self, name: str) -> InstrumentState:
        """Load a corpus, checkpoint and slider basis for one instrument."""
        from torch.utils.data import DataLoader

        profile = get_profile(name)
        data_dir = profile.paths.data_dir
        print(f"Loading {profile.display_name} from {data_dir}...")

        dataset = DrumDataset(data_dir, profile)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        model, _ = load_vae_from_checkpoint(profile.paths.checkpoint, self.device)

        latents, spectrograms = extract_latents(model, loader, self.device)
        print(f"Fitting the {self.control_basis} slider basis...")
        basis = analyze_latent_space(
            latents, spectrograms, profile,
            basis=self.control_basis, model=model,
        )
        for i, name_ in enumerate(basis.names):
            print(f"  {name_} range: [{basis.mins[i]:.3f}, {basis.maxs[i]:.3f}]")

        state = InstrumentState(
            profile=profile, model=model, basis=basis,
            dataset=dataset, data_dir=data_dir,
        )
        self._instruments[profile.name] = state
        return state
