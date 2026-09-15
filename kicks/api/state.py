"""Loaded synthesis state: one model and slider basis per instrument, and the
vocoder each instrument's profile asks for.

Instruments are loaded lazily and kept, so a server started for kicks can serve
snares on request without a restart — at the cost of one model's memory each.
Vocoders are cached by backend and weights directory, so two instruments that
agree on a backend share one copy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ..analysis.basis import SliderBasis
from ..analysis.calibration import fit_or_load_basis
from ..analysis.descriptors import DecoderResponse
from ..analysis.evaluation import Reference, build_reference
from ..config import get_device, load_vae_from_checkpoint
from ..data import DrumDataset
from ..instruments import DEFAULT_INSTRUMENT, InstrumentProfile, get_profile
from ..nn import VAE
from ..audio.vocoder import load_vocoder, resolve_vocoder_type


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

    @property
    def response(self) -> DecoderResponse:
        return DecoderResponse(self.model, self.profile)


class ServerState:
    """Process-wide synthesis state, populated during the app lifespan."""

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        #: ``KICKS_VOCODER`` / ``--vocoder``: force one backend for every instrument.
        self.vocoder_override: str | None = None
        self.control_basis: str = "descriptor"
        self.default_instrument: str = DEFAULT_INSTRUMENT
        self._instruments: dict[str, InstrumentState] = {}
        self._vocoders: dict[tuple[str, str | None], object] = {}

    # -- lifecycle ----------------------------------------------------------

    def startup(
        self,
        instrument: str | None = None,
        vocoder_type: str | None = None,
        control_basis: str | None = None,
    ) -> InstrumentState:
        self.device = get_device()
        self.vocoder_override = vocoder_type or os.environ.get("KICKS_VOCODER") or None
        if self.vocoder_override:
            resolve_vocoder_type(None, self.vocoder_override)  # fail fast on a typo
        self.control_basis = control_basis or os.environ.get("KICKS_CONTROL", "descriptor")
        if self.control_basis not in ("pca", "descriptor"):
            raise ValueError("KICKS_CONTROL must be 'descriptor' or 'pca'")
        profile = get_profile(instrument)
        self.default_instrument = profile.name
        return self.load(profile.name)

    @property
    def loaded(self) -> list[str]:
        return list(self._instruments)

    @property
    def vocoder_type(self) -> str:
        """What ``/health`` reports: the forced backend, or ``profile`` when each
        instrument uses its own."""
        return self.vocoder_override or "profile"

    def vocoder_type_for(self, profile: InstrumentProfile) -> str:
        return resolve_vocoder_type(profile, self.vocoder_override)

    def vocoder_for(self, profile: InstrumentProfile):
        """The loaded backend for one instrument, shared where profiles agree.

        BigVGAN's fine-tuned weights are per instrument (``paths.vocoder_dir``),
        so the cache key carries the directory; DisCoder and Griffin-Lim have
        one set of weights and key on the backend alone.
        """
        kind = self.vocoder_type_for(profile)
        key = (kind, profile.paths.vocoder_dir if kind == "bigvgan" else None)
        if key not in self._vocoders:
            self._vocoders[key] = load_vocoder(self.device, kind, profile.paths.vocoder_dir)
        return self._vocoders[key]

    @property
    def vocoders(self) -> dict[str, str]:
        """Loaded instrument -> backend, for ``/health``."""
        return {name: self.vocoder_type_for(inst.profile) for name, inst in self._instruments.items()}

    def get(self, name: str | None = None) -> InstrumentState:
        """Return an instrument's state, loading it on first request."""
        key = get_profile(name or self.default_instrument).name
        if key not in self._instruments:
            self._instruments[key] = self.load(key)
        return self._instruments[key]

    def load(self, name: str) -> InstrumentState:
        """Load a corpus, checkpoint and slider basis for one instrument."""
        profile = get_profile(name)
        data_dir = profile.paths.data_dir
        print(f"Loading {profile.display_name} from {data_dir}...")

        model, _ = load_vae_from_checkpoint(profile.paths.checkpoint, self.device)
        dataset = DrumDataset(data_dir, profile, n_frames=model.n_frames)
        print(f"Fitting the {self.control_basis} slider basis...")
        basis = fit_or_load_basis(
            model, dataset, profile, self.device, mode=self.control_basis,
        )
        for i, name_ in enumerate(basis.names):
            print(f"  {name_} range: [{basis.mins[i]:.3f}, {basis.maxs[i]:.3f}]")
        self.vocoder_for(profile)  # load its backend now, not on the first render

        state = InstrumentState(
            profile=profile, model=model, basis=basis,
            dataset=dataset, data_dir=data_dir,
        )
        self._instruments[profile.name] = state
        return state
