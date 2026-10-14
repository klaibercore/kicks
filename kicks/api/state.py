"""Loaded synthesis state: one model and slider basis per instrument, and the
vocoder each instrument's profile asks for.

Instruments are loaded lazily and kept, so a server started for kicks can serve
snares on request without a restart — at the cost of one model's memory each.
Vocoders are cached by backend and weights directory, so two instruments that
agree on a backend share one copy.

The waveform diffusion backend is served beside the VAE, never instead of it,
and only for the instruments given a checkpoint (``kicks serve --diffusion
hihat=PATH``, i.e. ``KICKS_DIFFUSION``). It needs no vocoder and no slider
basis; its sliders map onto the checkpoint's own label bank.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

import torch

from ..analysis.basis import SliderBasis
from ..analysis.descriptors import DecoderResponse
from ..analysis.evaluation import Reference, build_reference
from ..config import get_device, load_diffusion_from_checkpoint, load_vae_from_checkpoint
from ..data import DrumDataset
from ..instruments import DEFAULT_INSTRUMENT, InstrumentProfile, get_profile
from ..nn import VAE, WaveformUNet
from ..audio.vocoder import load_vocoder, resolve_vocoder_type
from ..synthesis.controlled import fit_control_basis


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


@dataclass
class DiffusionState:
    """A served waveform diffusion checkpoint for one instrument."""

    profile: InstrumentProfile
    model: WaveformUNet
    checkpoint: str
    meta: dict
    data_dir: str
    eval_ref: Reference | None = None
    #: One sampling walk at a time: renders run off the event loop, and two
    #: concurrent walks would only share the device and both finish later.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def reference(self) -> Reference:
        if self.eval_ref is None:
            self.eval_ref = build_reference(self.data_dir, self.profile)
        return self.eval_ref

    @property
    def names(self) -> list[str]:
        """Slider labels, one per conditioning descriptor, in checkpoint order."""
        return [d.label for d in self.profile.descriptors]


@dataclass
class ProceduralState:
    """A physical voice, available without a corpus or checkpoint."""
    profile: InstrumentProfile

    @property
    def names(self) -> list[str]:
        return [p.label for p in self.profile.procedural.parameters]

    @property
    def keys(self) -> list[str]:
        # Namespaced: decay_ms and drive are also post-processing effects.
        return [f"voice_{p.key}" for p in self.profile.procedural.parameters]


@dataclass
class ResearchState:
    profile: InstrumentProfile
    engine: object
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def names(self):
        return [d.label for d in self.profile.descriptors]

    @property
    def model(self):
        return self.engine.model


def parse_research_checkpoints(spec):
    """INSTRUMENT:KIND=PATH entries; separate from active/promoted weights."""
    from ..nn.research import KINDS
    served = {}
    for item in (spec or '').split(','):
        if not item.strip(): continue
        identity,sep,path=item.partition('=')
        name,colon,kind=identity.partition(':')
        if not sep or not colon or not path.strip() or kind not in KINDS:
            raise ValueError('expected INSTRUMENT:KIND=PATH for a research backend')
        key=(get_profile(name.strip()).name,kind)
        if key in served: raise ValueError('duplicate research backend')
        served[key]=path.strip()
    return served


def parse_diffusion_checkpoints(spec: str | None) -> dict[str, str]:
    """``"hihat=a.pth,snare=b.pth"`` -> ``{"hihat": "a.pth", "snare": "b.pth"}``."""
    served: dict[str, str] = {}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        name, sep, path = item.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"expected INSTRUMENT=PATH, got {item.strip()!r}")
        served[get_profile(name.strip()).name] = path.strip()
    return served


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
        #: Instrument -> diffusion checkpoint served beside its VAE.
        self.diffusion_checkpoints: dict[str, str] = {}
        self._diffusion: dict[str, DiffusionState] = {}
        self.default_backend = "vae"
        self.research_checkpoints = {}
        self._research = {}

    # -- lifecycle ----------------------------------------------------------

    def startup(
        self,
        instrument: str | None = None,
        vocoder_type: str | None = None,
        control_basis: str | None = None,
        backend: str | None = None,
    ) -> InstrumentState | DiffusionState | ProceduralState | ResearchState:
        self.device = get_device()
        self.vocoder_override = vocoder_type or os.environ.get("KICKS_VOCODER") or None
        if self.vocoder_override:
            resolve_vocoder_type(None, self.vocoder_override)  # fail fast on a typo
        self.control_basis = control_basis or os.environ.get("KICKS_CONTROL", "descriptor")
        if self.control_basis not in ("pca", "descriptor"):
            raise ValueError("KICKS_CONTROL must be 'descriptor' or 'pca'")
        self.diffusion_checkpoints = parse_diffusion_checkpoints(os.environ.get("KICKS_DIFFUSION"))
        self.research_checkpoints = parse_research_checkpoints(os.environ.get('KICKS_RESEARCH'))
        for key,path in self.research_checkpoints.items():
            if not os.path.isfile(path): raise FileNotFoundError(f'research checkpoint for {key} not found: {path}')
        for name, path in self.diffusion_checkpoints.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"diffusion checkpoint for {name} not found: {path}")
        profile = get_profile(instrument)
        self.default_instrument = profile.name
        backend = backend or os.environ.get("KICKS_BACKEND", "auto")
        from ..nn.research import KINDS
        if backend not in ("auto", "vae", "diffusion", "dsp", *KINDS):
            raise ValueError("invalid KICKS_BACKEND")
        if backend == "auto":
            backend = "vae" if os.path.exists(profile.paths.checkpoint) else (
                "diffusion" if profile.name in self.diffusion_checkpoints else "dsp")
        self.default_backend = backend
        if backend in KINDS:
            return self.get_research(profile.name,backend)
        if backend == "dsp":
            return self.get_dsp(profile.name)
        if backend == "diffusion":
            return self.get_diffusion(profile.name)
        return self.load(profile.name)

    def get_dsp(self, name: str | None = None) -> ProceduralState:
        profile = get_profile(name or self.default_instrument)
        if profile.procedural is None:
            raise KeyError(profile.name)
        return ProceduralState(profile)

    def get_research(self,name,kind):
        from ..synthesis.research import ResearchEngine
        profile=get_profile(name)
        key=(profile.name,kind)
        if key not in self.research_checkpoints: raise KeyError(key)
        if key not in self._research:
            engine=ResearchEngine(self.research_checkpoints[key],profile.name,device=os.environ.get('KICKS_RESEARCH_DEVICE','cpu'))
            if engine.kind!=kind: raise ValueError('configured research backend differs from checkpoint kind')
            self._research[key]=ResearchState(profile,engine)
        return self._research[key]

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

    def get_diffusion(self, name: str) -> DiffusionState:
        """Return an instrument's served diffusion checkpoint, loading it on first request."""
        key = get_profile(name).name
        if key not in self.diffusion_checkpoints:
            raise KeyError(key)
        if key not in self._diffusion:
            profile = get_profile(key)
            path = self.diffusion_checkpoints[key]
            print(f"Loading the {profile.display_name} diffusion checkpoint {path}...")
            model, meta = load_diffusion_from_checkpoint(path, self.device, profile)
            self._diffusion[key] = DiffusionState(
                profile=profile, model=model, checkpoint=path,
                meta={k: meta.get(k) for k in ("epoch", "training")},
                data_dir=profile.paths.data_dir,
            )
        return self._diffusion[key]

    def load(self, name: str) -> InstrumentState:
        """Load a corpus, checkpoint and slider basis for one instrument."""
        profile = get_profile(name)
        data_dir = profile.paths.data_dir
        print(f"Loading {profile.display_name} from {data_dir}...")

        model, _ = load_vae_from_checkpoint(profile.paths.checkpoint, self.device)
        dataset = DrumDataset(data_dir, profile, n_frames=model.n_frames)
        print(f"Fitting the {self.control_basis} slider basis...")
        basis = fit_control_basis(
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
