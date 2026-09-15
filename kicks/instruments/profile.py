"""Instrument profiles: everything in the pipeline that depends on *what* is
being synthesised.

A profile is plain data. It names a drum, says where its corpus, checkpoints
and caches live, and parameterises every stage that has to know what it is
listening to:

* **descriptors** — the perceptual slider axes, as rectangles of the log-mel
  spectrogram (see :class:`Region` / :class:`DescriptorSpec`),
* **metrics** — which waveform measurements `kicks eval` scores, how heavily,
  and how to phrase a verdict (:class:`MetricSpec`),
* **onsets** — how far apart two hits must be to count as two hits,
* **strip** — which band carries the hit when isolating it from a loop,
* **transient loss** — the mel band / frame windows the HF fidelity term guards,
* **paths** — corpus, checkpoint, prior and cache locations.

Adding a drum type means adding a module next to ``kick.py`` and registering it;
no other module in the package hardcodes an instrument.

Band and frame indices refer to the fixed spectrogram contract in
:mod:`kicks.audio.constants` (128 mel bands over 0-22050 Hz, 256 frames of
~5.8 ms). For reference, the BigVGAN mel band -> frequency mapping is roughly::

    band   0 ~   43 Hz      band  50 ~  1.8 kHz      band  95 ~  7.7 kHz
    band   6 ~  215 Hz      band  60 ~  2.5 kHz      band 100 ~  9.0 kHz
    band  12 ~  395 Hz      band  70 ~  3.5 kHz      band 110 ~ 12.4 kHz
    band  30 ~  950 Hz      band  85 ~  5.6 kHz      band 120 ~ 17.1 kHz
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from ..audio.constants import LOG_MEL_MAX, LOG_MEL_MIN, MS_PER_FRAME, frames_to_ms

# ---------------------------------------------------------------------------
# Spectrogram regions and perceptual descriptors
# ---------------------------------------------------------------------------

Span = tuple[int, "int | None"]


@dataclass(frozen=True)
class Region:
    """A rectangle of the normalized log-mel spectrogram.

    ``bands`` and ``frames`` are half-open ``(start, stop)`` index pairs;
    ``None`` as a stop means "to the end", exactly like a Python slice.
    """

    bands: Span
    frames: Span

    def view(self, spec: np.ndarray) -> np.ndarray:
        """Slice a (n_mels, n_frames) spectrogram down to this region."""
        return spec[self.bands[0]: self.bands[1], self.frames[0]: self.frames[1]]

    def mean(self, spec: np.ndarray) -> float:
        return float(self.view(spec).mean())

    def describe(self) -> str:
        """Human-readable window, e.g. ``bands 0-12, 17-1486 ms``."""
        b0, b1 = self.bands
        f0, f1 = self.frames
        f1_ms = "end" if f1 is None else f"{frames_to_ms(f1):.0f}"
        return (f"bands {b0}-{'end' if b1 is None else b1}, "
                f"{frames_to_ms(f0):.0f}-{f1_ms} ms")


DescriptorKind = Literal[
    "mean", "log_ratio", "fraction", "inverse_ratio", "power_db_ratio", "centroid_ms",
]

_EPS = 1e-8


@dataclass(frozen=True)
class DescriptorSpec:
    """One perceptual axis, computed from one or two spectrogram regions.

    The four kinds cover every descriptor the project needs so far, which keeps
    a new instrument's descriptors declarative instead of another hand-written
    function:

    ``mean``           mean energy in ``region`` (e.g. sub weight, sizzle).
    ``log_ratio``      ``clip(log(region / reference) / scale, 0, 1)`` — a
                       transient-vs-body contrast such as punch.
    ``fraction``       ``region / (region + reference)`` — a balance between two
                       bands such as brightness, naturally bounded in [0, 1].
    ``inverse_ratio``  ``1 - clip(region / reference, 0, 1)`` — an early-to-late
                       energy ratio read as decay speed (higher = faster decay).
    """

    key: str
    label: str
    kind: DescriptorKind
    region: Region
    reference: Region | None = None
    scale: float = 1.0
    doc: str = ""

    def compute(self, spec: np.ndarray) -> float:
        """Evaluate this descriptor on a (n_mels, n_frames) spectrogram."""
        if self.kind in ("power_db_ratio", "centroid_ms"):
            # Recover linear magnitude before measuring energy: ratios of
            # normalized logarithms vary with loudness and the silence padding.
            magnitude = np.maximum(
                np.exp(spec.astype(np.float64) * (LOG_MEL_MAX - LOG_MEL_MIN) + LOG_MEL_MIN)
                - np.exp(LOG_MEL_MIN), 0.0,
            )
            energy = magnitude ** 2
            a = self.region.view(energy)
            if self.kind == "centroid_ms":
                weights = a.sum(axis=0)
                times = (np.arange(a.shape[1]) + self.region.frames[0]) * MS_PER_FRAME
                return float(weights @ times / (weights.sum() + 1e-20))
            if self.reference is None:
                raise ValueError(f"descriptor {self.key!r} needs a reference region")
            b = self.reference.view(energy)
            return float(10 * np.log10((a.mean() + 1e-20) / (b.mean() + 1e-20)))
        a = self.region.mean(spec)
        if self.kind == "mean":
            return float(a)
        if self.reference is None:
            raise ValueError(f"descriptor {self.key!r} of kind {self.kind!r} needs a reference region")
        b = self.reference.mean(spec)
        if self.kind == "log_ratio":
            ratio = a / (b + _EPS)
            return float(np.clip(np.log(max(ratio, _EPS)) / self.scale, 0.0, 1.0))
        if self.kind == "fraction":
            return float(a / (a + b + _EPS))
        if self.kind == "inverse_ratio":
            return float(1.0 - np.clip(a / (b + _EPS), 0.0, 1.0))
        raise ValueError(f"unknown descriptor kind {self.kind!r}")


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """A waveform metric plus how to score it and phrase its verdict.

    ``good`` / ``low`` / ``high`` are the sentences shown when the measured
    value sits inside, below, or above the corpus distribution. An empty
    ``low`` or ``high`` marks that direction as *not an artefact* — deviating
    that way is never penalised (a quieter noise floor than the corpus is a
    win, not a fault).
    """

    key: str
    label: str
    fmt: str            # value formatting, e.g. "{:.0f} ms"
    weight: float
    log: bool           # score in log-space (for skewed ms-scale metrics)
    good: str
    low: str
    high: str
    gate: bool = False          # hard penalty instead of a weighted contribution
    multivariate: bool = True   # include in the Mahalanobis / Frechet statistics

    def with_text(self, **kwargs: object) -> "MetricSpec":
        """Copy with some fields replaced — used by profiles to re-word verdicts."""
        return replace(self, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Waveform-domain analysis parameters
# ---------------------------------------------------------------------------

Band = tuple["float | None", "float | None"]


@dataclass(frozen=True)
class OnsetSpec:
    """How to tell one hit from two in a waveform.

    ``envelope_win`` is deliberately wide for bass-heavy instruments: a 25-30 Hz
    sub fundamental has a 33-40 ms period and ripples straight through a short
    RMS window, making every cycle look like a separate hit. Bright, short
    instruments can use a much shorter window and a smaller minimum distance.
    """

    envelope_win: int = 4096
    min_distance_ms: float = 150.0
    height_frac: float = 0.25
    prominence_frac: float = 0.20
    double_hit_ms: float = 150.0    # two onsets further apart than this = a real second hit
    loop_onsets: int = 3            # this many onsets or more = a loop / roll


@dataclass(frozen=True)
class StripSpec:
    """How to isolate a single hit inside a longer file.

    ``onset_band`` is the band the instrument actually lives in, used to build
    the detection envelope: ``(lo, hi)`` in Hz, with ``None`` meaning "open" on
    that side (so ``(None, 200.0)`` is a lowpass and ``(3000.0, None)`` is a
    highpass). ``interrupt_band`` is where a *different* instrument landing
    later in the file shows up — a hi-hat after a kick, a kick under a hat —
    and truncates the hit at that point.
    """

    onset_band: Band = (None, 200.0)
    interrupt_band: Band | None = (2000.0, None)
    min_duration_ms: float = 50.0
    max_duration_ms: float = 1200.0
    threshold: float = 0.01          # decay endpoint, as a fraction of peak
    fade_ms: float = 10.0
    transient_skip_ms: float = 30.0  # ignore the hit's own click when looking for interruptions
    interrupt_ratio: float = 5.0     # a spike this many times the baseline = another instrument
    loop_min_lag_ms: float = 100.0
    loop_acf_threshold: float = 0.3


@dataclass(frozen=True)
class EvalWindows:
    """Time windows the waveform metrics measure over, in milliseconds.

    The defaults describe a kick: a 25 ms beater click, high-frequency energy
    that should be gone by 80 ms, a body that holds pitch from 20-250 ms. A
    hi-hat compresses all of that by roughly a factor of four; a snare sits in
    between. Getting these wrong does not crash anything — it quietly measures
    the wrong part of the sound, which is worse, so each profile states them.
    """

    click_ms: float = 25.0                 # transient window for HF click and early centroid
    tail_split_ms: float = 80.0            # HF energy before vs. after this point
    tail_flatness_ms: Band = (100.0, 400.0)   # window judged for hiss vs. ringing
    body_ms: Band = (20.0, 250.0)          # where the fundamental is measured
    glide_early_ms: Band = (10.0, 60.0)    # f0 sampled here...
    glide_late_ms: Band = (100.0, 250.0)   # ...versus here
    centroid_late_ms: Band = (75.0, 200.0)  # body centroid, compared against the transient
    noise_floor_ms: float = 300.0          # trailing window checked for vocoder hiss
    decay_floor_db: float = -30.0          # decay endpoint, relative to peak


@dataclass(frozen=True)
class TransientLossSpec:
    """Windows for the high-frequency transient fidelity term in the VAE loss.

    ``band`` is where the instrument's "click" lives, ``click_frames`` how long
    the transient lasts, ``tail_start`` where HF energy should already be gone.
    A one-sided penalty past ``tail_start`` stops HF smearing into the tail
    without pushing genuinely long-tailed samples toward silence.
    """

    band: int = 50
    click_frames: int = 6
    tail_start: int = 14
    weight: float = 0.5


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathSpec:
    """Where an instrument's corpus, weights and caches live.

    Every path is derived from three roots (overridable via ``KICKS_DATA_DIR``,
    ``KICKS_MODEL_DIR``, ``KICKS_OUTPUT_DIR``) plus the instrument's corpus
    folder name and artefact subdirectory. The kick keeps the project's original
    flat layout (``models/vae_best.pth``, ``output/eval_reference.json``) by
    using an empty subdirectory, so existing checkpoints and caches keep working.
    """

    corpus_name: str            # folder under the data root, e.g. "kicks"
    subdir: str = ""            # artefact subfolder under models/ and output/
    data_root: str = "data"
    model_root: str = "models"
    output_root: str = "output"

    def _model(self, *parts: str) -> str:
        return os.path.join(self.model_root, self.subdir, *parts)

    def _output(self, *parts: str) -> str:
        return os.path.join(self.output_root, self.subdir, *parts)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.data_root, self.corpus_name)

    @property
    def backup_dir(self) -> str:
        return self.data_dir + "_backup"

    @property
    def loops_dir(self) -> str:
        return self.data_dir + "_loops"

    @property
    def quarantine_dir(self) -> str:
        return self.data_dir + "_quarantine"

    @property
    def model_dir(self) -> str:
        return self._model()

    @property
    def output_dir(self) -> str:
        return self._output()

    @property
    def checkpoint(self) -> str:
        return self._model("vae_best.pth")

    @property
    def eval_checkpoint(self) -> str:
        return self._model("vae_best_eval.pth")

    @property
    def final_checkpoint(self) -> str:
        return self._model("vae_checkpoint.pth")

    @property
    def loss_curves(self) -> str:
        return self._model("loss_curves.png")

    @property
    def latent_prior(self) -> str:
        return self._model("latent_prior.npz")

    @property
    def vocoder_dir(self) -> str:
        return self._model("vocoder")

    @property
    def eval_reference(self) -> str:
        return self._output("eval_reference.json")

    @property
    def cluster_analysis(self) -> str:
        return self._output("cluster_analysis.json")

    @property
    def clean_manifest(self) -> str:
        return self._output("clean_manifest.json")

    @property
    def samples_dir(self) -> str:
        return self._output("samples")

    def with_roots(
        self,
        data_root: str | None = None,
        model_root: str | None = None,
        output_root: str | None = None,
    ) -> "PathSpec":
        return replace(
            self,
            data_root=data_root or self.data_root,
            model_root=model_root or self.model_root,
            output_root=output_root or self.output_root,
        )


# ---------------------------------------------------------------------------
# The profile itself
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentProfile:
    """A complete description of one drum type."""

    name: str                       # registry key, e.g. "kick"
    display_name: str               # "Kick", "Hi-Hat"
    noun: str                       # "kick", "hi-hat" — used in verdict phrasing
    plural: str                     # "kicks", "hi-hats"
    paths: PathSpec
    descriptors: tuple[DescriptorSpec, ...]
    metrics: tuple[MetricSpec, ...]
    onsets: OnsetSpec = field(default_factory=OnsetSpec)
    strip: StripSpec = field(default_factory=StripSpec)
    transient_loss: TransientLossSpec = field(default_factory=TransientLossSpec)
    eval_windows: EvalWindows = field(default_factory=EvalWindows)
    hf_band: tuple[float, float] = (2000.0, 16000.0)
    fundamental_band: tuple[float, float] | None = (25.0, 150.0)
    latent_dim: int = 32
    #: Descriptor whose cross-talk the PCA slider basis cancels. Moving any
    #: other slider then leaves this one perceptually where it was — worth
    #: doing for the axis a listener notices most (decay length, usually).
    decorrelated_descriptor: str | None = "decay"
    description: str = ""
    waveform_controls: bool = False

    # -- descriptors --------------------------------------------------------

    @property
    def descriptor_keys(self) -> list[str]:
        return [d.key for d in self.descriptors]

    @property
    def descriptor_labels(self) -> list[str]:
        return [d.label for d in self.descriptors]

    @property
    def n_sliders(self) -> int:
        """Number of slider axes — one per perceptual descriptor."""
        return len(self.descriptors)

    def label_for(self, key: str) -> str:
        for d in self.descriptors:
            if d.key == key:
                return d.label
        return key.capitalize()

    # -- metrics ------------------------------------------------------------

    @property
    def scored_metrics(self) -> tuple[MetricSpec, ...]:
        """Metrics that contribute to the weighted score (everything but gates)."""
        return tuple(m for m in self.metrics if not m.gate)

    @property
    def gate_metrics(self) -> tuple[MetricSpec, ...]:
        """Metrics applied as hard penalties, e.g. multiple onsets."""
        return tuple(m for m in self.metrics if m.gate)

    @property
    def metric_keys(self) -> list[str]:
        return [m.key for m in self.metrics]

    def metric(self, key: str) -> MetricSpec:
        for m in self.metrics:
            if m.key == key:
                return m
        raise KeyError(f"{self.name} profile has no metric {key!r}")

    @property
    def pitched(self) -> bool:
        """Whether the instrument has a fundamental worth tracking."""
        return self.fundamental_band is not None

    def with_paths(self, **kwargs: str) -> "InstrumentProfile":
        return replace(self, paths=self.paths.with_roots(**kwargs))
