"""Physical voice schemas. Decays are T60; pitch time is an exponential tau."""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    low: float
    high: float
    default: float
    unit: str = ""
    logarithmic: bool = False
    description: str = ""

    def value(self, position: float) -> float:
        if not math.isfinite(position) or not 0 <= position <= 1:
            raise ValueError(f"{self.key} position must be finite and in [0, 1]")
        if self.logarithmic:
            return self.low * (self.high / self.low) ** position
        return self.low + position * (self.high - self.low)

    def position(self, value: float) -> float:
        if not math.isfinite(value) or not self.low <= value <= self.high:
            raise ValueError(f"{self.key} must be in [{self.low}, {self.high}] {self.unit}")
        if self.logarithmic:
            return math.log(value / self.low) / math.log(self.high / self.low)
        return (value - self.low) / (self.high - self.low)


@dataclass(frozen=True)
class VoiceSpec:
    renderer: str
    parameters: tuple[Parameter, ...]
    presets: dict[str, dict[str, float]] = field(hash=False)
    choke_group: str | None = None

    @property
    def defaults(self) -> dict[str, float]:
        return {p.key: p.default for p in self.parameters}

    def resolve(self, values: dict[str, float] | None = None, preset: str | None = None) -> dict[str, float]:
        if preset is not None and preset not in self.presets:
            raise ValueError(f"unknown preset {preset!r}; choose {', '.join(self.presets)}")
        result = {**self.defaults, **(self.presets.get(preset, {}) if preset else {}), **(values or {})}
        unknown = set(result) - set(self.defaults)
        if unknown:
            raise ValueError(f"unknown physical parameters: {', '.join(sorted(unknown))}")
        for p in self.parameters:
            p.position(result[p.key])
        return result

    def from_positions(self, positions: list[float]) -> dict[str, float]:
        if len(positions) != len(self.parameters):
            raise ValueError(f"expected {len(self.parameters)} physical controls")
        return {p.key: p.value(v) for p, v in zip(self.parameters, positions)}


KICK = VoiceSpec("pitched_kick", (
    Parameter("tune_hz", "Tune", 30, 100, 52, "Hz", True, "Settled fundamental frequency."),
    Parameter("pitch_drop_hz", "Pitch drop", 0, 350, 130, "Hz", description="Initial pitch above the fundamental."),
    Parameter("pitch_ms", "Pitch time", 3, 90, 22, "ms", True, "Exponential pitch time constant."),
    Parameter("decay_ms", "Body decay", 100, 1800, 650, "ms", True, "Amplitude reaches -60 dB at this time."),
    Parameter("click", "Beater", 0, 1, .28, description="Short filtered beater transient."),
    Parameter("tone", "Overtones", 0, 1, .18, description="Upper body modes."),
    Parameter("drive", "Saturation", 0, 1, .15, description="Oversampled saturation."),
), {
    "round": {},
    "sub": {"tune_hz": 40, "pitch_drop_hz": 65, "pitch_ms": 35, "decay_ms": 1400, "click": .08, "tone": .04},
    "punch": {"tune_hz": 58, "pitch_drop_hz": 250, "pitch_ms": 12, "decay_ms": 420, "click": .55, "drive": .35},
    "dry": {"tune_hz": 68, "pitch_drop_hz": 110, "pitch_ms": 8, "decay_ms": 200, "click": .35},
})

SNARE = VoiceSpec("modal_snare", (
    Parameter("tune_hz", "Shell tune", 120, 380, 195, "Hz", True, "Lowest shell resonance."),
    Parameter("body_ms", "Shell decay", 50, 900, 280, "ms", True, "Shell decay to -60 dB."),
    Parameter("wire_ms", "Wire decay", 40, 1200, 330, "ms", True, "Independent wire-noise decay to -60 dB."),
    Parameter("wires", "Wire mix", 0, 1, .58, description="Balance of shell and wires."),
    Parameter("bright_hz", "Wire colour", 1500, 14000, 6500, "Hz", True, "Wire texture low-pass cutoff."),
    Parameter("crack", "Stick", 0, 1, .45, description="Stick attack strength."),
    Parameter("inharmonic", "Shell character", 0, 1, .45, description="Blend electronic and inharmonic modes."),
    Parameter("drive", "Saturation", 0, 1, .12, description="Oversampled saturation."),
), {
    "classic": {},
    "tight": {"body_ms": 120, "wire_ms": 130, "wires": .65, "crack": .65},
    "deep": {"tune_hz": 145, "body_ms": 480, "wire_ms": 410, "wires": .4, "bright_hz": 4800},
    "rattle": {"wire_ms": 700, "wires": .82, "bright_hz": 10000, "inharmonic": .8},
})

HIHAT = VoiceSpec("metallic_hat", (
    Parameter("metal_hz", "Metal tune", 250, 1400, 650, "Hz", True, "Inharmonic source base frequency."),
    Parameter("decay_ms", "Decay", 25, 1800, 120, "ms", True, "Amplitude decay to -60 dB."),
    Parameter("openness", "Openness", 0, 1, .12, description="Changes attack, spectral motion and tail shape."),
    Parameter("noise", "Air mix", 0, 1, .42, description="Stochastic air versus metallic partials."),
    Parameter("bright_hz", "Brightness", 3500, 18000, 13000, "Hz", True, "High-frequency cutoff."),
    Parameter("highpass_hz", "Low cut", 800, 7000, 3200, "Hz", True, "Removes lower metallic partials."),
    Parameter("fm", "Metal complexity", 0, 1, .45, description="FM depth and spectral complexity."),
), {
    "closed": {},
    "open": {"decay_ms": 950, "openness": .85, "noise": .5, "highpass_hz": 2400},
    "pedal": {"decay_ms": 75, "openness": .03, "noise": .68, "metal_hz": 430, "bright_hz": 8500},
    "metal": {"decay_ms": 280, "noise": .12, "fm": .8, "highpass_hz": 1800},
}, choke_group="hihat")
