"""Original, seeded oscillator/modal/noise drum voices. NumPy/SciPy only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy import signal

from ..audio.constants import AUDIO_LENGTH, SAMPLE_RATE
from ..audio.waveform import bandpass, fit_length, resample
from ..instruments import InstrumentProfile, get_profile

RENDERER_VERSION = "dsp_v1"
OVERSAMPLE = 4


def phase_accumulate(frequency: np.ndarray, sample_rate: int, phase: float = 0) -> np.ndarray:
    """Phase at sample n integrates frequency before that sample."""
    return phase + 2 * np.pi * np.r_[0., np.cumsum(frequency[:-1], dtype=np.float64)] / sample_rate


def _envelope(t, decay_ms, attack_ms=.3):
    return np.minimum(t * 1000 / attack_ms, 1) * np.exp(-np.log(1000) * t * 1000 / decay_ms)


def _kick(p, t, sr, rng):
    frequency = p["tune_hz"] + p["pitch_drop_hz"] * np.exp(-t * 1000 / p["pitch_ms"])
    phase = phase_accumulate(frequency, sr)
    body = np.sin(phase) * _envelope(t, p["decay_ms"], .4)
    body += p["tone"] * .3 * np.sin(phase * 2.07) * _envelope(t, p["decay_ms"] * .45)
    noise = bandpass(rng.standard_normal(len(t)), (1800, 14000), sr)
    click = (noise * .7 + np.sin(2 * np.pi * 3200 * t) * .3) * _envelope(t, 18, .12)
    return body + p["click"] * click


def _snare(p, t, sr, rng):
    ratios = np.array([1, 2, 3, 4, 5]) * (1 - p["inharmonic"]) + np.array([1, 1.59, 2.14, 2.65, 3.72]) * p["inharmonic"]
    shell = np.zeros(len(t))
    for i, ratio in enumerate(ratios):
        shell += np.sin(2 * np.pi * p["tune_hz"] * ratio * t) * _envelope(t, p["body_ms"] / (1 + i * .45), .25) / (1 + i)
    noise = rng.standard_normal(len(t))
    wire = bandpass(noise, (850, p["bright_hz"]), sr)
    air = bandpass(noise, (5000, min(18000, p["bright_hz"] * 1.6)), sr) if p["bright_hz"] * 1.6 > 5000 else np.zeros(len(t))
    wires = wire * _envelope(t, p["wire_ms"], .5) + .25 * air * _envelope(t, p["wire_ms"] * .55)
    stick = bandpass(noise, (2000, 16000), sr) * _envelope(t, 12, .1)
    return (1 - p["wires"] * .8) * shell + p["wires"] * wires * 2 + p["crack"] * stick


def _hat(p, t, sr, rng):
    metal = np.zeros(len(t))
    for ratio in (1, 1.304, 1.466, 1.787, 1.932, 2.536):
        base = 2 * np.pi * p["metal_hz"] * ratio * t
        metal += np.sin(base + (1 + 8 * p["fm"]) * np.sin(base * 3.417 + ratio)) / 6
    metal = bandpass(metal, (p["highpass_hz"], None), sr)
    noise = bandpass(rng.standard_normal(len(t)), (p["highpass_hz"], None), sr)
    knots = rng.uniform(.65, 1., int(np.ceil(t[-1] * 150)) + 2)
    motion = np.interp(t * 150, np.arange(len(knots)), knots)
    env = _envelope(t, p["decay_ms"], .1 + p["openness"] * .7)
    metal *= env * (1 - p["openness"] * .2 * np.exp(-t * 30))
    noise *= env * motion * (1 + p["openness"] * (1 - np.exp(-t * 50)))
    return bandpass(metal * (1 - p["noise"]) * 2.5 + noise * p["noise"] * .5, (None, p["bright_hz"]), sr)


_RENDERERS = {"pitched_kick": _kick, "modal_snare": _snare, "metallic_hat": _hat}


def render(profile: InstrumentProfile | str, parameters: dict[str, float] | None = None,
           *, preset: str | None = None, seed: int = 0, velocity: float = 1,
           sample_rate: int = SAMPLE_RATE, length: int = AUDIO_LENGTH) -> np.ndarray:
    """Mono float32 hit. Physical values use the units in the profile schema."""
    profile = get_profile(profile) if isinstance(profile, str) else profile
    spec = profile.procedural
    if spec is None or spec.renderer not in _RENDERERS:
        raise ValueError(f"{profile.name} has no procedural voice")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 4294967295]")
    if not np.isfinite(velocity) or not 0 <= velocity <= 1:
        raise ValueError("velocity must be in [0, 1]")
    if sample_rate not in (44100, 48000, 88200, 96000) or not isinstance(length, int) or not 32 <= length <= sample_rate * 10:
        raise ValueError("unsupported sample rate or length (32 samples to 10 seconds)")
    p = spec.resolve(parameters, preset)
    sr = sample_rate * OVERSAMPLE
    t = np.arange(length * OVERSAMPLE, dtype=np.float64) / sr
    audio = _RENDERERS[spec.renderer](p, t, sr, np.random.default_rng(seed))
    drive = p.get("drive", 0)
    if drive:
        audio = np.tanh(audio * (1 + drive * 5))
    audio = bandpass(signal.resample_poly(audio, 1, OVERSAMPLE), (15, None), sample_rate, order=2)
    # File-boundary handling, not a tail quality gate.
    fade = min(int(sample_rate * .005), length // 4)
    audio[-fade:] *= np.linspace(1, 0, fade)
    audio[0] = 0
    peak = np.max(np.abs(audio))
    if not np.isfinite(audio).all() or peak < 1e-12:
        raise ValueError("renderer produced invalid or silent audio")
    return (audio * (.9 * velocity ** 1.6 / peak)).astype(np.float32)


def choke(audio: np.ndarray, at_ms: float, release_ms: float = 5, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """End a voice with a continuous half-cosine release."""
    if not np.isfinite(at_ms) or at_ms < 0 or not np.isfinite(release_ms) or release_ms <= 0:
        raise ValueError("choke time must be nonnegative and release positive")
    result = audio.copy()
    start = min(len(result), round(at_ms * sample_rate / 1000))
    end = min(len(result), start + max(2, round(release_ms * sample_rate / 1000)))
    if end > start:
        result[start:end] *= .5 + .5 * np.cos(np.linspace(0, np.pi, end - start))
    result[end:] = 0
    return result


def hybrid(base: np.ndarray, texture: np.ndarray, *, mix: float = .25,
           highpass_hz: float = 1200, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Onset-align an explicitly supplied texture; preserve the low-end body.

    Sample-assisted synthesis, not a learned residual decomposition.
    """
    if not np.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError("texture mix must be in [0, 1]")
    texture = np.asarray(texture, dtype=np.float64)
    if texture.ndim != 1 or not texture.size or not np.isfinite(texture).all():
        raise ValueError("texture must be a finite nonempty mono waveform")
    if mix == 0:
        return base.copy()
    active = np.flatnonzero(np.abs(texture) > np.max(np.abs(texture)) * .02)
    if not active.size:
        raise ValueError("texture is silent")
    texture = fit_length(texture[max(0, active[0] - 4):], len(base))
    texture = bandpass(texture, (highpass_hz, None), sample_rate)
    texture /= max(float(np.max(np.abs(texture))), 1e-12)
    fade = min(round(sample_rate * .005), len(texture) // 4)
    texture[:fade] *= np.linspace(0, 1, fade)
    texture[-fade:] *= np.linspace(1, 0, fade)
    result = base * (1 - mix) + texture * mix * .9
    return (result * min(1, .9 / max(float(np.max(np.abs(result))), 1e-12))).astype(np.float32)


@dataclass(frozen=True)
class Preset:
    instrument: str
    parameters: dict[str, float]
    seed: int = 0
    renderer_version: str = RENDERER_VERSION
    sample_rate: int = SAMPLE_RATE
    length: int = AUDIO_LENGTH
    texture: dict | None = None

    def __post_init__(self):
        profile = get_profile(self.instrument)
        if profile.procedural is None:
            raise ValueError("instrument has no procedural voice")
        profile.procedural.resolve(self.parameters)
        if self.renderer_version != RENDERER_VERSION:
            raise ValueError(f"unsupported renderer version: {self.renderer_version}")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("invalid preset seed")
        if self.sample_rate not in (44100, 48000, 88200, 96000) or not isinstance(self.length, int) or not 32 <= self.length <= self.sample_rate * 10:
            raise ValueError("unsupported preset sample rate or length")
        if self.texture is not None and (set(self.texture) != {"path", "sha256", "mix"} or not 0 <= self.texture["mix"] <= 1):
            raise ValueError("invalid texture identity")

    @property
    def identity(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({**asdict(self), "sha256": self.identity}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Preset":
        data = json.loads(Path(path).read_text())
        identity = data.pop("sha256", None)
        preset = cls(**data)
        if identity is not None and identity != preset.identity:
            raise ValueError("preset identity does not match its contents")
        return preset

    def audio(self) -> np.ndarray:
        audio = render(self.instrument, self.parameters, seed=self.seed, sample_rate=self.sample_rate, length=self.length)
        if self.texture is not None:
            import soundfile as sf
            path = Path(self.texture["path"])
            if sha256(path.read_bytes()).hexdigest() != self.texture["sha256"]:
                raise ValueError("texture bytes no longer match the preset")
            data, sr = sf.read(path, always_2d=True, dtype="float32")
            audio = hybrid(audio, resample(data.mean(axis=1), sr, self.sample_rate),
                           mix=self.texture["mix"], sample_rate=self.sample_rate)
        return audio


def generate_dsp(instrument="kick", count=4, out_dir="output/dsp", seed=0,
                 preset=None, parameters=None, texture_path=None, texture_mix=.25):
    import soundfile as sf

    if count < 1:
        raise ValueError("count must be positive")
    profile = get_profile(instrument)
    values = profile.procedural.resolve(parameters, preset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texture = None if not texture_path else {"path": str(Path(texture_path).resolve()),
        "sha256": sha256(Path(texture_path).read_bytes()).hexdigest(), "mix": texture_mix}
    configs = [Preset(profile.name, values, (seed + i) % 2**32, texture=texture) for i in range(count)]
    targets = [out / f"{profile.name}_{preset or 'custom'}_{config.seed}.wav" for config in configs]
    for path in targets:
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(f"refusing to overwrite {path}; choose another directory or seed")
    paths = []
    for config, path in zip(configs, targets):
        sf.write(path, config.audio(), SAMPLE_RATE, subtype="PCM_24")
        config.save(path.with_suffix(".json"))
        paths.append(str(path))
    return paths
