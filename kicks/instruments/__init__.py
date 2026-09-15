"""Instrument registry.

``get_profile("snare")`` is the only way the rest of the package learns what it
is synthesising. To add a drum type, write a module here exposing a ``PROFILE``
and add it to ``_PROFILES``.
"""

from __future__ import annotations

import os

from .hihat import PROFILE as HIHAT
from .kick import PROFILE as KICK
from .metrics import standard_metrics
from .profile import (
    Band,
    DescriptorKind,
    DescriptorSpec,
    EvalWindows,
    InstrumentProfile,
    MetricSpec,
    OnsetSpec,
    PathSpec,
    Region,
    StripSpec,
    TransientLossSpec,
)
from .snare import PROFILE as SNARE

_PROFILES: dict[str, InstrumentProfile] = {
    KICK.name: KICK,
    SNARE.name: SNARE,
    HIHAT.name: HIHAT,
}

#: Instrument used when nothing else is specified.
DEFAULT_INSTRUMENT = KICK.name

#: Accepted spellings that are not the canonical registry key.
_ALIASES = {
    "kicks": "kick", "bd": "kick", "bassdrum": "kick", "bass_drum": "kick",
    "snares": "snare", "sd": "snare", "snaredrum": "snare",
    "hihats": "hihat", "hh": "hihat", "hat": "hihat", "hats": "hihat",
    "hi-hat": "hihat", "hi_hat": "hihat",
}


def available() -> list[str]:
    """Registered instrument names, in registration order."""
    return list(_PROFILES)


def normalize(name: str) -> str:
    """Resolve an alias or casing variant to a canonical instrument name."""
    key = name.strip().lower().replace(" ", "")
    return _ALIASES.get(key, key)


def get_profile(name: str | None = None) -> InstrumentProfile:
    """Look up an instrument profile.

    Resolution order: the explicit ``name`` argument, then ``KICKS_INSTRUMENT``
    from the environment, then :data:`DEFAULT_INSTRUMENT`. Path roots are
    overlaid from ``KICKS_DATA_DIR`` / ``KICKS_MODEL_DIR`` / ``KICKS_OUTPUT_DIR``
    when those are set.
    """
    requested = name or os.environ.get("KICKS_INSTRUMENT") or DEFAULT_INSTRUMENT
    key = normalize(requested)
    if key not in _PROFILES:
        raise KeyError(
            f"Unknown instrument {requested!r}. Available: {', '.join(available())}"
        )
    profile = _PROFILES[key]

    data_root = os.environ.get("KICKS_DATA_DIR")
    model_root = os.environ.get("KICKS_MODEL_DIR")
    output_root = os.environ.get("KICKS_OUTPUT_DIR")
    if data_root or model_root or output_root:
        profile = profile.with_paths(
            data_root=data_root, model_root=model_root, output_root=output_root,
        )
    return profile


def register(profile: InstrumentProfile) -> None:
    """Add (or replace) a profile at runtime."""
    _PROFILES[profile.name] = profile


__all__ = [
    "Band", "DEFAULT_INSTRUMENT", "DescriptorKind", "DescriptorSpec",
    "EvalWindows",
    "InstrumentProfile", "MetricSpec", "OnsetSpec", "PathSpec", "Region",
    "StripSpec", "TransientLossSpec", "available", "get_profile", "normalize",
    "register", "standard_metrics",
]
