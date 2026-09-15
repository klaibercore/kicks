"""A VAE drum synthesizer.

The pipeline is instrument-agnostic — kicks, snares and hi-hats all run through
the same corpus loader, VAE, vocoder and evaluator. What each drum *is* lives in
:mod:`kicks.instruments` as a profile: its descriptors, its evaluation metrics,
its onset and strip heuristics, and where its corpus and checkpoints live.

Adding a drum type means adding a profile, not editing the pipeline.
"""

from .data import DrumDataset
from .instruments import InstrumentProfile, available, get_profile
from .nn import VAE

__version__ = "0.2.0"

__all__ = [
    "DrumDataset", "InstrumentProfile", "VAE", "available", "get_profile",
]
