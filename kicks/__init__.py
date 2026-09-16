"""A VAE drum synthesizer.

The pipeline is instrument-agnostic — kicks, snares and hi-hats all run through
the same corpus loader, VAE, vocoder and evaluator. What each drum *is* lives in
:mod:`kicks.instruments` as a profile: its descriptors, its evaluation metrics,
its onset and strip heuristics, and where its corpus and checkpoints live.

Adding a drum type means adding a profile, not editing the pipeline.
"""

__version__ = "0.2.0"

__all__ = [
    "DrumDataset", "InstrumentProfile", "VAE", "available", "get_profile",
]


def __getattr__(name):
    # Commands such as the training dashboard do not need the audio/model stack.
    from importlib import import_module

    modules = {"DrumDataset": ".data", "VAE": ".nn",
               "InstrumentProfile": ".instruments", "available": ".instruments",
               "get_profile": ".instruments"}
    if name in modules:
        value = getattr(import_module(modules[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
