"""Synthesis engines, imported only when requested."""

__all__ = [
    "draw_labels",
    "fit_latent_prior",
    "generate",
    "generate_diffusion",
    "sample_hits",
]


def __getattr__(name):
    from importlib import import_module

    if name in __all__:
        module = ".diffusion" if name in {"draw_labels", "generate_diffusion", "sample_hits"} else ".generator"
        value = getattr(import_module(module, __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
