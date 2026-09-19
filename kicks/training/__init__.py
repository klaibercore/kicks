"""Model training: VAE loss, the VAE loop and the diffusion loop."""

__all__ = [
    "diffusion_loss",
    "multi_resolution_loss",
    "train",
    "train_diffusion",
    "transient_loss",
    "vae_loss",
]

_MODULES = {
    "train": ".trainer",
    "diffusion_loss": ".diffusion",
    "train_diffusion": ".diffusion",
}


def __getattr__(name):
    # The dashboard only needs the standard library, even in a minimal viewer
    # environment. Import tensor/training code only for callers that use it.
    from importlib import import_module

    if name in __all__:
        module = import_module(_MODULES.get(name, ".loss"), __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
