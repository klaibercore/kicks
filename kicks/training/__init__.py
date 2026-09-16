"""Model training: VAE loss and the training loop."""

__all__ = ["multi_resolution_loss", "train", "transient_loss", "vae_loss"]


def __getattr__(name):
    # The dashboard only needs the standard library, even in a minimal viewer
    # environment. Import tensor/training code only for callers that use it.
    from importlib import import_module

    if name in __all__:
        module = import_module(".trainer" if name == "train" else ".loss", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
