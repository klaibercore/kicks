"""Analysis exports are lazy so waveform audits do not load model/plotting code."""

__all__ = ["DescriptorBasis", "SliderBasis", "analyze_latent_space", "slider_positions_to_axis_values",
           "compute_descriptors", "correlation_matrix", "descriptor_matrix", "descriptor_stats",
           "descriptor_vector", "extract_latents", "fit_gmm", "select_n_clusters"]


def __getattr__(name):
    from importlib import import_module
    if name not in __all__:
        raise AttributeError(name)
    group = ".basis" if name in {"DescriptorBasis", "SliderBasis", "analyze_latent_space", "slider_positions_to_axis_values"} else ".latents" if name in {"extract_latents", "fit_gmm", "select_n_clusters"} else ".descriptors"
    value = getattr(import_module(group, __name__), name)
    globals()[name] = value
    return value
