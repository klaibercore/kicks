"""Corpus and latent-space analysis, and perceptual evaluation."""

from .basis import (
    DescriptorBasis,
    SliderBasis,
    analyze_latent_space,
    slider_positions_to_axis_values,
)
from .descriptors import (
    compute_descriptors,
    correlation_matrix,
    descriptor_matrix,
    descriptor_stats,
    descriptor_vector,
)
from .latents import extract_latents, fit_gmm, select_n_clusters

__all__ = [
    "DescriptorBasis", "SliderBasis", "analyze_latent_space",
    "compute_descriptors", "correlation_matrix", "descriptor_matrix",
    "descriptor_stats", "descriptor_vector", "extract_latents", "fit_gmm",
    "select_n_clusters", "slider_positions_to_axis_values",
]
