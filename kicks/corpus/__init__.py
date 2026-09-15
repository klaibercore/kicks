"""Corpus preparation: isolating hits and removing what is not a clean one-shot."""

from .clean import run_clean
from .strip import detect_hit_region, run_strip, strip_file

__all__ = ["detect_hit_region", "run_clean", "run_strip", "strip_file"]
