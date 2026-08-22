"""Backward-compatible imports for the modular PatchCore implementation."""

from .algorithm.patchcore.patchcore import *
from .algorithm.patchcore.patchcore import PatchCore, build_patchcore_from_stats

__all__ = ["PatchCore", "build_patchcore_from_stats"]
