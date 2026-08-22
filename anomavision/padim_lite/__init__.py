"""Backward-compatible imports for the modular PaDiM-lite implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..algorithm.padim import padim_lite as _implementation

PadimLite = _implementation.PadimLite
MahalanobisDistance = _implementation.MahalanobisDistance
ResnetEmbeddingsExtractor = _implementation.ResnetEmbeddingsExtractor


def build_padim_from_stats(
    stats: Dict[str, Any], device: str = "cpu", force_precision: Optional[str] = None
) -> PadimLite:
    """Build PaDiM-lite while preserving the legacy module patch surface."""
    original_extractor = _implementation.ResnetEmbeddingsExtractor
    _implementation.ResnetEmbeddingsExtractor = ResnetEmbeddingsExtractor
    try:
        return _implementation.build_padim_from_stats(
            stats, device=device, force_precision=force_precision
        )
    finally:
        _implementation.ResnetEmbeddingsExtractor = original_extractor


def load_padim_lite(
    stats_path: str, device: str = "cpu", force_precision: Optional[str] = None
) -> PadimLite:
    """Load PaDiM-lite while preserving the legacy module patch surface."""
    original_extractor = _implementation.ResnetEmbeddingsExtractor
    _implementation.ResnetEmbeddingsExtractor = ResnetEmbeddingsExtractor
    try:
        return _implementation.load_padim_lite(
            stats_path, device=device, force_precision=force_precision
        )
    finally:
        _implementation.ResnetEmbeddingsExtractor = original_extractor


__all__ = [
    "PadimLite",
    "MahalanobisDistance",
    "ResnetEmbeddingsExtractor",
    "build_padim_from_stats",
    "load_padim_lite",
]
