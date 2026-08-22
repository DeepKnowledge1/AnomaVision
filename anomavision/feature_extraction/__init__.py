"""Backward-compatible imports for shared feature extraction utilities."""

from ..algorithm.common.feature_extraction import (
    ResnetEmbeddingsExtractor,
    concatenate_layers,
)

__all__ = ["ResnetEmbeddingsExtractor", "concatenate_layers"]
