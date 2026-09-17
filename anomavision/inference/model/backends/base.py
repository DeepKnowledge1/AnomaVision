# inference/model/backends/base.py

"""
Abstract base protocol for inference backends.
"""

from __future__ import annotations

from typing import Protocol, Tuple, Union

import numpy as np
import torch

Batch = Union[torch.Tensor, np.ndarray]
ScoresMaps = Tuple[np.ndarray, np.ndarray]


class InferenceBackend(Protocol):
    """Protocol for all inference backend classes."""

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run inference on the input batch."""
        ...

    def extract_drift_embeddings(self, batch: Batch) -> np.ndarray:
        """Return the stable model representation used for drift monitoring.

        Backends that expose internal model features may implement this method.
        Production monitoring treats it as optional so inference backends that
        cannot expose internal representations continue to work unchanged.
        """
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...
