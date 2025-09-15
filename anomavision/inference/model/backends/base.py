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

    def close(self) -> None:
        """Release backend resources."""
        ...
