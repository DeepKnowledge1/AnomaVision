# inference/model/backends/torchscript_backend.py

"""
TorchScript backend implementation.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from .base import Batch, ScoresMaps, InferenceBackend
from anodet.utils import get_logger

logger = get_logger(__name__)


class TorchScriptBackend(InferenceBackend):
    """Inference backend based on TorchScript."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        *,
        num_threads: int | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if num_threads and device.lower() == "cpu":
            torch.set_num_threads(num_threads)

        logger.info("Loading TorchScript model with device=%s", self.device)

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run TorchScript inference on the input batch."""
        if isinstance(batch, np.ndarray):
            input_tensor = torch.from_numpy(batch).to(self.device)
        else:
            input_tensor = batch.to(self.device)

        logger.debug(
            "TorchScript input shape: %s dtype: %s",
            input_tensor.shape,
            input_tensor.dtype,
        )

        with torch.no_grad():
            outputs = self.model(input_tensor)

        if isinstance(outputs, (list, tuple)):
            scores, maps = outputs[0], outputs[1]
        else:
            # Handle single output case
            scores, maps = outputs, outputs

        # Convert to numpy
        scores = scores.cpu().numpy()
        maps = maps.cpu().numpy()

        logger.debug("TorchScript output shapes: %s, %s", scores.shape, maps.shape)
        return scores, maps

    def close(self) -> None:
        """Release the TorchScript model."""
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
