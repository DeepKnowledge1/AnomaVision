# inference/model/backends/torchscript_backend.py

"""
TorchScript backend implementation.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from anomavision.utils import get_logger

from .base import Batch, InferenceBackend, ScoresMaps

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
        """Initialize TorchScript backend for optimized inference.

        Loads pre-compiled TorchScript models for fast inference with minimal
        Python overhead. Automatically handles device placement and threading
        configuration.

        Args:
            model_path (str): Path to TorchScript model file (.pts, .pt extensions).
            device (str, optional): Target device. Falls back to CPU if CUDA
                unavailable. Defaults to "cuda".
            num_threads (int | None, optional): Number of CPU threads for inference.
                Only applied when using CPU device.

        Example:
            >>> backend = TorchScriptBackend("model.pts", "cuda")
            >>> backend = TorchScriptBackend("model.pts", "cpu", num_threads=8)
        """

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if num_threads and device.lower() == "cpu":
            torch.set_num_threads(num_threads)

        logger.info("Loading TorchScript model with device=%s", self.device)

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run TorchScript inference on input batch.

        Executes the compiled TorchScript model with minimal Python overhead.
        Handles both single output and tuple output models automatically.

        Args:
            batch (Batch): Input batch of images. Supports numpy arrays and torch tensors.

        Returns:
            ScoresMaps: Tuple containing:
                - scores (np.ndarray): Per-image anomaly scores
                - maps (np.ndarray): Pixel-level anomaly maps

        Note:
            For single-output models, returns the same tensor for both scores and maps.

        Example:
            >>> batch = np.random.randn(3, 3, 224, 224)
            >>> scores, maps = backend.predict(batch)
        """

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
        """Release TorchScript model and clear GPU cache.

        Removes model reference and explicitly clears CUDA cache if using GPU.
        Ensures complete cleanup of GPU resources.
        """

        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def warmup(self, batch=None, runs: int = 2) -> None:
        """Warm up TorchScript model for consistent performance.

        Performs initial forward passes to optimize CUDA kernels and memory
        allocation. Critical for achieving consistent inference times.

        Args:
            batch: Input batch for warmup. Must be provided for TorchScript warmup.
            runs (int, optional): Number of warmup iterations. Defaults to 2.

        Example:
            >>> dummy_batch = torch.randn(1, 3, 224, 224)
            >>> backend.warmup(dummy_batch, runs=4)
        """

        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device, non_blocking=True)
        else:
            batch = torch.as_tensor(batch, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for _ in range(max(1, runs)):
                _ = self.model(batch)

        logger.info(
            "TorchScriptBackend warm-up completed (runs=%d, shape=%s).",
            runs,
            tuple(batch.shape),
        )
