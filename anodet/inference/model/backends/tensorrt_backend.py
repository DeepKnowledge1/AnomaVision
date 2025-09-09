# inference/model/backends/tensorrt_backend.py

"""
TensorRT backend — currently not implemented.
"""

from __future__ import annotations

from anodet.utils import get_logger

from .base import Batch, InferenceBackend, ScoresMaps

logger = get_logger(__name__)


class TensorRTBackend(InferenceBackend):
    """Stub for TensorRT backend."""

    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize TensorRT backend (currently not implemented).

        Placeholder for future TensorRT backend implementation. TensorRT provides
        highly optimized inference for NVIDIA GPUs but requires additional
        implementation work.

        Args:
            model_path (str): Path to TensorRT engine file (.engine extension).
            device (str, optional): Target device, should be "cuda". Defaults to "cuda".

        Raises:
            NotImplementedError: Always raised as TensorRT is not yet implemented.

        Note:
            TensorRT backend is planned for future implementation to provide
            maximum performance on NVIDIA GPUs.
        """

        logger.warning("TensorRT backend is not implemented.")
        raise NotImplementedError("TensorRT support not implemented yet.")

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run TensorRT inference (not implemented).

        Args:
            batch (Batch): Input batch for inference.

        Raises:
            NotImplementedError: Always raised as TensorRT is not yet implemented.
        """

        raise NotImplementedError("TensorRT predict not implemented yet.")

    def close(self) -> None:
        """Release TensorRT resources (not implemented).

        Placeholder for future resource cleanup implementation.
        """
        pass

    def warmup(self, batch=None, runs: int = 2) -> None:
        """Warm up TensorRT backend (not implemented).

        Args:
            batch: Input batch for warmup.
            runs (int, optional): Number of warmup iterations.

        Raises:
            NotImplementedError: Always raised as TensorRT is not yet implemented.
        """

        logger.warning("TensorRT backend is not implemented; warm-up skipped.")
        raise NotImplementedError("TensorRT warm-up not implemented yet.")
