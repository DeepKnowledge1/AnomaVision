# inference/model/backends/tensorrt_backend.py

"""
TensorRT backend — currently not implemented.
"""

from __future__ import annotations

from .base import Batch, ScoresMaps, InferenceBackend
from anodet.utils import get_logger

logger = get_logger(__name__)


class TensorRTBackend(InferenceBackend):
    """Stub for TensorRT backend."""

    def __init__(self, model_path: str, device: str = "cuda"):
        logger.warning("TensorRT backend is not implemented.")
        raise NotImplementedError("TensorRT support not implemented yet.")

    def predict(self, batch: Batch) -> ScoresMaps:
        raise NotImplementedError("TensorRT predict not implemented yet.")

    def close(self) -> None:
        pass
