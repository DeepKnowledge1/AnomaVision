# inference/model/backends/openvino_backend.py

"""
OpenVINO backend — currently not implemented.
"""

from __future__ import annotations


from .base import Batch, ScoresMaps, InferenceBackend

from anodet.utils import get_logger

logger = get_logger(__name__)

class OpenVinoBackend(InferenceBackend):
    """Stub for OpenVINO backend."""

    def __init__(self, model_path: str, device: str = "cpu"):
        logger.warning("OpenVINO backend is not implemented.")
        raise NotImplementedError("OpenVINO support not implemented yet.")

    def predict(self, batch: Batch) -> ScoresMaps:
        raise NotImplementedError("OpenVINO predict not implemented yet.")

    def close(self) -> None:
        pass
