# inference/model/backends/onnx_backend.py

"""
ONNX Runtime backend implementation.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import onnxruntime as ort

from .base import Batch, ScoresMaps, InferenceBackend
from anodet.utils import get_logger

logger = get_logger(__name__)


class OnnxBackend(InferenceBackend):
    """Inference backend based on ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        *,
        intra_threads: int | None = None,
        inter_threads: int | None = None,
    ):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        if intra_threads:
            sess_options.intra_op_num_threads = intra_threads
        if inter_threads:
            sess_options.inter_op_num_threads = inter_threads

        if device.lower().startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        logger.info("Initializing OnnxRuntime with providers=%s", providers)

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        self.input_names: List[str] = [inp.name for inp in self.session.get_inputs()]
        self.output_names: List[str] = [out.name for out in self.session.get_outputs()]

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run ONNX inference on the input batch."""
        if isinstance(batch, np.ndarray):
            input_arr = batch
        else:
            input_arr = batch.detach().cpu().numpy()

        logger.debug("ONNX input shape: %s dtype: %s", input_arr.shape, input_arr.dtype)

        outputs = self.session.run(self.output_names, {self.input_names[0]: input_arr})

        scores, maps = outputs[0], outputs[1]
        logger.debug("ONNX output shapes: %s, %s", scores.shape, maps.shape)
        return scores, maps

    def close(self) -> None:
        """Release the ONNX session."""
        self.session = None
