# inference/model/backends/onnx_backend.py

"""
ONNX Runtime backend implementation.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import onnxruntime as ort

from anodet.utils import get_logger

from .base import Batch, InferenceBackend, ScoresMaps

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
        """Initialize ONNX Runtime backend for model inference.

        Creates an optimized ONNX Runtime session with automatic provider selection
        based on device availability. Configures graph optimizations and threading
        for optimal performance.

        Args:
            model_path (str): Path to the ONNX model file (.onnx extension).
            device (str, optional): Target device for inference. "cuda" enables
                GPU acceleration if available, otherwise falls back to CPU.
                Defaults to "cuda".
            intra_threads (int | None, optional): Number of threads for intra-op
                parallelism. If None, uses ONNX Runtime defaults.
            inter_threads (int | None, optional): Number of threads for inter-op
                parallelism. If None, uses ONNX Runtime defaults.

        Example:
            >>> backend = OnnxBackend("model.onnx", "cuda")
            >>> backend = OnnxBackend("model.onnx", "cpu", intra_threads=4)
        """

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
        """Run ONNX inference on input batch.

        Executes the ONNX model on the provided input batch and returns anomaly
        detection results. Handles automatic conversion from PyTorch tensors to
        numpy arrays as required by ONNX Runtime.

        Args:
            batch (Batch): Input batch of images. Can be:
                - torch.Tensor of shape (B, C, H, W)
                - numpy.ndarray of shape (B, C, H, W)

        Returns:
            ScoresMaps: Tuple containing:
                - scores (np.ndarray): Per-image anomaly scores of shape (B,)
                - maps (np.ndarray): Pixel-level anomaly maps of shape (B, H, W)

        Example:
            >>> import numpy as np
            >>> batch = np.random.randn(2, 3, 224, 224)
            >>> scores, maps = backend.predict(batch)
        """

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
        """Release ONNX Runtime session resources.

        Properly destroys the ONNX Runtime session to free memory and GPU resources.
        Should be called when the backend is no longer needed.
        """

        self.session = None

    def warmup(self, batch, runs: int = 2) -> None:
        """Warm up ONNX Runtime for optimal inference performance.

        Performs initial inference runs to initialize CUDA kernels and optimize
        memory allocation patterns. Reduces first-inference latency in production.

        Args:
            batch: Input batch for warmup inference. Same format as predict().
            runs (int, optional): Number of warmup iterations. Defaults to 2.

        Example:
            >>> dummy_batch = np.random.randn(1, 3, 224, 224)
            >>> backend.warmup(dummy_batch, runs=3)
        """
        if isinstance(batch, np.ndarray):
            input_arr = batch
        else:
            input_arr = batch.detach().cpu().numpy()

        feeds = {self.input_names[0]: input_arr}
        for _ in range(max(1, runs)):
            _ = self.session.run(self.output_names, feeds)

        logger.info(
            "OnnxBackend warm-up completed (runs=%d, shape=%s).",
            runs,
            tuple(input_arr.shape),
        )
