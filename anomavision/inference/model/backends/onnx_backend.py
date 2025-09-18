# inference/model/backends/onnx_backend.py

from __future__ import annotations

from typing import List

import numpy as np
import onnxruntime as ort
import torch

from anomavision.utils import get_logger

from .base import Batch, InferenceBackend, ScoresMaps

logger = get_logger(__name__)


class OnnxBackend(InferenceBackend):
    """
    ONNX Runtime backend implementation with GPU I/O binding.

    This backend provides optimized inference for ONNX models using
    ONNX Runtime, with full GPU support and zero-copy I/O binding
    when inputs are PyTorch CUDA tensors.

    Features:
        - Automatic provider selection ("cuda" → CUDAExecutionProvider).
        - I/O binding for direct GPU tensor input/output (avoids
          CPU↔GPU copies and improves performance).
        - Warmup runs to stabilize CUDA kernel loading and allocation.
        - Clean fallback to CPUExecutionProvider if requested.

    Example:
        >>> backend = OnnxBackend("model.onnx", "cuda")
        >>> dummy = torch.randn(1, 3, 224, 224, device="cuda")
        >>> scores, maps = backend.predict(dummy)
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize ONNX Runtime backend for model inference.

        Args:
            model_path (str): Path to the ONNX model file (.onnx extension).
            device (str, optional): Target device ("cuda" or "cpu").
                Defaults to "cuda".

        Notes:
            - For CUDA: uses CUDAExecutionProvider only (no CPU fallback).
            - For CPU: uses CPUExecutionProvider with default threading.
        """
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.enable_cpu_mem_arena = True

        if device.lower().startswith("cuda"):
            # GPU execution: avoid hidden CPU fallback
            providers = ["CUDAExecutionProvider"]
            # Threads are irrelevant for GPU, keep minimal
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
        else:
            providers = ["CPUExecutionProvider"]

        logger.info("Initializing ONNX Runtime with providers=%s", providers)

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        logger.info(f"ONNX Runtime providers: {self.session.get_providers()}")
        logger.info(
            f"ONNX Runtime provider options: {self.session.get_provider_options()}"
        )

        self.input_names: List[str] = [inp.name for inp in self.session.get_inputs()]
        self.output_names: List[str] = [out.name for out in self.session.get_outputs()]
        self.device = device.lower()

    def predict(self, batch: Batch) -> ScoresMaps:
        """
        Run ONNX inference on input batch.

        Uses I/O binding for zero-copy GPU execution if input is a
        PyTorch CUDA tensor. Otherwise, falls back to NumPy/CPU input.

        Args:
            batch (Batch): Input batch of shape (B, C, H, W), either:
                - torch.Tensor (GPU or CPU)
                - numpy.ndarray

        Returns:
            ScoresMaps: Tuple of:
                - scores (np.ndarray): Per-image anomaly scores (B,)
                - maps (np.ndarray): Pixel-level anomaly maps (B, H, W)
        """
        if self.device == "cuda" and isinstance(batch, torch.Tensor) and batch.is_cuda:
            # --- GPU fast path using I/O binding ---
            io_binding = self.session.io_binding()

            # Bind input tensor directly from GPU memory (zero-copy)
            inp = batch.contiguous()
            io_binding.bind_input(
                name=self.input_names[0],
                device_type="cuda",
                device_id=inp.device.index or 0,
                element_type=np.float32,
                shape=tuple(inp.shape),
                buffer_ptr=inp.data_ptr(),
            )

            # Bind outputs to GPU memory
            for out in self.output_names:
                io_binding.bind_output(out, "cuda")

            # Execute inference with bound inputs/outputs
            self.session.run_with_iobinding(io_binding)

            # Copy results back to CPU numpy (needed for further processing)
            ort_outputs = io_binding.copy_outputs_to_cpu()
            scores, maps = ort_outputs[0], ort_outputs[1]

        else:
            # --- CPU / NumPy fallback path ---
            if isinstance(batch, np.ndarray):
                input_arr = batch
            else:
                input_arr = batch.detach().cpu().numpy()

            outputs = self.session.run(
                self.output_names, {self.input_names[0]: input_arr}
            )
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
