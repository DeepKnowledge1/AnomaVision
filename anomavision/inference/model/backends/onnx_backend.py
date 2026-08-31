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
    """ONNX Runtime inference backend with production CPU/GPU tuning."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device.lower()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        try:
            sess_options.add_free_dimension_override_by_name("batch_size", 1)
        except Exception:
            pass

        if self.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider"]
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
        else:
            providers = ["CPUExecutionProvider"]

        logger.info("Initializing ONNX Runtime with providers=%s", providers)
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        logger.info("ONNX Runtime providers: %s", self.session.get_providers())
        logger.info(
            "ONNX Runtime provider options: %s", self.session.get_provider_options()
        )

        self.input_names: List[str] = [x.name for x in self.session.get_inputs()]
        self.output_names: List[str] = [x.name for x in self.session.get_outputs()]

    def predict(self, batch: Batch) -> ScoresMaps:
        if self.device.startswith("cuda") and isinstance(batch, torch.Tensor) and batch.is_cuda:
            io_binding = self.session.io_binding()
            inp = batch.contiguous()
            dtype_map = {
                torch.float32: np.float32,
                torch.float16: np.float16,
                torch.float64: np.float64,
            }
            io_binding.bind_input(
                name=self.input_names[0],
                device_type="cuda",
                device_id=inp.device.index or 0,
                element_type=dtype_map.get(inp.dtype, np.float32),
                shape=tuple(inp.shape),
                buffer_ptr=inp.data_ptr(),
            )
            for name in self.output_names:
                io_binding.bind_output(name, "cuda")
            self.session.run_with_iobinding(io_binding)
            outputs = io_binding.copy_outputs_to_cpu()
        else:
            if isinstance(batch, np.ndarray):
                input_arr = np.ascontiguousarray(batch, dtype=np.float32)
            else:
                input_arr = np.ascontiguousarray(
                    batch.detach().cpu().numpy(), dtype=np.float32
                )

            # Keep the CPU execution on ORT's I/O-binding path as well. This
            # avoids rebuilding the feed/output dictionaries on every call and
            # lets ORT reuse its CPU memory planning for the fixed batch=1 path.
            io_binding = self.session.io_binding()
            io_binding.bind_cpu_input(self.input_names[0], input_arr)
            for name in self.output_names:
                io_binding.bind_output(name, "cpu")
            self.session.run_with_iobinding(io_binding)
            outputs = io_binding.copy_outputs_to_cpu()

        if len(outputs) < 2:
            raise RuntimeError(
                f"Expected at least 2 outputs (scores, maps), got {len(outputs)}"
            )
        scores, maps = outputs[0], outputs[1]
        return scores, maps

    def close(self) -> None:
        self.session = None

    def warmup(self, batch, runs: int = 2) -> None:
        for _ in range(max(1, runs)):
            self.predict(batch)
        logger.info("OnnxBackend warm-up completed (runs=%d).", runs)
