"""TensorRT inference backend for native AnomaVision engines."""

from __future__ import annotations

import numpy as np

from anomavision.utils import get_logger

from .base import Batch, InferenceBackend, ScoresMaps

logger = get_logger(__name__)


class TensorRTBackend(InferenceBackend):
    """Execute a serialized TensorRT engine with PyCUDA."""

    def __init__(self, model_path: str, device: str = "cuda"):
        """Load a serialized TensorRT engine.

        Args:
            model_path: Path to the serialized TensorRT engine.
            device: CUDA device identifier. TensorRT requires a CUDA device.

        Raises:
            ValueError: If ``device`` is not a CUDA device.
            ImportError: If TensorRT or PyCUDA is unavailable.
            FileNotFoundError: If ``model_path`` does not exist.
            RuntimeError: If TensorRT cannot deserialize the engine.
        """
        if not str(device).startswith("cuda"):
            raise ValueError("TensorRT inference requires a CUDA device.")
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "TensorRT inference requires NVIDIA TensorRT and PyCUDA."
            ) from exc

        self._cuda = cuda
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(model_path, "rb") as handle:
            self.engine = self._runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {model_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_name = next(
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        )
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]
        logger.info(
            "TensorRT engine loaded: input=%s outputs=%s",
            self.input_name,
            self.output_names,
        )

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run inference and return image scores and anomaly maps.

        Args:
            batch: A contiguous NCHW array or tensor accepted by the engine.

        Returns:
            A tuple ``(image_scores, score_maps)`` containing the first two
            TensorRT output tensors.
        """
        if hasattr(batch, "detach"):
            batch = batch.detach().cpu().numpy()
        input_array = np.ascontiguousarray(batch, dtype=np.float32)
        self.context.set_input_shape(self.input_name, tuple(input_array.shape))
        allocations = []
        host_outputs = []
        try:
            input_device = self._cuda.mem_alloc(input_array.nbytes)
            allocations.append(input_device)
            self.context.set_tensor_address(self.input_name, int(input_device))
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = self._trt.nptype(self.engine.get_tensor_dtype(name))
                host = np.empty(shape, dtype=dtype)
                device = self._cuda.mem_alloc(host.nbytes)
                allocations.append(device)
                host_outputs.append(host)
                self.context.set_tensor_address(name, int(device))
            self._cuda.memcpy_htod_async(input_device, input_array, self.stream)
            if not self.context.execute_async_v3(self.stream.handle):
                raise RuntimeError("TensorRT execution failed.")
            for host, device in zip(host_outputs, allocations[1:]):
                self._cuda.memcpy_dtoh_async(host, device, self.stream)
            self.stream.synchronize()
        finally:
            for allocation in allocations:
                allocation.free()

        if len(host_outputs) < 2:
            return host_outputs[0], host_outputs[0]
        return host_outputs[0], host_outputs[1]

    def close(self) -> None:
        """Release TensorRT context, engine, runtime, and CUDA resources."""
        self.context = None
        self.engine = None
        self._runtime = None
        self.stream = None

    def warmup(self, batch=None, runs: int = 2) -> None:
        """Execute repeated inference calls to stabilize engine performance.

        Args:
            batch: Representative input batch used for warm-up.
            runs: Number of warm-up calls; at least one call is performed.
        """
        if batch is None:
            raise ValueError("TensorRT warmup requires a sample batch.")
        for _ in range(max(1, runs)):
            self.predict(batch)
        logger.info("TensorRT warm-up completed: runs=%d", runs)
