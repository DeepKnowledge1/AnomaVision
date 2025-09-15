# inference/model/wrapper.py

"""
Main entry point for model inference.

Selects an appropriate backend based on model extension and delegates
all inference calls to that backend.
"""

from __future__ import annotations

import os

from anomavision.utils import get_logger

from ..model.backends.base import InferenceBackend, ScoresMaps
from ..modelType import ModelType

logger = get_logger(__name__)


def make_backend(model_path: str, device: str) -> InferenceBackend:
    """Factory function to create appropriate inference backend based on model type.

    Automatically detects the model format from file extension and instantiates
    the corresponding backend implementation. Supports multiple inference frameworks
    for optimal performance across different deployment scenarios.

    Args:
        model_path (str): Path to the model file. The file extension determines
            which backend will be selected:
            - .onnx → ONNX Runtime backend
            - .torchscript/.pts → TorchScript backend
            - .pth → PyTorch backend
            - .engine → TensorRT backend (not implemented)
            - .xml/.bin → OpenVINO backend

        device (str): Target device for inference. Common values:
            - "cpu" → CPU execution
            - "cuda" → GPU execution (if available)
            - Device-specific identifiers for specialized backends

    Returns:
        InferenceBackend: Initialized backend instance ready for inference.

    Raises:
        NotImplementedError: If the detected model type is not supported.
        FileNotFoundError: If model_path does not exist.
        ImportError: If required backend dependencies are not installed.

    Example:
        >>> backend = make_backend("model.onnx", "cuda")
        >>> backend = make_backend("model.pth", "cpu")
        >>> scores, maps = backend.predict(input_batch)
    """

    logger.info(f"Creating backend for model: {model_path}")
    model_type = ModelType.from_extension(model_path)
    logger.info(f"Detected model type: {model_type}")

    if model_type == ModelType.ONNX:
        logger.info("Loading ONNX backend...")
        from .backends.onnx_backend import OnnxBackend

        logger.debug("Selected ONNX backend for %s", model_path)
        backend = OnnxBackend(model_path, device)
        logger.info("ONNX backend created successfully")
        return backend

    if model_type == ModelType.TORCHSCRIPT:
        logger.info("Loading TorchScript backend...")
        from .backends.torchscript_backend import TorchScriptBackend

        logger.debug("Selected TorchScript backend for %s", model_path)
        backend = TorchScriptBackend(model_path, device)
        logger.info("TorchScript backend created successfully")
        return backend

    if model_type == ModelType.PYTORCH:
        from .backends.torch_backend import TorchBackend

        logger.debug("Selected PyTorch backend for %s", model_path)
        return TorchBackend(model_path, device)

    if model_type == ModelType.TENSORRT:
        from .backends.tensorrt_backend import TensorRTBackend

        logger.debug("Selected TensorRT backend for %s", model_path)
        return TensorRTBackend(model_path, device)

    if model_type == ModelType.OPENVINO:
        logger.info("Loading OpenVINO backend...")
        from .backends.openvino_backend import OpenVinoBackend

        logger.debug("Selected OpenVINO backend for %s", model_path)
        backend = OpenVinoBackend(model_path, device)
        logger.info("OpenVINO backend created successfully")
        return backend

    raise NotImplementedError(f"ModelType {model_type} is not supported.")


class ModelWrapper:
    """
    Thin wrapper around inference backends.  Clients use this class to
    abstract away the backend-specific initialization and prediction API.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        logger.info(f"Initializing ModelWrapper with {model_path} on {device}")
        self.device = device
        self.backend: InferenceBackend = make_backend(model_path, device)
        logger.info("ModelWrapper initialization completed successfully")

    def predict(self, batch) -> ScoresMaps:
        """
        Run inference on the given batch using the selected backend.
        Returns (scores, maps) as numpy arrays.
        """
        logger.debug(f"Running prediction via {self.backend.__class__.__name__}")
        result = self.backend.predict(batch)
        logger.debug("Prediction completed successfully")
        return result

    def close(self) -> None:
        """Release resources associated with the backend."""
        logger.info("Closing ModelWrapper and releasing resources")
        self.backend.close()

    def warmup(self, batch=None, runs: int = 2) -> None:
        if hasattr(self.backend, "warmup"):
            return self.backend.warmup(batch=batch, runs=runs)
        else:
            logger.info(
                f"{self.backend.__class__.__name__} does not support warm-up. Skipping."
            )
