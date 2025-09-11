# inference/model/backends/torch_backend.py
"""
PyTorch backend implementation.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from anomavision.padim_lite import (  # NEW: stats-only .pth → runtime module
    build_padim_from_stats,
)
from anomavision.utils import get_logger

from .base import Batch, InferenceBackend, ScoresMaps

logger = get_logger(__name__)


class TorchBackend(InferenceBackend):
    """Inference backend based on PyTorch."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        *,
        use_amp: bool = True,
    ):
        """Initialize PyTorch backend with automatic model type detection.

        Supports multiple PyTorch model formats including TorchScript, standard
        PyTorch models, and PaDiM statistics files. Automatically handles model
        preparation and optimization for inference.

        Args:
            model_path (str): Path to PyTorch model file. Supported formats:
                - TorchScript (.pts, .pt files)
                - Standard PyTorch models (.pth with nn.Module)
                - PaDiM statistics (.pth with mean/cov_inv dict)
            device (str, optional): Target device. Automatically falls back to CPU
                if CUDA is requested but unavailable. Defaults to "cpu".
            use_amp (bool, optional): Enable automatic mixed precision (FP16) for
                faster inference on supported GPUs. Defaults to True.

        Example:
            >>> backend = TorchBackend("model.pts", "cuda", use_amp=True)
            >>> backend = TorchBackend("stats.pth", "cpu")  # PaDiM statistics
        """

        # --- Device selection: CPU-first; use CUDA only if requested & available
        req = str(device or "cpu").lower()
        if req.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if req.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available; falling back to CPU.")
            self.device = torch.device("cpu")

        loaded_obj = None

        # 1) Try TorchScript first
        try:
            logger.info("Trying torch.jit.load: %s", model_path)
            loaded_obj = torch.jit.load(model_path, map_location=self.device)
            logger.info("Loaded TorchScript model from %s", model_path)
        except Exception as e_jit:
            logger.info(
                "torch.jit.load failed (%s). Falling back to torch.load.", str(e_jit)
            )

        # 2) Fallback: raw torch.load (may be nn.Module or a stats dict)
        if loaded_obj is None:
            logger.info("Trying torch.load: %s", model_path)
            loaded_obj = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            logger.info("Loaded object type: %s", type(loaded_obj).__name__)

        # 3) If it's a stats-only dict, build a PadimLite runtime module on CPU
        if isinstance(loaded_obj, dict) and {
            "mean",
            "cov_inv",
            "channel_indices",
            "layer_indices",
            "backbone",
        }.issubset(loaded_obj.keys()):
            logger.info(
                "Detected statistics-only artifact (.pth). Building PadimLite on CPU."
            )
            model = build_padim_from_stats(loaded_obj, device="cpu")
        else:
            model = loaded_obj

        # 4) Unwrap DataParallel if present
        if hasattr(model, "module"):
            logger.info("Unwrapping DataParallel container.")
            model = model.module

        # 5) Finalize
        if hasattr(model, "eval"):
            model.eval()
        # Disable grads if parameters exist
        if hasattr(model, "parameters"):
            for p in model.parameters():
                p.requires_grad_(False)

        self.model = model
        # AMP only makes sense on CUDA
        self.use_amp = bool(use_amp and self.device.type == "cuda")

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run PyTorch inference with automatic mixed precision support.

        Executes inference using PyTorch with optional AMP acceleration. Handles
        device placement and tensor conversion automatically.

        Args:
            batch (Batch): Input batch of images. Converted to torch.Tensor if needed.

        Returns:
            ScoresMaps: Tuple containing:
                - scores (np.ndarray): Per-image anomaly scores
                - maps (np.ndarray): Pixel-level anomaly maps

        Note:
            Uses model.predict() method for consistent interface across all
            model types and export formats.

        Example:
            >>> batch = torch.randn(2, 3, 224, 224)
            >>> scores, maps = backend.predict(batch)
        """

        logger.debug("Running inference via TorchBackend")

        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch, dtype=torch.float32)

        batch = batch.to(self.device, non_blocking=True)
        logger.debug("Torch input shape: %s", tuple(batch.shape))

        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.use_amp and self.device.type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            # Always use .predict to match your runtime/export path
            scores, maps = self.model.predict(batch)

        scores_np = scores.detach().cpu().numpy()
        maps_np = maps.detach().cpu().numpy()
        logger.debug("Torch output shapes: %s, %s", scores_np.shape, maps_np.shape)
        return scores_np, maps_np

    def close(self) -> None:
        """Release PyTorch model and clear GPU memory.

        Removes model reference and triggers garbage collection. Essential for
        preventing memory leaks in applications that load multiple models.
        """

        self.model = None

    def warmup(self, batch, runs: int = 2) -> None:
        """Warm up PyTorch backend with AMP support.

        Performs warmup inference runs using the same settings as production
        inference, including automatic mixed precision if enabled.

        Args:
            batch: Input batch for warmup. Converted to appropriate tensor format.
            runs (int, optional): Number of warmup iterations. Defaults to 2.

        Example:
            >>> warmup_batch = torch.randn(1, 3, 224, 224)
            >>> backend.warmup(warmup_batch, runs=3)
        """

        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        else:
            batch = batch.to(self.device, non_blocking=True)

        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.use_amp and self.device.type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            for _ in range(max(1, runs)):
                _ = self.model.predict(batch)

        logger.info(
            "TorchBackend warm-up completed (runs=%d, shape=%s).",
            runs,
            tuple(batch.shape),
        )
