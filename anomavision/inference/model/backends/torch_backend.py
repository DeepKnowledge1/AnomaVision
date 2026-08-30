# inference/model/backends/torch_backend.py
"""PyTorch inference backend."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from anomavision.algorithm.efficientad import build_efficientad_from_stats
from anomavision.algorithm.padim.padim_lite import build_padim_from_stats
from anomavision.algorithm.patchcore import build_patchcore_from_stats
from anomavision.utils import get_logger

from .base import Batch, InferenceBackend, ScoresMaps

logger = get_logger(__name__)


class TorchBackend(InferenceBackend):
    """Inference backend based on PyTorch."""

    def __init__(self, model_path: str, device: str = "cpu", *, use_amp: bool = True):
        req = str(device or "cpu").lower()
        if req.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if req.startswith("cuda"):
                logger.warning("CUDA requested but not available; falling back to CPU.")
            self.device = torch.device("cpu")

        loaded_obj = None
        try:
            loaded_obj = torch.jit.load(model_path, map_location=self.device)
            logger.info("Loaded TorchScript model from %s", model_path)
        except Exception as exc:
            logger.info("torch.jit.load failed (%s); falling back to torch.load.", exc)

        if loaded_obj is None:
            loaded_obj = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(loaded_obj, dict) and {"mean", "cov_inv", "channel_indices", "layer_indices", "backbone"}.issubset(loaded_obj):
            model = build_padim_from_stats(loaded_obj, device=device)
        elif isinstance(loaded_obj, dict) and {"memory_bank", "layer_indices", "backbone"}.issubset(loaded_obj):
            model = build_patchcore_from_stats(loaded_obj, device=device)
        elif isinstance(loaded_obj, dict) and {"student", "backbone", "threshold"}.issubset(loaded_obj):
            logger.info("Detected EfficientAD statistics artifact.")
            model = build_efficientad_from_stats(loaded_obj, device=device)
        else:
            model = loaded_obj

        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "parameters"):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        self.model = model
        self.use_amp = bool(use_amp and self.device.type == "cuda")

    def _autocast(self):
        return (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.use_amp
            else nullcontext()
        )

    def predict(self, batch: Batch) -> ScoresMaps:
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch, dtype=torch.float32)
        batch = batch.to(self.device, non_blocking=True)
        with torch.inference_mode(), self._autocast():
            scores, maps = self.model.predict(batch)
        return scores.detach().cpu().numpy(), maps.detach().cpu().numpy()

    def close(self) -> None:
        self.model = None

    def warmup(self, batch, runs: int = 2) -> None:
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        else:
            batch = batch.to(self.device, non_blocking=True)
        with torch.inference_mode(), self._autocast():
            for _ in range(max(1, runs)):
                self.model.predict(batch)
