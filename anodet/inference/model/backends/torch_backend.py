# inference/model/backends/torch_backend.py

"""
PyTorch backend implementation.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from .base import Batch, ScoresMaps, InferenceBackend
from anodet.utils import get_logger

logger = get_logger(__name__)


class TorchBackend(InferenceBackend):
    """Inference backend based on PyTorch."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        *,
        use_amp: bool = True,
    ):
        if device.lower().startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            logger.info("Loaded TorchScript model from %s", model_path)
        except Exception:
            self.model = torch.load(model_path, map_location=self.device)
            logger.info("Loaded raw PyTorch model from %s", model_path)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.use_amp = use_amp and self.device.type == "cuda"

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run inference using PyTorch."""
        logger.info("Running inference via TorchBackend")

        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch)

        batch = batch.to(self.device, non_blocking=True)
        logger.debug("Torch input shape: %s", tuple(batch.shape))

        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.use_amp
            else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:

            # calling self.model.forward(batch) and self.model.predict(batch) are giving different results

            # if hasattr(self.model, "forward"):
            #     scores, maps = self.model.forward(batch)
            # else:
            scores, maps = self.model.predict(batch)

        scores_np = scores.cpu().numpy()
        maps_np = maps.cpu().numpy()
        logger.debug("Torch output shapes: %s, %s", scores_np.shape, maps_np.shape)
        return scores_np, maps_np

    def close(self) -> None:
        """Release PyTorch model."""
        self.model = None
