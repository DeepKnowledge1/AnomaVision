"""Optional CAD-Mem adapter for a fitted AnomaVision PatchCore.

The original PatchCore implementation remains unchanged. CAD-Mem operates on
its L2-normalized patch embeddings and requires a separate normal calibration
loader. Updates are explicit; prediction never mutates the model.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from .memory import CADMem, CADMemConfig


class CADMemPatchCore(torch.nn.Module):
    """Conservative, CPU-memory adaptation layer around a fitted PatchCore.

    Not ONNX/TorchScript exportable. Use the unmodified PatchCore for export.
    """

    def __init__(self, patchcore: torch.nn.Module, config: Optional[CADMemConfig] = None):
        super().__init__()
        self.patchcore = patchcore
        self.config = config or CADMemConfig()
        self.memory: Optional[CADMem] = None

    def fit(self, train_loader, calibration_loader) -> None:
        """Fit PatchCore on training normals and calibrate on disjoint normals."""
        self.patchcore.fit(train_loader)
        self.calibrate(calibration_loader)

    @torch.no_grad()
    def calibrate(self, calibration_loader) -> None:
        if not self.patchcore.is_fitted:
            raise RuntimeError("Fit PatchCore before CAD-Mem calibration")
        batches = []
        for item in calibration_loader:
            images = item[0] if isinstance(item, (tuple, list)) else item
            features, _, _ = self.patchcore._extract(images)
            batches.append(features.reshape(-1, features.shape[-1]).cpu().numpy())
        if not batches:
            raise ValueError("Calibration loader is empty")
        anchor = self.patchcore.memory_bank.detach().float().cpu().numpy()
        self.memory = CADMem(anchor, np.concatenate(batches), self.config)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, return_map: bool = True):
        if self.memory is None:
            raise RuntimeError("CAD-Mem requires a separate normal calibration set")
        features, h, w = self.patchcore._extract(images)
        patches = features.reshape(-1, features.shape[-1]).detach().cpu().numpy()
        distances = self.memory.score(patches)
        scores_grid = torch.from_numpy(distances.copy()).to(
            device=images.device, dtype=features.dtype
        ).reshape(images.shape[0], h, w)
        image_scores = scores_grid.flatten(1).amax(1)
        if not return_map:
            return image_scores, None
        score_maps = F.interpolate(
            scores_grid.unsqueeze(1), size=images.shape[-2:],
            mode="bilinear", align_corners=False
        ).squeeze(1)
        return image_scores, score_maps

    def predict(self, images: torch.Tensor, export: bool = False):
        if export:
            raise RuntimeError("CAD-Mem online adaptation is not exportable")
        return self.forward(images)

    @torch.no_grad()
    def update(self, images: torch.Tensor) -> int:
        """Explicitly admit candidate patches from one time window.

        Never call on evaluation data when measuring a static baseline.
        """
        if self.memory is None:
            raise RuntimeError("Calibrate CAD-Mem before updating")
        features, _, _ = self.patchcore._extract(images)
        patches = features.reshape(-1, features.shape[-1]).cpu().numpy()
        return self.memory.update(patches)

    def reset_adaptation(self) -> None:
        if self.memory is None:
            raise RuntimeError("Calibrate CAD-Mem before resetting")
        self.memory.reset_adaptation()
