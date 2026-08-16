"""Lightweight PatchCore anomaly detector.

The implementation intentionally keeps the PaDiM public contract: ``fit`` accepts a
DataLoader, ``predict`` returns ``(image_scores, score_map)``, and statistics can be
saved as a compact ``.pth`` artifact for deployment.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .feature_extraction import ResnetEmbeddingsExtractor


class PatchCore(torch.nn.Module):
    """PatchCore with a bounded, optionally randomly subsampled memory bank."""

    def __init__(
        self,
        backbone: str = "resnet18",
        device: torch.device = torch.device("cpu"),
        layer_indices: Optional[List[int]] = None,
        memory_bank: Optional[torch.Tensor] = None,
        coreset_ratio: float = 0.1,
        max_memory_patches: Optional[int] = 50000,
        n_neighbors: int = 1,
    ) -> None:
        super().__init__()
        if not 0 < coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in the interval (0, 1].")
        if n_neighbors != 1:
            raise ValueError("This lightweight implementation supports n_neighbors=1 only.")
        self.device = torch.device(device)
        self.backbone = backbone
        self.layer_indices = list(layer_indices or [0, 1])
        self.coreset_ratio = float(coreset_ratio)
        self.max_memory_patches = max_memory_patches
        self.n_neighbors = n_neighbors
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        if memory_bank is not None:
            self.register_buffer("memory_bank", memory_bank.float().to(self.device))
        else:
            self.register_buffer("memory_bank", torch.empty(0, 0, device=self.device))

    @property
    def is_fitted(self) -> bool:
        return self.memory_bank.ndim == 2 and self.memory_bank.shape[0] > 0

    def _extract(self, batch: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        embeddings, width, height = self.embeddings_extractor(
            batch.to(self.device), layer_indices=self.layer_indices
        )
        return F.normalize(embeddings.float(), dim=-1), width, height

    @torch.no_grad()
    def fit(self, dataloader: torch.utils.data.DataLoader, extractions: int = 1) -> None:
        """Build a compact normal patch memory bank from one or more passes."""
        chunks = []
        for _ in range(extractions):
            for item in dataloader:
                batch = item[0] if isinstance(item, (tuple, list)) else item
                embeddings, _, _ = self._extract(batch)
                chunks.append(embeddings.reshape(-1, embeddings.shape[-1]).cpu())
        if not chunks:
            raise ValueError("Cannot fit PatchCore with an empty dataloader.")
        bank = torch.cat(chunks, dim=0)
        keep = max(1, int(bank.shape[0] * self.coreset_ratio))
        if self.max_memory_patches is not None:
            keep = min(keep, int(self.max_memory_patches))
        if keep < bank.shape[0]:
            indices = torch.randperm(bank.shape[0])[:keep]
            bank = bank[indices]
        self.memory_bank = bank.to(self.device)

    @torch.no_grad()
    def forward(
        self, batch: torch.Tensor, return_map: bool = True, export: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.is_fitted:
            raise RuntimeError("PatchCore is not fitted. Call fit() first.")
        embeddings, width, height = self._extract(batch)
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        distances = torch.cdist(flat, self.memory_bank)
        nearest = distances.amin(dim=1).reshape(batch.shape[0], width, height)
        scores = nearest.flatten(1).amax(1)
        if not return_map:
            return scores, None
        score_map = F.interpolate(
            nearest.unsqueeze(1), size=batch.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        return scores, score_map

    def predict(self, batch: torch.Tensor, export: bool = False):
        return self.forward(batch, return_map=True, export=export)

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.embeddings_extractor.to_device(self.device)
        self.memory_bank = self.memory_bank.to(self.device)

    def save_statistics(self, path: str, half: Optional[bool] = False) -> None:
        if not self.is_fitted:
            raise RuntimeError("PatchCore is not fitted. Call fit() first.")
        bank = self.memory_bank.detach().cpu()
        if half:
            bank = bank.half()
        torch.save(
            {
                "memory_bank": bank,
                "backbone": self.backbone,
                "layer_indices": self.layer_indices,
                "coreset_ratio": self.coreset_ratio,
                "max_memory_patches": self.max_memory_patches,
                "model_type": "patchcore",
                "dtype": "fp16" if half else "fp32",
            },
            path,
        )


def build_patchcore_from_stats(
    stats: dict, device: str = "cpu", force_precision: Optional[str] = None
) -> PatchCore:
    """Build a deployment-ready PatchCore from its compact memory bank."""
    bank = stats["memory_bank"].float().cpu()
    model = PatchCore(
        backbone=str(stats["backbone"]),
        layer_indices=list(stats["layer_indices"]),
        memory_bank=bank,
        coreset_ratio=float(stats.get("coreset_ratio", 1.0)),
        max_memory_patches=stats.get("max_memory_patches"),
        device=torch.device(device),
    )
    if force_precision == "fp16" and model.device.type == "cuda":
        model = model.half()
    return model
