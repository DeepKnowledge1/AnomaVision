"""Normal-only CADMem research wrapper using the existing PatchCore extractor.

Intentionally NOT registered with the production CLI, model loader, or export
backends until serialization, parity, and backward-compatibility are validated.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..patchcore.patchcore import PatchCore
from .memory import BoundedNuisanceMemory


class CADMem(nn.Module):
    """Paired appearance transformations must preserve patch correspondence.

    The transform accepts and returns preprocessed [B,3,H,W] images. It must
    represent genuine benign variations, never synthetic defects.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        device: str = "cpu",
        layer_indices: Optional[list[int]] = None,
        patch_grid: Optional[int] = 14,
        rank: int = 4,
        max_prototypes: int = 1024,
        neighbors: int = 32,
        budget_quantile: float = 0.95,
        candidate_neighbors: Optional[int] = None,
        chunk_size: int = 256,
        normalize_residual: bool = False,
        scale_floor: float = 0.01,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.feature_model = PatchCore(
            backbone=backbone,
            device=self.device,
            layer_indices=layer_indices or [1, 2],
            patch_grid=patch_grid,
            search_chunk_size=chunk_size,
        )
        self.memory = BoundedNuisanceMemory(
            rank=rank,
            max_prototypes=max_prototypes,
            neighbors=neighbors,
            budget_quantile=budget_quantile,
            candidate_neighbors=candidate_neighbors,
            chunk_size=chunk_size,
            normalize_residual=normalize_residual,
            scale_floor=scale_floor,
        )

    @torch.no_grad()
    def fit(
        self,
        dataloader,
        transform: Callable[[Tensor], Tensor],
        transformations_per_batch: int = 8,
    ) -> None:
        if transformations_per_batch < 1:
            raise ValueError("transformations_per_batch must be positive")
        originals, transformed = [], []
        for item in dataloader:
            images = item[0] if isinstance(item, (tuple, list)) else item
            features, _, _ = self.feature_model._extract(images)
            variants = []
            for _ in range(transformations_per_batch):
                shifted = transform(images.clone())
                if shifted.shape != images.shape:
                    raise ValueError("Transformation must preserve image shape")
                embeddings, _, _ = self.feature_model._extract(shifted)
                variants.append(embeddings.reshape(-1, embeddings.shape[-1]).cpu())
            originals.append(features.reshape(-1, features.shape[-1]).cpu())
            transformed.append(torch.stack(variants, dim=0))
        if not originals:
            raise ValueError("Normal dataloader is empty")
        normal = torch.cat(originals, dim=0)
        counterfactual = torch.cat(transformed, dim=1)
        self.memory.fit_features(normal, counterfactual)
        self.memory.to(self.device)

    @torch.no_grad()
    def forward(self, batch: Tensor, return_map: bool = True, export: bool = False):
        if not self.memory.is_fitted:
            raise RuntimeError("CADMem is not fitted")
        embeddings, width, height = self.feature_model._extract(batch)
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        patch_scores = self.memory.score(flat).reshape(batch.shape[0], width, height)
        image_scores = patch_scores.flatten(1).amax(dim=1)
        if not return_map:
            return image_scores, None
        maps = F.interpolate(
            patch_scores.unsqueeze(1),
            size=batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return image_scores, maps

    def predict(self, batch: Tensor, export: bool = False):
        return self.forward(batch, return_map=True, export=export)

    def to_device(self, device: str) -> None:
        self.device = torch.device(device)
        self.feature_model.to_device(self.device)
        self.memory.to(self.device)
