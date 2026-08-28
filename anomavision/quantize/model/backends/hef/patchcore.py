"""Fixed-shape PatchCore graph for Hailo end-to-end deployment."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from anomavision.algorithm.common.feature_extraction import ResnetEmbeddingsExtractor


class PatchCoreHailoGraph(nn.Module):
    """Export-friendly PatchCore graph with the complete scoring pipeline.

    The production Hailo path uses the standard 224x224 / 14x14 PatchCore
    layout. Smaller shapes are also accepted so the graph remains compatible
    with the repository's end-to-end export tests and other fixed-shape
    validation artifacts.
    """

    def __init__(
        self,
        backbone: str,
        layer_indices: List[int],
        memory_bank: torch.Tensor,
        patch_grid: int = 14,
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        if patch_grid < 1:
            raise ValueError("patch_grid must be positive")
        if input_size[0] < 1 or input_size[1] < 1:
            raise ValueError("input_size must contain positive dimensions")

        self.input_size = tuple(int(v) for v in input_size)
        self.patch_grid = int(patch_grid)
        self.layer_indices = list(layer_indices)
        self.extractor = ResnetEmbeddingsExtractor(backbone, torch.device("cpu"))

        # Each normalized memory-bank vector becomes a 1x1 convolution filter
        # for the standard 64-channel production graph. A generic matrix
        # multiplication is used for other channel counts used by tests and
        # validation artifacts.
        self.register_buffer("memory_bank", F.normalize(memory_bank.float(), dim=-1))

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, _, _ = self.extractor(image, layer_indices=self.layer_indices)
        batch, patches, channels = embeddings.shape

        side = int(patches**0.5)
        if side * side != patches:
            raise RuntimeError("PatchCore export requires a square fixed patch grid")
        if side != self.patch_grid:
            raise RuntimeError(
                f"PatchCore embedding grid ({side}x{side}) does not match "
                f"patch_grid={self.patch_grid}"
            )
        if channels != self.memory_bank.shape[1]:
            raise RuntimeError(
                "PatchCore memory-bank feature dimension does not match "
                f"the extracted embeddings: {self.memory_bank.shape[1]} != {channels}"
            )

        features = F.normalize(embeddings, dim=-1)

        if channels == 64:
            # [B, 64, grid, grid] x [memory, 64, 1, 1]
            # -> [B, memory, grid, grid]. This is the Hailo-friendly path used
            # by the production ResNet18 PatchCore artifact.
            features_2d = features.reshape(batch, side, side, channels).permute(0, 3, 1, 2)
            similarity = F.conv2d(
                features_2d,
                self.memory_bank.unsqueeze(-1).unsqueeze(-1),
            )
            best_similarity = similarity.amax(dim=1)
        else:
            # Keep the reference graph semantics for arbitrary fixed feature
            # dimensions used by tests and small validation graphs.
            similarity = torch.matmul(features, self.memory_bank.transpose(0, 1))
            best_similarity = similarity.amax(dim=-1)

        distances = torch.sqrt(torch.clamp(2.0 - 2.0 * best_similarity, min=0.0))
        score_map = distances.reshape(batch, 1, side, side)
        score_map = F.interpolate(
            score_map,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        image_scores = distances.amax(dim=1)
        return image_scores, score_map
