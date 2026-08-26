"""Fixed-shape PatchCore graph for Hailo end-to-end deployment."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from anomavision.algorithm.common.feature_extraction import ResnetEmbeddingsExtractor


class PatchCoreHailoGraph(nn.Module):
    """Export-friendly PatchCore graph with the complete scoring pipeline.

    The graph keeps the existing PatchCore math unchanged: normalized patch
    embeddings, nearest memory-bank cosine distance, square-root distance,
    spatial upsampling, and image-level maximum.
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

        self.input_size = tuple(int(v) for v in input_size)
        self.patch_grid = int(patch_grid)
        self.layer_indices = list(layer_indices)
        self.extractor = ResnetEmbeddingsExtractor(backbone, torch.device("cpu"))
        self.register_buffer("memory_bank", F.normalize(memory_bank.float(), dim=-1))

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, width, height = self.extractor(
            image, layer_indices=self.layer_indices
        )

        # Match PatchCore._extract: adaptive pooling is applied before L2
        # normalization when the native feature grid is larger than patch_grid.
        features = embeddings.reshape(
            image.shape[0], width, height, embeddings.shape[-1]
        ).permute(0, 3, 1, 2)
        if width > self.patch_grid or height > self.patch_grid:
            features = F.adaptive_avg_pool2d(
                features, (self.patch_grid, self.patch_grid)
            )
        height, width = features.shape[-2:]
        features = features.permute(0, 2, 3, 1).reshape(
            image.shape[0], height * width, features.shape[1]
        )
        features = F.normalize(features, dim=-1)

        similarity = torch.matmul(features, self.memory_bank.transpose(0, 1))
        # Keep the reduced feature dimension. Hailo supports ReduceMax on the
        # features axis only when keepdim=True. The trailing singleton also
        # keeps the tensor directly reshapeable into the spatial score map.
        distances = torch.sqrt(
            torch.clamp(
                2.0 - 2.0 * similarity.amax(dim=-1, keepdim=True),
                min=0.0,
            )
        )

        score_map = F.interpolate(
            distances.reshape(image.shape[0], 1, height, width),
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Put patches on the spatial axis and reduce over the final/features
        # axis so Hailo can lower this ReduceMax as well. The singleton
        # dimension is retained intentionally for the supported reduction.
        image_scores = distances.transpose(1, 2).amax(dim=-1, keepdim=True)
        return image_scores, score_map
