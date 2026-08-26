"""Fixed-shape PatchCore graph for Hailo end-to-end deployment."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from anomavision.algorithm.common.feature_extraction import ResnetEmbeddingsExtractor


class PatchCoreHailoGraph(nn.Module):
    """Export-friendly PatchCore graph with the complete scoring pipeline."""

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
        if tuple(input_size) != (224, 224) or patch_grid != 14:
            raise ValueError("Hailo PatchCore export currently requires input_size=(224, 224) and patch_grid=14")

        self.input_size = (224, 224)
        self.patch_grid = 14
        self.layer_indices = list(layer_indices)
        self.extractor = ResnetEmbeddingsExtractor(backbone, torch.device("cpu"))
        self.register_buffer("memory_bank", F.normalize(memory_bank.float(), dim=-1))

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, _, _ = self.extractor(image, layer_indices=self.layer_indices)

        # Everything below is deliberately static for the fixed 224x224 Hailo graph.
        # ResNet layer output is 56x56 with 64 channels for this PatchCore artifact.
        # Avoid image.shape / tensor.shape values in reshape because they become
        # dynamic ONNX shape tensors (Concat) and Hailo may classify them as
        # unsupported shuffle layers.
        features = embeddings.reshape(1, 56, 56, 64).permute(0, 3, 1, 2)

        # 56x56 -> 14x14 without AdaptiveAvgPool2d, which fails TorchScript ONNX
        # export when the spatial shape is hidden behind the preceding reshape.
        features = F.avg_pool2d(features, kernel_size=4, stride=4)
        features = features.permute(0, 2, 3, 1).reshape(1, 196, 64)
        features = F.normalize(features, dim=-1)

        similarity = torch.matmul(features, self.memory_bank.transpose(0, 1))
        distances = torch.sqrt(
            torch.clamp(
                2.0 - 2.0 * similarity.amax(dim=-1, keepdim=True),
                min=0.0,
            )
        )

        # Fixed 14x14 -> 224x224 score map.
        score_map = F.interpolate(
            distances.reshape(1, 1, 14, 14),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        image_scores = distances.transpose(1, 2).amax(dim=-1, keepdim=True)
        return image_scores, score_map
