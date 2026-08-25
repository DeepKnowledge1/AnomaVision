"""PaDiM graph for AMD/Xilinx KV260."""

from __future__ import annotations

import torch
import torch.nn as nn

from .....algorithm.common.mahalanobis import MahalanobisDistance

class PadimKV260(nn.Module):
    """PaDiM graph adapted for KV260 DPU compilation."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()

        self.backbone = model.embeddings_extractor.backbone
        self.layer_indices = model.layer_indices

        self.channel_indices = model.channel_indices
        self.mahalanobis = MahalanobisDistance(
            model.mahalanobisDistance._mean_flat,
            model.mahalanobisDistance._cov_inv_flat,
        )

    def forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        features = self.backbone.layer1(x)

        batch_size, channels, height, width = features.shape

        features = features.permute(0, 2, 3, 1)
        features = features.reshape(batch_size, height * width, channels)

        features = features[:, :, self.channel_indices]

        score_map = self.mahalanobis(
            features,
            width=width,
            height=height,
        )

        image_score = score_map.flatten(1).amax(dim=1)

        return image_score, score_map
