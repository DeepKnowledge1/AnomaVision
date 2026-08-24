"""Single-DPU PaDiM graph for AMD/Xilinx KV260."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PadimKV260(nn.Module):
    """DPU-friendly PaDiM compiler-isolation graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        extractor = model.embeddings_extractor
        self.backbone = extractor.backbone

        channel_indices = model.channel_indices.detach().cpu().to(torch.int64)
        mahal = model.mahalanobisDistance
        mean = mahal._mean_flat.detach().float().cpu()
        cov_inv = mahal._cov_inv_flat.detach().float().cpu()

        projection = torch.zeros(50, 64, 1, 1, dtype=torch.float32)
        projection[torch.arange(50), channel_indices, 0, 0] = 1.0
        self.register_buffer("channel_projection", projection)
        self.register_buffer("mean_flat", mean.unsqueeze(0).contiguous())
        self.register_buffer(
            "inv_var_flat",
            torch.diagonal(cov_inv, dim1=1, dim2=2).unsqueeze(0).contiguous(),
        )

    def forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        features = self.backbone.layer1(x)
        features = F.conv2d(features, self.channel_projection)
        features = features.flatten(2).transpose(1, 2).contiguous()
        distance_sq = (features * features).sum(dim=2)
        distance_sq = distance_sq.reshape(x.shape[0], 1, 56, 56)

        # Compiler isolation: remove all score-map resize/reduction operations.
        image_score = distance_sq[:, :, 0, 0]
        return image_score, distance_sq
