"""Single-DPU PaDiM graph for AMD/Xilinx KV260."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PadimKV260(nn.Module):
    """Minimal DPU-friendly PaDiM compiler graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.embeddings_extractor.backbone

        channel_indices = model.channel_indices.detach().cpu().to(torch.int64)
        cov_inv = model.mahalanobisDistance._cov_inv_flat.detach().float().cpu()

        projection = torch.zeros(50, 64, 1, 1, dtype=torch.float32)
        for i, channel in enumerate(channel_indices.tolist()):
            projection[i, channel, 0, 0] = 1.0
        self.register_buffer("channel_projection", projection)

        # Build the diagonal in Python so no indexing/select operation enters
        # the traced graph.
        inv_var = torch.empty(3136, 50, dtype=torch.float32)
        for n in range(3136):
            for d in range(50):
                inv_var[n, d] = cov_inv[n, d, d].item()
        self.register_buffer("inv_var_flat", inv_var.unsqueeze(0).contiguous())

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

        # No indexing, amax, interpolate, or custom operators in this test.
        return distance_sq, distance_sq
