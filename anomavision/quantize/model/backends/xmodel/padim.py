"""Single-DPU PaDiM graph for AMD/Xilinx KV260.

This backend is intentionally independent from PatchCore.

The original PaDiM model uses a full 50x50 inverse covariance matrix for each
of the 3136 spatial locations. That exact location-dependent quadratic form
cannot be represented as a normal shared DPU convolution.

For the KV260 single-DPU graph we therefore use the diagonal of each inverse
covariance matrix, preserving the location-dependent variance weighting while
expressing the graph with DPU-friendly operations.

The original PaDiM model and MahalanobisDistance implementation are not
modified by this backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PadimKV260(nn.Module):
    """DPU-friendly PaDiM graph for KV260."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()

        extractor = model.embeddings_extractor
        self.backbone = extractor.backbone

        channel_indices = model.channel_indices.detach().cpu().to(torch.int64)

        if channel_indices.numel() != 50:
            raise ValueError(
                "KV260 PaDiM backend expects exactly 50 selected channels, "
                f"got {channel_indices.numel()}"
            )

        if model.layer_indices != [0]:
            raise ValueError(
                "KV260 PaDiM backend expects layer_indices=[0], "
                f"got {model.layer_indices}"
            )

        mahal = model.mahalanobisDistance
        mean = mahal._mean_flat.detach().float().cpu()
        cov_inv = mahal._cov_inv_flat.detach().float().cpu()

        if mean.shape != (3136, 50):
            raise ValueError(
                "KV260 PaDiM backend expects mean shape [3136, 50], "
                f"got {tuple(mean.shape)}"
            )

        if cov_inv.shape != (3136, 50, 50):
            raise ValueError(
                "KV260 PaDiM backend expects cov_inv shape [3136, 50, 50], "
                f"got {tuple(cov_inv.shape)}"
            )

        # Fixed 1x1 projection: [B, 64, 56, 56] -> [B, 50, 56, 56].
        projection = torch.zeros(50, 64, 1, 1, dtype=torch.float32)
        projection[torch.arange(50), channel_indices, 0, 0] = 1.0
        self.register_buffer("channel_projection", projection)

        # PaDiM statistics are stored as [N, D]. Keep this exact ordering and
        # use [B, N, D] tensors in the forward graph. This is important for
        # Vitis AI: NNDCT's deploy optimizer was converting the 4-D feature
        # tensor to NHWC while leaving a 4-D constant in NCHW, causing:
        #   (1,56,56,50) vs (1,50,56,56)
        # broadcast failures during export.
        mean = mean.unsqueeze(0).contiguous()  # [1, 3136, 50]
        inv_var = torch.diagonal(cov_inv, dim1=1, dim2=2).contiguous()
        inv_var = inv_var.unsqueeze(0).contiguous()  # [1, 3136, 50]

        self.register_buffer("mean_flat", mean)
        self.register_buffer("inv_var_flat", inv_var)

    def forward(self, x: torch.Tensor):
        # ResNet18 layer1 -> [B, 64, 56, 56].
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        features = self.backbone.layer1(x)

        # Select PaDiM's 50 channels.
        features = F.conv2d(features, self.channel_projection)

        # Convert to [B, N, D] without a 4-D NHWC constant/broadcast.
        features = features.flatten(2).transpose(1, 2).contiguous()

        # Diagonal Mahalanobis distance:
        #   d² = sum((x - mean)^2 * diag(cov_inv))
        delta = features - self.mean_flat
        distance_sq = delta * delta
        distance_sq = distance_sq * self.inv_var_flat
        distance_sq = distance_sq.sum(dim=2)
        distance_sq = torch.clamp(distance_sq, min=0.0)

        # [B, 3136] -> [B, 1, 56, 56].
        distance_sq = distance_sq.reshape(
            x.shape[0], 1, 56, 56
        )

        # Keep the score-map path simple and DPU-friendly.
        score_map = F.interpolate(
            distance_sq,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Image-level anomaly score.
        image_score = F.adaptive_max_pool2d(
            distance_sq,
            output_size=(1, 1),
        ).flatten(1)

        return image_score, score_map
