"""Single-DPU PaDiM graph for AMD/Xilinx KV260.

This backend is intentionally independent from PatchCore.

The original PaDiM model uses a full 50x50 inverse covariance matrix for each
of the 3136 spatial locations. That exact location-dependent quadratic form
cannot be represented as a normal shared DPU convolution.

For the KV260 single-DPU graph we therefore use the diagonal of each inverse
covariance matrix, preserving the location-dependent variance weighting while
expressing the graph with DPU-friendly NCHW operations.

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

        channel_indices = (
            model.channel_indices.detach().cpu().to(torch.int64)
        )

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

        # ------------------------------------------------------------------
        # 1. Fixed channel selection
        #
        # Original feature map:
        #   [B, 64, 56, 56]
        #
        # PaDiM uses 50 selected channels.
        #
        # A fixed 1x1 convolution avoids index_select/gather in the
        # exported graph.
        # ------------------------------------------------------------------

        projection = torch.zeros(
            50,
            64,
            1,
            1,
            dtype=torch.float32,
        )

        projection[
            torch.arange(50),
            channel_indices,
            0,
            0,
        ] = 1.0

        self.register_buffer(
            "channel_projection",
            projection,
        )

        # ------------------------------------------------------------------
        # 2. Extract diagonal of inverse covariance
        #
        # cov_inv:
        #   [3136, 50, 50]
        #
        # diagonal:
        #   [3136, 50]
        #
        # Each spatial position keeps its own 50 variance weights.
        # ------------------------------------------------------------------

        inv_var = torch.diagonal(
            cov_inv,
            dim1=1,
            dim2=2,
        ).contiguous()

        # ------------------------------------------------------------------
        # 3. Convert statistics to NCHW
        #
        # [3136, 50]
        #       ↓
        # [56, 56, 50]
        #       ↓
        # [1, 50, 56, 56]
        # ------------------------------------------------------------------

        mean_map = (
            mean.reshape(56, 56, 50)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )

        inv_var_map = (
            inv_var.reshape(56, 56, 50)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )

        self.register_buffer(
            "mean_map",
            mean_map,
        )

        self.register_buffer(
            "inv_var_map",
            inv_var_map,
        )

        # ------------------------------------------------------------------
        # 4. Fixed 1x1 convolution for summing the 50 channels
        # ------------------------------------------------------------------

        sum_weights = torch.ones(
            1,
            50,
            1,
            1,
            dtype=torch.float32,
        )

        self.register_buffer(
            "sum_weights",
            sum_weights,
        )

    def forward(self, x: torch.Tensor):

        # ------------------------------------------------------------------
        # ResNet18
        # ------------------------------------------------------------------

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        features = self.backbone.layer1(x)

        # [B, 64, 56, 56]
        #       ↓
        # [B, 50, 56, 56]
        features = F.conv2d(
            features,
            self.channel_projection,
        )

        # ------------------------------------------------------------------
        # Location-dependent mean subtraction
        # ------------------------------------------------------------------

        delta = features - self.mean_map

        # ------------------------------------------------------------------
        # Diagonal Mahalanobis distance
        #
        # d² = Σ ((x - μ)² * cov_inv_diag)
        #
        # All tensors remain NCHW.
        # ------------------------------------------------------------------

        distance_sq = delta * delta

        distance_sq = (
            distance_sq * self.inv_var_map
        )

        # ------------------------------------------------------------------
        # Sum 50 feature dimensions
        #
        # [B, 50, 56, 56]
        #       ↓
        # [B, 1, 56, 56]
        # ------------------------------------------------------------------

        distance_sq = F.conv2d(
            distance_sq,
            self.sum_weights,
        )

        # ------------------------------------------------------------------
        # Upsample anomaly map
        # ------------------------------------------------------------------

        score_map = F.interpolate(
            distance_sq,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # ------------------------------------------------------------------
        # Image-level anomaly score
        # ------------------------------------------------------------------

        image_score = F.adaptive_max_pool2d(
            distance_sq,
            output_size=(1, 1),
        ).flatten(1)

        return image_score, score_map
