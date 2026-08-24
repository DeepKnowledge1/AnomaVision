"""Single-DPU PatchCore graph for AMD/Xilinx KV260."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCoreKV260(nn.Module):
    """DPU-friendly whole PatchCore graph for KV260."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()

        self.backbone = model.embeddings_extractor.backbone

        memory_bank = F.normalize(
            model.memory_bank.float(),
            dim=-1,
        )

        # Keep memory bank in [819, 64, 1, 1] so Vitis AI can treat
        # each memory vector as a 1x1 convolution filter.
        self.register_buffer(
            "memory_bank",
            memory_bank.reshape(819, 64, 1, 1),
        )

    def forward(self, x: torch.Tensor):

        # ResNet layer1
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        features = self.backbone.layer1(x)
        # features = features

        # Normalize feature vectors.
        # norm_sq = (features * features).sum(
        #     dim=1,
        #     keepdim=True,
        # )

        # # Avoid sqrt/division in the DPU graph.
        # # For ResNet features, norm_sq is safely positive.
        # inv_norm = torch.rsqrt(norm_sq + 1e-12)
        # features = features * inv_norm

        # Compute cosine similarity against memory bank.
        #
        # [B, 64, 56, 56] -> [B, 819, 56, 56]
        similarity = F.conv2d(
            features,
            self.memory_bank,
        )

        # Hierarchical reduction over the 819 memory vectors.
        similarity = similarity.reshape(
            x.shape[0],
            91,
            9,
            56,
            56,
        )

        similarity = similarity.max(dim=2).values

        similarity = similarity.reshape(
            x.shape[0],
            13,
            7,
            56,
            56,
        )

        similarity = similarity.max(dim=2).values

        max_similarity = similarity.max(
            dim=1,
            keepdim=True,
        ).values

        # DPU-friendly squared cosine distance:
        #
        # Original:
        #   sqrt(2 - 2*cosine_similarity)
        #
        # We keep:
        #   2 - 2*cosine_similarity
        #
        # This removes sqrt/relu/clamp from the XIR graph.
        distance = torch.add(
            max_similarity * -2.0,
            2.0,
        )

        # 56x56 -> 224x224
        score_map = F.interpolate(
            distance,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Image-level anomaly score.
        image_score = F.max_pool2d(
            distance,
            kernel_size=(56, 56),
            stride=1,
        ).flatten(1)

        return image_score, score_map
