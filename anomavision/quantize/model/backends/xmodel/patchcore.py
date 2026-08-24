"""Single-DPU PatchCore graph for AMD/Xilinx KV260.

The normal AnomaVision PatchCore implementation is unchanged. This backend
keeps the ResNet layer-1 feature map in native NCHW layout so the KV260 DPU
compiler does not have to insert CPU transpose subgraphs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCoreKV260(nn.Module):
    """DPU-friendly whole PatchCore graph for KV260."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.extractor = model.embeddings_extractor
        self.backbone = self.extractor.backbone

        memory_bank = F.normalize(model.memory_bank.float(), dim=-1)

        # Dot product with each normalized memory vector is a 1x1 convolution.
        self.memory_projection = nn.Conv2d(
            in_channels=memory_bank.shape[1],
            out_channels=memory_bank.shape[0],
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            self.memory_projection.weight.copy_(
                memory_bank.reshape(memory_bank.shape[0], memory_bank.shape[1], 1, 1)
            )
        self.memory_projection.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        # Reproduce ResnetEmbeddingsExtractor(layer_indices=[0]) but keep the
        # native NCHW feature map. The public extractor still returns NLC and
        # remains unchanged for normal AnomaVision inference.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        features = self.backbone.layer1(x)

        # features: (B, 64, 56, 56)
        similarity = self.memory_projection(features)

        # Reduce 819 memory entries without an 819-wide pool kernel. The
        # reduction is hierarchical and uses reshape + max operations only.
        batch = similarity.shape[0]
        similarity = similarity.reshape(batch, 91, 9, 56, 56)
        similarity = similarity.max(dim=2).values
        similarity = similarity.reshape(batch, 13, 7, 56, 56)
        max_similarity = similarity.max(dim=2).values
        max_similarity = max_similarity.max(dim=1, keepdim=True).values

        # Monotonic cosine-distance transform without SUB/DIV.
        distance = torch.add(max_similarity * -1.0, 1.0)

        # 56x56 -> 224x224 anomaly map.
        score_map = F.interpolate(
            distance,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Image-level score stays in the DPU graph.
        image_score = F.max_pool2d(
            distance,
            kernel_size=(56, 56),
            stride=1,
        ).flatten(1)

        return image_score, score_map
