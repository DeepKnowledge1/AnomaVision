"""Single-DPU PatchCore graph for AMD/Xilinx KV260.

The normal AnomaVision PatchCore implementation is unchanged. This backend
uses a DPU-friendly deployment approximation: normalized memory-bank vectors
are implemented as a 1x1 convolution, and the memory-bank reduction is a
spatial MaxPool2d. Runtime feature L2 normalization is intentionally omitted
because it requires input-dependent rsqrt/div operations that force CPU
subgraphs on the KV260.
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

        memory_bank = F.normalize(model.memory_bank.float(), dim=-1)

        # Dot product with each memory vector is exactly a 1x1 convolution.
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
        features, _, _ = self.extractor(x, layer_indices=[0])

        # (B, 3136, 64) -> (B, 64, 56, 56)
        features = features.reshape(x.shape[0], 56, 56, -1).permute(0, 3, 1, 2)

        # (B, 64, 56, 56) -> (B, 819, 56, 56)
        similarity = self.memory_projection(features)

        # Make memory-bank entries a spatial dimension so the DPU can use
        # MaxPool2d instead of an unsupported channel-wise amax.
        batch = similarity.shape[0]
        similarity = similarity.reshape(batch, 1, 819, 56 * 56)
        max_similarity = F.max_pool2d(
            similarity,
            kernel_size=(819, 1),
            stride=(819, 1),
        )
        max_similarity = max_similarity.reshape(batch, 1, 56, 56)

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
