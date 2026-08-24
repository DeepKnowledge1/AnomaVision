"""Single-DPU PatchCore graph for AMD/Xilinx KV260.

The normal AnomaVision PatchCore implementation is unchanged. This backend
keeps the PatchCore similarity tensor in NCHW and uses a DPU-friendly channel
maximum instead of aten::amax, which caused Vitis AI to insert a CPU transpose
before the reduction.
"""

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
        # ResNet layer1. Keep the feature map in native NCHW layout.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        features = self.backbone.layer1(x)

        # Compute similarity against all 819 normalized memory vectors.
        # [B, 64, 56, 56] -> [B, 819, 56, 56]
        similarity = F.conv2d(
            features,
            self.memory_bank,
        )

        # Reduce the memory-bank/channel dimension with torch.max rather than
        # torch.amax. Vitis AI maps the explicit max reduction more reliably
        # for the KV260 DPU and avoids the aten::amax CPU transpose path.
        max_similarity = torch.max(
            similarity,
            dim=1,
            keepdim=True,
        ).values

        # DPU-friendly squared cosine distance:
        #   2 - 2 * max(cosine_similarity)
        # Avoid sqrt/relu/clamp because they introduce unsupported/CPU ops.
        distance = torch.add(
            max_similarity * -2.0,
            2.0,
        )

        # 56x56 -> 224x224 anomaly map.
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
