"""DPU-friendly PatchCore graph for AMD/Xilinx KV260.

The regular PatchCore implementation is intentionally unchanged. This wrapper
keeps the same cosine-distance PatchCore computation while expressing the hot
path with primitives that the KV260 DPU compiler can keep together:

* L2 normalization is written as multiply-by-rsqrt instead of division.
* cosine distance uses add + multiply instead of subtraction.
* memory-bank reduction is performed over the channel dimension, which maps to
  the KV260 reduction-max path.
* image-level max uses global MaxPool2d instead of a spatial reduction.

The mathematical distance remains sqrt(2 - 2*cosine_similarity), matching the
existing PatchCore implementation up to normal floating-point epsilon behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCoreKV260(nn.Module):
    """Whole PatchCore inference graph expressed for the KV260 DPU."""

    def __init__(self, model: nn.Module, eps: float = 1e-12) -> None:
        super().__init__()
        self.extractor = model.embeddings_extractor
        self.eps = float(eps)

        memory_bank = model.memory_bank.float()
        self.register_buffer("memory_bank", self._l2_normalize(memory_bank))

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """L2-normalize without a division operator in the traced graph."""
        squared = x * x
        norm_sq = squared.sum(dim=-1, keepdim=True)
        inv_norm = torch.rsqrt(norm_sq.clamp_min(eps * eps))
        return x * inv_norm

    def forward(self, x: torch.Tensor):
        features, _, _ = self.extractor(x, layer_indices=[0])

        # Existing PatchCore uses normalized patch embeddings.
        features = self._l2_normalize(features, self.eps)

        # (B, 3136, 64) @ (64, 819) -> (B, 3136, 819)
        similarity = torch.matmul(features, self.memory_bank.transpose(0, 1))

        # Put the memory-bank dimension into the DPU channel dimension:
        # (B, 3136, 819) -> (B, 819, 56, 56)
        similarity = similarity.reshape(x.shape[0], 56, 56, -1)
        similarity = similarity.permute(0, 3, 1, 2)

        # Reduction over channel maps to the KV260 DPU reduction-max path.
        max_similarity = torch.amax(similarity, dim=1, keepdim=True)

        # Algebraically identical to 2 - 2*max_similarity, but uses add/mul
        # instead of an unsupported DPU eltwise subtraction.
        distance_sq = torch.add(
            max_similarity * -2.0,
            2.0,
        )
        distance_sq = distance_sq.clamp_min(0.0)
        distances = torch.sqrt(distance_sq)

        # PatchCore spatial anomaly map: 56x56 -> 224x224.
        score_map = F.interpolate(
            distances,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Global max pooling keeps image-level scoring inside the DPU graph.
        image_score = F.max_pool2d(
            distances,
            kernel_size=(56, 56),
            stride=1,
        ).flatten(1)

        return image_score, score_map
