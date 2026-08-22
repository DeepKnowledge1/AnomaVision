"""End-to-end Hailo export graphs for AnomaVision anomaly detectors.

The Hailo path deliberately keeps *all* learned and numerical inference steps in
one export graph: backbone feature extraction, multi-scale feature fusion,
normalization/statistic transforms, distance calculation, score-map upsampling,
and image-score reduction. Hailo's Dataflow Compiler performs the device-side
quantization from a representative calibration set; this module does not claim
that a CPU-side INT8 tensor cast is equivalent to a compiled Hailo graph.

The graphs use fixed input resolution and fixed PaDiM/PatchCore artifacts. This
is required for predictable tensor shapes and for bounded Kria K26 deployment.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from anomavision.feature_extraction import ResnetEmbeddingsExtractor


class _FixedFeatureExtractor(nn.Module):
    """Export-friendly multi-scale ResNet feature extractor."""

    def __init__(self, backbone: str, layer_indices: List[int]) -> None:
        super().__init__()
        self.extractor = ResnetEmbeddingsExtractor(backbone, torch.device("cpu"))
        self.layer_indices = list(layer_indices)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embeddings, _, _ = self.extractor(image, layer_indices=self.layer_indices)
        return embeddings


class PadimEndToEndGraph(nn.Module):
    """Fixed-shape PaDiM graph including Mahalanobis scoring and reductions.

    Inputs are normalized RGB tensors in NCHW format. Outputs are the image score
    and a score map. The Gaussian mean and inverse covariance are constants in the
    graph, so the Hailo compiler can quantize the distance calculation instead of
    leaving it on the Kria CPU.
    """

    def __init__(
        self,
        backbone: str,
        layer_indices: List[int],
        channel_indices: torch.Tensor,
        mean: torch.Tensor,
        cov_inv: torch.Tensor,
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.input_size = tuple(int(v) for v in input_size)
        self.extractor = _FixedFeatureExtractor(backbone, layer_indices)
        self.register_buffer("channel_indices", channel_indices.to(torch.long))
        self.register_buffer("mean", mean.float())
        self.register_buffer("cov_inv", cov_inv.float())

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extractor(image)
        features = torch.index_select(features, 2, self.channel_indices)
        delta = features - self.mean.unsqueeze(0)
        left = torch.matmul(delta.unsqueeze(2), self.cov_inv.unsqueeze(0))
        dist2 = torch.matmul(left, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        patch_scores = torch.sqrt(torch.clamp(dist2, min=0.0))
        patch_count = patch_scores.shape[1]
        side = int(patch_count**0.5)
        if side * side != patch_count:
            raise RuntimeError("PaDiM export requires a square fixed patch grid")
        score_map = patch_scores.reshape(image.shape[0], 1, side, side)
        score_map = F.interpolate(
            score_map, size=self.input_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        image_scores = patch_scores.amax(dim=1)
        return image_scores, score_map


class PatchCoreEndToEndGraph(nn.Module):
    """Fixed-shape PatchCore graph including memory-bank distance and reductions."""

    def __init__(
        self,
        backbone: str,
        layer_indices: List[int],
        memory_bank: torch.Tensor,
        patch_grid: Optional[int] = 14,
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.input_size = tuple(int(v) for v in input_size)
        self.patch_grid = patch_grid
        self.extractor = _FixedFeatureExtractor(backbone, layer_indices)
        self.register_buffer("memory_bank", F.normalize(memory_bank.float(), dim=-1))

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extractor(image)
        batch, patches, channels = features.shape
        side = int(patches**0.5)
        if side * side != patches:
            raise RuntimeError("PatchCore export requires a square fixed patch grid")
        features = F.normalize(features, dim=-1)
        similarity = torch.matmul(features, self.memory_bank.transpose(0, 1))
        distances = torch.sqrt(
            torch.clamp(2.0 - 2.0 * similarity.amax(dim=-1), min=0.0)
        )
        score_map = distances.reshape(batch, 1, side, side)
        score_map = F.interpolate(
            score_map, size=self.input_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        image_scores = distances.amax(dim=1)
        return image_scores, score_map


def exportable_output_names() -> List[str]:
    """Return stable output names shared by ONNX, HEF, and runtime adapters."""

    return ["image_scores", "score_map"]
