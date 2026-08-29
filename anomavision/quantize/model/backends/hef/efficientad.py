"""Fixed-shape EfficientAD graph for Hailo end-to-end deployment."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientADHailoGraph(nn.Module):
    """Export-friendly EfficientAD inference graph.

    The teacher, student and calibrated map statistics are kept inside the graph.
    Hailo can therefore quantize the complete anomaly-scoring path instead of
    running the scoring logic on the host CPU.

    The graph intentionally avoids ``pow``, ``mean`` reductions, division and
    global ``amax``. Squaring is written as multiplication, channel averaging
    as average pooling, normalization as multiply-by-inverse-std, and the final
    image score as fixed-size max pooling.
    """

    def __init__(self, model: nn.Module, input_size: Tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        if tuple(input_size) != (224, 224):
            raise ValueError("Hailo EfficientAD export currently requires input_size=(224, 224)")

        self.input_size = (224, 224)
        self.teacher = model.teacher.eval()
        self.student = model.student.eval()
        self.register_buffer("map_mean", model.map_mean.detach().float().clone())
        map_std = model.map_std.detach().float().clone().clamp_min(1e-6)
        self.register_buffer("map_inv_std", map_std.reciprocal())

    def forward(self, image: torch.Tensor):
        teacher_features = self.teacher(image)
        student_features = self.student(image)
        diff = student_features - teacher_features
        squared = diff * diff

        # EfficientAD produces 112 channels at 14x14. Average pooling performs
        # the channel mean without exporting a ReduceMean node.
        raw = F.avg_pool2d(squared, kernel_size=(112, 1), stride=(112, 1))
        raw = F.interpolate(
            raw, size=(224, 224), mode="bilinear", align_corners=False
        )
        normalized = (raw - self.map_mean.unsqueeze(0)) * self.map_inv_std.unsqueeze(0)

        # Fixed global max via pooling avoids an unsupported global ReduceMax.
        image_scores = F.max_pool2d(
            normalized, kernel_size=(224, 224), stride=(224, 224)
        ).flatten(1)
        return image_scores, normalized.squeeze(1)
