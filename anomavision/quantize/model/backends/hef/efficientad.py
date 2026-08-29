"""Fixed-shape EfficientAD graph for Hailo end-to-end deployment."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientADHailoGraph(nn.Module):
    """Export-friendly EfficientAD inference graph for Hailo."""

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

        # Reduce the 112 feature channels with a 1x1 convolution instead of
        # ReduceMean/AvgPool. This is equivalent to mean(dim=1) but maps to a
        # standard convolution that Hailo can quantize reliably.
        self.channel_mean = nn.Conv2d(112, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.channel_mean.weight.fill_(1.0 / 112.0)
        self.channel_mean.weight.requires_grad_(False)

    def forward(self, image: torch.Tensor):
        teacher_features = self.teacher(image)
        student_features = self.student(image)
        diff = student_features - teacher_features
        squared = diff * diff

        # EfficientAD produces 112 channels at 14x14. Average pooling performs
        # the channel mean without exporting a ReduceMean node.
        # raw = F.avg_pool2d(squared, kernel_size=(112, 1), stride=(112, 1))
        raw = squared.mean(dim=1, keepdim=True)
        raw = F.interpolate(
            raw, size=(224, 224), mode="bilinear", align_corners=False
        )
        normalized = (raw - self.map_mean.unsqueeze(0)) * self.map_inv_std.unsqueeze(0)

        image_scores = F.max_pool2d(
            normalized, kernel_size=(224, 224), stride=(224, 224)
        ).flatten(1)
        return image_scores, normalized.squeeze(1)
