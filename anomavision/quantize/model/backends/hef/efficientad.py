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

        # AdaptiveAvgPool2d(1) becomes GlobalAveragePool in ONNX. Hailo's
        # quantizer can reject the large spatial reduction, so use fixed
        # AvgPool2d kernels for this fixed 224x224 graph instead.
        self._replace_adaptive_avgpools(self.teacher)
        self._replace_adaptive_avgpools(self.student)

        self.register_buffer("map_mean", model.map_mean.detach().float().clone())
        map_std = model.map_std.detach().float().clone().clamp_min(1e-6)
        self.register_buffer("map_inv_std", map_std.reciprocal())

        self.channel_mean = nn.Conv2d(112, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.channel_mean.weight.fill_(1.0 / 112.0)
        self.channel_mean.weight.requires_grad_(False)

    @staticmethod
    def _replace_adaptive_avgpools(module: nn.Module) -> None:
        """Replace all AdaptiveAvgPool2d(1) modules with fixed spatial kernels."""
        spatial_by_stage = {
            "1": 112,
            "2": 56,
            "3": 28,
            "4": 14,
            "5": 14,
        }

        for name, child in module.named_children():
            if name == "features":
                for stage, stage_module in child.named_children():
                    kernel = spatial_by_stage.get(stage)
                    if kernel is not None:
                        EfficientADHailoGraph._replace_pools_in_module(stage_module, kernel)
            elif isinstance(child, nn.AdaptiveAvgPool2d):
                output_size = child.output_size
                if output_size != 1 and output_size != (1, 1):
                    raise ValueError(
                        f"Unsupported AdaptiveAvgPool2d output_size={output_size}; expected 1"
                    )
            else:
                EfficientADHailoGraph._replace_adaptive_avgpools(child)

    @staticmethod
    def _replace_pools_in_module(module: nn.Module, kernel: int) -> None:
        """Recursively replace AdaptiveAvgPool2d(1) using a known stage size."""
        for name, child in module.named_children():
            if isinstance(child, nn.AdaptiveAvgPool2d):
                output_size = child.output_size
                if output_size != 1 and output_size != (1, 1):
                    raise ValueError(
                        f"Unsupported AdaptiveAvgPool2d output_size={output_size}; expected 1"
                    )
                setattr(module, name, nn.AvgPool2d(kernel_size=kernel, stride=kernel))
            else:
                EfficientADHailoGraph._replace_pools_in_module(child, kernel)

    def forward(self, image: torch.Tensor):
        teacher_features = self.teacher(image)
        student_features = self.student(image)
        diff = student_features - teacher_features
        squared = diff * diff

        # Reduce the 112 feature channels with a fixed 1x1 convolution.
        raw = self.channel_mean(squared)
        raw = F.interpolate(
            raw, size=(224, 224), mode="bilinear", align_corners=False
        )
        normalized = (raw - self.map_mean.unsqueeze(0)) * self.map_inv_std.unsqueeze(0)

        image_scores = F.max_pool2d(
            normalized, kernel_size=(224, 224), stride=(224, 224)
        ).flatten(1)
        return image_scores, normalized.squeeze(1)
