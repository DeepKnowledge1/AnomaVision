"""Minimal PaDiM backbone graph for AMD/Xilinx KV260."""

from __future__ import annotations

import torch
import torch.nn as nn


class PadimKV260(nn.Module):
    """Minimal PaDiM graph used to isolate KV260 DPU compilation."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.embeddings_extractor.backbone

    def forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        features = self.backbone.layer1(x)

        # Return the backbone feature map twice so the existing quantization
        # script remains unchanged while we isolate compiler compatibility.
        return features, features
