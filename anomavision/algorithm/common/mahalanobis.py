from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MahalanobisDistance(nn.Module):
    def __init__(self, mean: torch.Tensor, cov_inv: torch.Tensor):
        super().__init__()
        self.register_buffer("_mean_flat", mean)
        self.register_buffer("_cov_inv_flat", cov_inv)
        self._hailo_chunk = 256
        self._hailo_conv_chunks = nn.ModuleList()
        self._build_hailo_convs(cov_inv)
        self._validate_initialization()

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "_hailo_chunk"):
            self._hailo_chunk = 256
        if not hasattr(self, "_hailo_conv_chunks"):
            self._hailo_conv_chunks = nn.ModuleList()
        if len(self._hailo_conv_chunks) == 0 and hasattr(self, "_cov_inv_flat"):
            self._build_hailo_convs(self._cov_inv_flat)

    def _build_hailo_convs(self, cov_inv: torch.Tensor) -> None:
        self._hailo_conv_chunks = nn.ModuleList()
        n, d, d2 = cov_inv.shape
        if d != d2:
            raise ValueError(f"Expected covariance shape (N,D,D), got {cov_inv.shape}")
        for start in range(0, n, self._hailo_chunk):
            end = min(start + self._hailo_chunk, n)
            count = end - start
            conv = nn.Conv2d(count * d, count * d, 1, groups=count, bias=False)
            weight = cov_inv[start:end].permute(0, 2, 1).reshape(count * d, d, 1, 1)
            with torch.no_grad():
                conv.weight.copy_(weight)
            conv.weight.requires_grad_(False)
            self._hailo_conv_chunks.append(conv)

    def _validate_initialization(self):
        if self._mean_flat is None:
            raise RuntimeError("Model not initialized: mean tensor is None.")
        if self._cov_inv_flat is None:
            raise RuntimeError("Model not initialized: inverse covariance is None.")

    def forward(self, features: torch.Tensor, width: int, height: int, chunk: int = 1024, export=False) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(features)}")
        if features.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B,N,D), got tensor with shape {features.shape}")

        B, N, D = features.shape
        if N != width * height:
            raise ValueError(f"Number of patches N ({N}) does not match width*height ({width*height})")

        device = features.device
        dtype = features.dtype
        self._mean_flat = self._mean_flat.to(device=device, dtype=dtype)
        self._cov_inv_flat = self._cov_inv_flat.to(device=device, dtype=dtype)

        if export:
            delta = features - self._mean_flat
            outputs = []
            for i, conv in enumerate(self._hailo_conv_chunks):
                start = i * self._hailo_chunk
                end = min(start + self._hailo_chunk, N)
                count = end - start
                x = delta[:, start:end, :].reshape(B, count * D, 1, 1)
                outputs.append(conv(x).reshape(B, count, D))
            left = torch.cat(outputs, dim=1)
            dist2 = (left * delta).sum(dim=-1)
            return torch.sqrt(torch.clamp(dist2, min=0.0)).reshape(B, width, height)

        # Original PyTorch/KV260 path unchanged.
        delta = features - self._mean_flat.unsqueeze(0)
        left = torch.matmul(delta.unsqueeze(2), self._cov_inv_flat.unsqueeze(0))
        dist2 = torch.matmul(left, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        dist2 = torch.clamp(dist2, min=0.0)
        return torch.sqrt(dist2).reshape(B, width, height)
