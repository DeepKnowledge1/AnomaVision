from typing import Optional

import torch
import torch.nn as nn


class MahalanobisDistance(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        cov_inv: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("_mean_flat", mean)
        self.register_buffer("_cov_inv_flat", cov_inv)
        self._validate_initialization()

    def _validate_initialization(self):
        if self._mean_flat is None:
            raise RuntimeError("Model not initialized: mean tensor is None.")
        if self._cov_inv_flat is None:
            raise RuntimeError("Model not initialized: inverse covariance is None.")

    def forward(
        self,
        features: torch.Tensor,
        width: int,
        height: int,
        chunk: int = 1024,
        export=False,
    ) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(features)}")
        if features.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor (B,N,D), got tensor with shape {features.shape}"
            )

        device = features.device
        dtype = features.dtype
        self._mean_flat = self._mean_flat.to(device=device, dtype=dtype)
        self._cov_inv_flat = self._cov_inv_flat.to(device=device, dtype=dtype)

        B, N, D = features.shape
        if N != width * height:
            raise ValueError(
                f"Number of patches N ({N}) does not match width*height ({width*height})"
            )

        if export:
            # Hailo-friendly per-patch batched matrix multiplication.
            # Flatten B*N so bmm sees each spatial covariance matrix as one batch item.
            delta = features - self._mean_flat.unsqueeze(0)
            delta_flat = delta.reshape(B * N, 1, D)
            cov_flat = self._cov_inv_flat.unsqueeze(0).expand(B, -1, -1, -1)
            cov_flat = cov_flat.reshape(B * N, D, D)
            left = torch.bmm(delta_flat, cov_flat)
            dist2 = torch.bmm(left, delta_flat.transpose(1, 2)).reshape(B, N)
            dist2 = torch.clamp(dist2, min=0.0)
            return torch.sqrt(dist2).reshape(B, width, height)

        # Original KV260/PyTorch path.
        delta = features - self._mean_flat.unsqueeze(0)
        left = torch.matmul(delta.unsqueeze(2), self._cov_inv_flat.unsqueeze(0))
        dist2 = torch.matmul(left, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        dist2 = torch.clamp(dist2, min=0.0)
        distances = torch.sqrt(dist2).reshape(B, width, height)
        return distances
