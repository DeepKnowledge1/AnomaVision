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
            # Hailo-friendly formulation with no Unsqueeze nodes.
            # features/mean: (B,N,D)
            # transpose -> (N,B,D), allowing one covariance matrix per patch.
            delta = features - self._mean_flat
            delta_nbd = delta.transpose(0, 1)

            # (N,B,D) @ (N,D,D) -> (N,B,D)
            left_nbd = torch.matmul(delta_nbd, self._cov_inv_flat)
            left = left_nbd.transpose(0, 1)

            # d^T Sigma^-1 d, computed elementwise to avoid a second MatMul.
            dist2 = (left * delta).sum(dim=-1)
            dist2 = torch.clamp(dist2, min=0.0)
            return torch.sqrt(dist2).reshape(B, width, height)

        # Original KV260/PyTorch path is intentionally unchanged.
        delta = features - self._mean_flat.unsqueeze(0)
        left = torch.matmul(delta.unsqueeze(2), self._cov_inv_flat.unsqueeze(0))
        dist2 = torch.matmul(left, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        dist2 = torch.clamp(dist2, min=0.0)
        distances = torch.sqrt(dist2).reshape(B, width, height)
        return distances
