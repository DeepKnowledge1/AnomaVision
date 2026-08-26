from typing import Optional

import torch
import torch.nn as nn


class MahalanobisDistance(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,  # (N, D)
        cov_inv: torch.Tensor,  # (N, D, D)
    ):
        """Initialize Mahalanobis distance module with precomputed statistics.

        Creates a module that computes Mahalanobis distances using precomputed mean
        vectors and inverse covariance matrices. Registers tensors as buffers for
        proper device management and model state handling.

        Args:
            mean (torch.Tensor): Mean vectors of shape (N, D) where N is number of
                spatial locations and D is feature dimension.
            cov_inv (torch.Tensor): Inverse covariance matrices of shape (N, D, D)
                for each spatial location.
        """
        super().__init__()

        self.register_buffer("_mean_flat", mean)  # (N, D)
        self.register_buffer("_cov_inv_flat", cov_inv)  # (N, D, D)
        self._validate_initialization()

    def _validate_initialization(self):
        if self._mean_flat is None:
            raise RuntimeError(
                "Model not initialized: mean tensor is None. "
                "Please fit the model first or provide mean tensor."
            )

        if self._cov_inv_flat is None:
            raise RuntimeError(
                "Model not initialized: inverse covariance is None. "
                "Please fit the model first or provide covariance tensor."
            )

    def forward(
        self,
        features: torch.Tensor,  # (B, N, D)
        width: int,
        height: int,
        chunk: int = 1024,
        export=False,
    ) -> torch.Tensor:
        """Compute Mahalanobis distances for anomaly detection."""

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
            # Hailo-friendly export path.
            # Keep the original KV260/PyTorch computation below unchanged.
            delta = features - self._mean_flat.unsqueeze(0)  # (B,N,D)

            # [B,N,D] @ [N,D,D] -> [B,N,D]
            # Avoid the Unsqueeze -> MatMul -> Unsqueeze -> MatMul pattern
            # that Hailo DFC 3.34 fails to translate.
            left = torch.matmul(delta, self._cov_inv_flat)

            # d^T Sigma^-1 d == (Sigma^-1 d) * d, summed over D.
            dist2 = (left * delta).sum(dim=-1)  # (B,N)

            dist2 = torch.clamp(dist2, min=0.0)
            return torch.sqrt(dist2).reshape(B, width, height)

        # Original KV260/PyTorch path - unchanged.
        delta = features - self._mean_flat.unsqueeze(0)  # (B, N, D)
        left = torch.matmul(delta.unsqueeze(2), self._cov_inv_flat.unsqueeze(0))
        dist2 = torch.matmul(left, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        dist2 = torch.clamp(dist2, min=0.0)
        distances = torch.sqrt(dist2).reshape(B, width, height)
        return distances
