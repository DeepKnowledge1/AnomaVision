import torch
import torch.nn as nn
from typing import Optional


class MahalanobisDistance(nn.Module):
    def __init__(
        self,
        mean: Optional[torch.Tensor],       # (N, D)
        cov_inv: Optional[torch.Tensor],    # (N, D, D)
    ):
        """
        A module that computes the Mahalanobis distance using precomputed mean and inverse covariance.

        Args:
            mean: Mean tensor of shape (N, D)
            cov_inv: Inverse covariance tensor of shape (N, D, D)
        """
        super().__init__()
        # Ensure right shapes for ONNX and buffer registration
        self.register_buffer("_mean_flat", mean)             # (N, D)
        self.register_buffer("_cov_inv_flat", cov_inv)       # (N, D, D)

    def forward(self, features: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """
        Compute Mahalanobis distances between features and the stored Gaussian distribution.

        Args:
            features: (B, N, D)  # B: batch, N: num patches, D: feature dim
            width: patch map width
            height: patch map height

        Returns:
            distances: (B, width, height)
        """
        # Check buffer shapes
        assert (
            self._mean_flat is not None and self._cov_inv_flat is not None
        ), "_mean_flat and covariance must be set before calling forward."
        B, N, D = features.shape

        # delta: (B, N, D)
        delta = features - self._mean_flat.unsqueeze(0)  # (B, N, D)
        # For ONNX compatibility: use batch matmul instead of torch.einsum
        # delta.unsqueeze(2): (B, N, 1, D)
        # _cov_inv_flat.unsqueeze(0): (1, N, D, D)
        # result: (B, N, 1, D)
        mahalanobis_left = torch.matmul(delta.unsqueeze(2), self._cov_inv_flat.unsqueeze(0))
        # (B, N, 1, D) x (B, N, D, 1) -> (B, N, 1, 1)
        mahalanobis = torch.matmul(mahalanobis_left, delta.unsqueeze(-1))  # (B, N, 1, 1)
        mahalanobis = mahalanobis.squeeze(-1).squeeze(-1)  # (B, N)
        mahalanobis = mahalanobis.clamp_min(0).sqrt()      # Numerical safety

        # Reshape to (B, width, height)
        distances = mahalanobis.view(B, width, height)
        return distances
