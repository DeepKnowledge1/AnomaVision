import torch
import torch.nn as nn
from typing import Optional


class MahalanobisDistance(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,  # (N, D)
        cov_inv: torch.Tensor,  # (N, D, D)
    ):
        """
        A module that computes the Mahalanobis distance using precomputed mean and inverse covariance.

        Args:
            mean: Mean tensor of shape (N, D)
            cov_inv: Inverse covariance tensor of shape (N, D, D)
        """
        super().__init__()

        # Ensure right shapes for ONNX and buffer registration
        self.register_buffer("_mean_flat", mean)  # (N, D)
        self.register_buffer("_cov_inv_flat", cov_inv)  # (N, D, D)
        self._validate_initialization()

    def _validate_initialization(self):
        """Validate that the model is properly initialized."""
        if self._mean_flat is None:
            raise RuntimeError("Model not initialized: mean tensor is None. "
                            "Please fit the model first or provide mean tensor.")

        if self._cov_inv_flat is None:
            raise RuntimeError("Model not initialized: inverse covariance is None. "
                            "Please fit the model first or provide covariance tensor.")


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
        # Validate inputs
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(features)}")

        if len(features.shape) != 3:
            raise ValueError(f"Expected 3D features tensor (B,N,D), got shape {features.shape}")

        self._mean_flat = self._mean_flat.to(features.device)
        self._cov_inv_flat = self._cov_inv_flat.to(features.device)

        B, N, D = features.shape

        # delta: (B, N, D)
        delta = features - self._mean_flat.unsqueeze(0)  # (B, N, D)
        # For ONNX compatibility: use batch matmul instead of torch.einsum
        # delta.unsqueeze(2): (B, N, 1, D)
        # _cov_inv_flat.unsqueeze(0): (1, N, D, D)
        # result: (B, N, 1, D)
        mahalanobis_left = torch.matmul(
            delta.unsqueeze(2), self._cov_inv_flat.unsqueeze(0)
        )
        # (B, N, 1, D) x (B, N, D, 1) -> (B, N, 1, 1)
        mahalanobis = torch.matmul(
            mahalanobis_left, delta.unsqueeze(-1)
        )  # (B, N, 1, 1)
        mahalanobis = mahalanobis.squeeze(-1).squeeze(-1)  # (B, N)
        mahalanobis = mahalanobis.clamp_min(0).sqrt()  # Numerical safety

        # Reshape to (B, width, height)
        if N != width * height:
            raise ValueError(f"Number of patches N ({N}) does not match width * height ({width * height})")
        distances = mahalanobis.view(B, width, height)
        return distances
