from typing import Optional

import torch
import torch.nn as nn


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
        chunk: int = 0,  # 0 or <=0 => no chunking (keeps old behavior)
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distances between features and stored Gaussian stats.

        Args:
            features: (B, N, D)  where N = width * height
            width: patch map width
            height: patch map height
            chunk: number of patches to process per chunk (set >0 to cap memory)

        Returns:
            (B, width, height) distances
        """
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor (B,N,D), got {type(features)} with shape {getattr(features,'shape',None)}"
            )

        # Move buffers to the correct device
        device = features.device
        self._mean_flat = self._mean_flat.to(device)
        self._cov_inv_flat = self._cov_inv_flat.to(device)

        B, N, D = features.shape
        if N != width * height:
            raise ValueError(
                f"Number of patches N ({N}) does not match width*height ({width*height})"
            )

        # === Fast vectorized path (keeps original behavior; good for ONNX tracing) ===
        if not chunk or chunk <= 0 or chunk >= N:
            delta = features - self._mean_flat.unsqueeze(0)  # (B, N, D)
            # (B,N,1,D) @ (1,N,D,D) -> (B,N,1,D)
            left = torch.matmul(delta.unsqueeze(2), self._cov_inv_flat.unsqueeze(0))
            # (B,N,1,D) @ (B,N,D,1) -> (B,N,1,1) -> (B,N)
            dist2 = torch.matmul(left, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            distances = dist2.clamp_min_(0).sqrt_().view(B, width, height)
            return distances

        # # === Chunked path (low peak RAM; fixes broadcasting by expanding over B) ===
        # out = features.new_empty(B, N)  # will hold squared distances per patch
        # mean = self._mean_flat
        # pinv = self._cov_inv_flat

        # for s in range(0, N, chunk):
        #     e = min(s + chunk, N)
        #     # f: (B, c, D)   m: (1, c, D)
        #     f = features[:, s:e, :].contiguous()
        #     m = mean[s:e, :].unsqueeze(0)
        #     d = f - m                                          # (B, c, D)

        #     # Broadcast inverse cov over batch: (1,c,D,D) -> (B,c,D,D)
        #     pinv_chunk = pinv[s:e].unsqueeze(0).expand(B, -1, -1, -1).contiguous()

        #     # (B,c,1,D) @ (B,c,D,D) -> (B,c,1,D) -> (B,c,D)
        #     y = torch.matmul(d.unsqueeze(2), pinv_chunk).squeeze(2)

        #     # quadratic form per patch: d^T * Sigma^{-1} * d  -> (B,c)
        #     out[:, s:e] = (y * d).sum(-1)

        # distances = out.clamp_min_(0).sqrt_().view(B, width, height)
        # return distances
