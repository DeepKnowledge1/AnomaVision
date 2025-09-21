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

        Example:
            >>> mean = torch.randn(100, 256)  # 100 locations, 256 features
            >>> cov_inv = torch.eye(256).unsqueeze(0).repeat(100, 1, 1)
            >>> mahal = MahalanobisDistance(mean, cov_inv)
        """
        super().__init__()

        # Ensure right shapes for ONNX and buffer registration
        self.register_buffer("_mean_flat", mean)  # (N, D)
        self.register_buffer("_cov_inv_flat", cov_inv)  # (N, D, D)
        self._validate_initialization()

    def _validate_initialization(self):
        """
        Validate that the Mahalanobis distance model is properly initialized.

        This internal method ensures that required statistical parameters (mean and
        inverse covariance) are available before attempting distance calculations.

        Raises:
            RuntimeError: If mean tensor is None - indicates model needs fitting or
                explicit mean parameter.
            RuntimeError: If inverse covariance tensor is None - indicates model
                needs fitting or explicit covariance parameter.

        Note:
            This is called automatically during forward pass to prevent
            computation with uninitialized parameters.
        """

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
        """Compute Mahalanobis distances for anomaly detection.

        Calculates Mahalanobis distances between input features and stored Gaussian
        statistics at each spatial location. Supports memory-efficient chunked
        computation for large feature maps.

        Args:
            features (torch.Tensor): Input features of shape (B, N, D) where
                B is batch size, N is number of patches (width * height),
                and D is feature dimension.
            width (int): Spatial width of the patch map.
            height (int): Spatial height of the patch map.
            chunk (int, optional): Number of patches to process per chunk for
                memory efficiency. If 0 or >= N, uses fully vectorized computation.
                Defaults to 1024.
            export (bool, optional): If True, forces vectorized computation path
                for ONNX export compatibility. Defaults to False.

        Returns:
            torch.Tensor: Mahalanobis distances of shape (B, width, height).
                Higher values indicate greater deviation from normal patterns.

        Raises:
            TypeError: If features is not a torch.Tensor.
            ValueError: If features doesn't have expected 3D shape or if
                N doesn't match width * height.

        Example:
            >>> features = torch.randn(4, 100, 256)  # 4 images, 100 patches, 256 dims
            >>> distances = mahal_dist(features, width=10, height=10)
            >>> print(distances.shape)  # torch.Size([4, 10, 10])
        """

        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(features)}")

        if features.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor (B,N,D), got tensor with shape {features.shape}"
            )

        # if not isinstance(features, torch.Tensor) or features.ndim != 3:
        #     raise ValueError(
        #         f"Expected 3D tensor (B,N,D), got {type(features)} with shape {getattr(features,'shape',None)}"
        #     )

        # Move buffers to the correct device
        device = features.device

        dtype = features.dtype  # check whether it's float32, float16, etc.

        # self._mean_flat = self._mean_flat.to(device=device, dtype=dtype)
        # self._cov_inv_flat = self._cov_inv_flat.to(device=device, dtype=dtype)

        self._mean_flat = self._mean_flat.to(features)
        self._cov_inv_flat = self._cov_inv_flat.to(features)

        # self._mean_flat = self._mean_flat.to(device)
        # self._cov_inv_flat = self._cov_inv_flat.to(device)

        B, N, D = features.shape
        if N != width * height:
            raise ValueError(
                f"Number of patches N ({N}) does not match width*height ({width*height})"
            )

        # Always use vectorized path during ONNX export (to keep graph small)
        if True:  #   export or not chunk or chunk <= 0 or chunk >= N:
            delta = features - self._mean_flat.unsqueeze(0)  # (B, N, D)
            # (B,N,1,D) @ (1,N,D,D) -> (B,N,1,D)
            left = torch.matmul(delta.unsqueeze(2), self._cov_inv_flat.unsqueeze(0))
            # (B,N,1,D) @ (B,N,D,1) -> (B,N,1,1) -> (B,N)
            dist2 = torch.matmul(left, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            distances = dist2.clamp_min_(0).sqrt_().view(B, width, height)
            return distances

        # === Chunked path (low peak RAM; fixes broadcasting by expanding over B) ===
        out = features.new_empty(B, N)  # will hold squared distances per patch
        mean = self._mean_flat
        pinv = self._cov_inv_flat

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            # f: (B, c, D)   m: (1, c, D)
            f = features[:, s:e, :].contiguous()
            m = mean[s:e, :].unsqueeze(0)
            d = f - m  # (B, c, D)

            # Broadcast inverse cov over batch: (1,c,D,D) -> (B,c,D,D)
            pinv_chunk = pinv[s:e].unsqueeze(0).expand(B, -1, -1, -1).contiguous()

            # (B,c,1,D) @ (B,c,D,D) -> (B,c,1,D) -> (B,c,D)
            y = torch.matmul(d.unsqueeze(2), pinv_chunk).squeeze(2)

            # quadratic form per patch: d^T * Sigma^{-1} * d  -> (B,c)
            out[:, s:e] = (y * d).sum(-1)

        distances = out.clamp_min_(0).sqrt_().view(B, width, height)
        return distances
