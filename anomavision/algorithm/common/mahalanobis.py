from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            # Hailo DFC does not reliably parse the arbitrary MatMul used by
            # Mahalanobis distance when covariance is an ONNX initializer. It
            # expects MatMul in supported patterns (not this per-patch case).
            # Express the same operation as grouped 1x1 convolutions instead.
            #
            # For each patch i:
            #   y[i,k] = sum_j delta[i,j] * cov_inv[i,j,k]
            # A group corresponds to one spatial patch and has D input/output
            # channels. Chunking keeps every Conv weight tensor below Hailo's
            # per-layer weight-size limit.
            delta = features - self._mean_flat
            conv_chunk = 256
            outputs = []

            for start in range(0, N, conv_chunk):
                end = min(start + conv_chunk, N)
                count = end - start

                delta_chunk = delta[:, start:end, :].reshape(
                    B, count * D, 1, 1
                )

                # cov_inv: (count, D, D)
                # Conv2d weights: (count*D, D, 1, 1)
                # Each group handles one patch independently.
                weight = self._cov_inv_flat[start:end].permute(0, 2, 1).reshape(
                    count * D, D, 1, 1
                )

                y = F.conv2d(
                    delta_chunk,
                    weight,
                    bias=None,
                    stride=1,
                    padding=0,
                    groups=count,
                )
                outputs.append(y.reshape(B, count, D))

            left = torch.cat(outputs, dim=1)
            dist2 = (left * delta).sum(dim=-1)
            dist2 = torch.clamp(dist2, min=0.0)
            return torch.sqrt(dist2).reshape(B, width, height)

        # Original PyTorch/KV260 path intentionally unchanged.
        delta = features - self._mean_flat.unsqueeze(0)
        left = torch.matmul(delta.unsqueeze(2), self._cov_inv_flat.unsqueeze(0))
        dist2 = torch.matmul(left, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        dist2 = torch.clamp(dist2, min=0.0)
        distances = torch.sqrt(dist2).reshape(B, width, height)
        return distances
