"""
Provides classes and functions for working with PaDiM.
"""

import random
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...utils import pytorch_cov, split_tensor_and_run_function
from ..common.feature_extraction import ResnetEmbeddingsExtractor
from ..common.mahalanobis import MahalanobisDistance

BACKBONE_FEATURE_SIZES = {
    "resnet18": OrderedDict([(0, [64]), (1, [128]), (2, [256]), (3, [512])]),
    "wide_resnet50": OrderedDict([(0, [256]), (1, [512]), (2, [1024]), (3, [2048])]),
}


class Padim(torch.nn.Module):
    """A padim model with functions to train and perform inference."""

    def __init__(
        self,
        backbone: str = "resnet18",
        device: torch.device = torch.device("cpu"),
        channel_indices: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        feat_dim: Optional[int] = 50,
    ) -> None:
        super(Padim, self).__init__()

        self.device = torch.device(device=device)
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        self.layer_indices = layer_indices

        if self.layer_indices is None:
            self.layer_indices = [0, 1]

        self.layer_hook = layer_hook

        if backbone not in BACKBONE_FEATURE_SIZES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Available: {list(BACKBONE_FEATURE_SIZES.keys())}"
            )
        self.net_feature_size = BACKBONE_FEATURE_SIZES[backbone]

        if channel_indices is not None:
            self.register_buffer("channel_indices", channel_indices.to(self.device))
        else:
            channel_indices_tensor = get_dims_indices(
                self.layer_indices, feat_dim, self.net_feature_size
            )
            self.register_buffer(
                "channel_indices", channel_indices_tensor.to(self.device)
            )

    @property
    def mean(self):
        """Get the mean tensor."""
        return self._mean

    @property
    def cov_inv(self):
        """Get the inverse covariance tensor."""
        return self._cov_inv

    @torch.no_grad()
    def _extract(self, batch: torch.Tensor):
        """Extract the PaDiM patch representation used for anomaly scoring and drift."""
        batch = batch.to(self.device, non_blocking=True)
        embedding_vectors, _, _ = self.embeddings_extractor(
            batch,
            channel_indices=self.channel_indices,
            layer_hook=self.layer_hook,
            layer_indices=self.layer_indices,
        )
        return embedding_vectors

    def forward(
        self, x: torch.Tensor, return_map: bool = True, export: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform anomaly detection inference on input images."""
        embedding_vectors, w, h = self.embeddings_extractor(
            x,
            channel_indices=self.channel_indices,
            layer_hook=self.layer_hook,
            layer_indices=self.layer_indices,
        )
        embedding_vectors = embedding_vectors.to(dtype=x.dtype)
        patch_scores = self.mahalanobisDistance(
            features=embedding_vectors, width=w, height=h, export=export, chunk=256
        )
        image_scores = patch_scores.flatten(1).amax(1)

        score_map = F.interpolate(
            patch_scores.unsqueeze(1),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return image_scores, score_map

    def to_device(self, device: torch.device) -> None:
        """Perform device conversion on backbone and statistics."""
        self.device = device
        if self.embeddings_extractor is not None:
            self.embeddings_extractor.to_device(device)

    def fit(
        self, dataloader: torch.utils.data.DataLoader, extractions: int = 1
    ) -> None:
        """Fit the PaDiM model to normal training data."""
        embedding_vectors = None
        for _ in range(extractions):
            extracted_embedding_vectors = self.embeddings_extractor.from_dataloader(
                dataloader,
                channel_indices=self.channel_indices,
                layer_hook=self.layer_hook,
                layer_indices=self.layer_indices,
            )
            if embedding_vectors is None:
                embedding_vectors = extracted_embedding_vectors
            else:
                embedding_vectors = torch.cat(
                    (embedding_vectors, extracted_embedding_vectors), 0
                )

        mean = torch.mean(embedding_vectors, dim=0)
        cov = pytorch_cov(
            embedding_vectors.permute(1, 0, 2), rowvar=False
        ) + 0.01 * torch.eye(embedding_vectors.shape[2])
        cov_inv = split_tensor_and_run_function(
            func=torch.inverse, tensor=cov, split_size=1
        )

        self.mahalanobisDistance = MahalanobisDistance(mean, cov_inv)

    def predict(
        self, batch: torch.Tensor, export: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make anomaly predictions on a batch."""
        assert (
            self.mahalanobisDistance._mean_flat is not None
            and self.mahalanobisDistance._cov_inv_flat is not None
        ), "Model is not trained. Please call `fit()` first."
        return self(batch, export=export)

    def save_statistics(self, path: str, half: Optional[bool] = None) -> None:
        """Save trained model statistics to disk."""
        if (
            self.mahalanobisDistance._mean_flat is None
            or self.mahalanobisDistance._cov_inv_flat is None
        ):
            raise RuntimeError("Model is not trained. Call fit() first.")

        if half is None:
            half = torch.cuda.is_available()
            print(
                f"Auto-detected precision: {'FP16' if half else 'FP32'} (GPU available: {torch.cuda.is_available()})"
            )

        mean_tensor = self.mahalanobisDistance._mean_flat.detach().cpu()
        cov_inv_tensor = self.mahalanobisDistance._cov_inv_flat.detach().cpu()
        channel_indices_tensor = self.channel_indices.detach().cpu()

        if half:
            mean_tensor = mean_tensor.half()
            cov_inv_tensor = cov_inv_tensor.half()
            dtype_info = "fp16"
        else:
            mean_tensor = mean_tensor.float()
            cov_inv_tensor = cov_inv_tensor.float()
            dtype_info = "fp32"

        stats = {
            "mean": mean_tensor,
            "cov_inv": cov_inv_tensor,
            "channel_indices": channel_indices_tensor,
            "layer_indices": list(self.layer_indices),
            "backbone": self.embeddings_extractor.backbone_name,
            "model_version": "1.0",
            "dtype": dtype_info,
        }

        torch.save(stats, path)
        print(f"Statistics saved to {path} using {dtype_info.upper()} precision")

    @staticmethod
    def load_statistics(
        path: str, device: str = "cpu", force_fp32: Optional[bool] = None
    ):
        """Load previously saved PaDiM statistics."""
        stats = torch.load(path, map_location="cpu", weights_only=False)
        saved_dtype = stats.get("dtype", "fp32")

        if force_fp32 is None:
            device_obj = torch.device(device)
            if device_obj.type == "cpu":
                force_fp32 = True
                reason = "CPU device detected"
            else:
                force_fp32 = False
                reason = "GPU device detected"
            print(
                f"Auto-detected precision handling: {'FP32' if force_fp32 else f'preserve saved ({saved_dtype.upper()})'} ({reason})"
            )

        if force_fp32:
            stats["mean"] = stats["mean"].float().to(device)
            stats["cov_inv"] = stats["cov_inv"].float().to(device)
            final_dtype = "fp32"
        else:
            stats["mean"] = stats["mean"].to(device)
            stats["cov_inv"] = stats["cov_inv"].to(device)
            final_dtype = saved_dtype

        stats["channel_indices"] = stats["channel_indices"].to(torch.int64).to(device)

        print(f"Statistics loaded from {path}")
        print(f"Saved as: {saved_dtype.upper()}, loaded as: {final_dtype.upper()}")
        return stats


def get_dims_indices(layers, feature_dim, net_feature_size):
    """Generate random channel indices for feature dimensionality reduction."""
    total_features = sum(net_feature_size[layer][0] for layer in layers)
    if feature_dim is None or feature_dim >= total_features:
        return torch.arange(total_features, dtype=torch.long)

    random.seed(0)
    return torch.tensor(
        sorted(random.sample(range(total_features), feature_dim)), dtype=torch.long
    )
