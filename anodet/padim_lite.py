# anodet/padim_lite.py
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .feature_extraction import ResnetEmbeddingsExtractor
from .mahalanobis import MahalanobisDistance


class PadimLite(torch.nn.Module):
    """
    Minimal runtime module for PaDiM that reconstructs the backbone on load
    and uses stored Gaussian stats (mean, cov_inv). Provides .predict(x)
    with the same outputs as your full model.
    """

    def __init__(
        self,
        backbone: str,
        layer_indices: List[int],
        channel_indices: torch.Tensor,
        mean: torch.Tensor,  # (N, D) fp32
        cov_inv: torch.Tensor,  # (N, D, D) fp32
        device: str = "cpu",
    ):
        """Initialize lightweight PaDiM inference module.

        Creates a minimal runtime module for PaDiM that reconstructs only the backbone
        and uses pre-computed Gaussian statistics. Designed for efficient inference
        without training capabilities.

        Args:
            backbone (str): ResNet backbone name (e.g., "resnet18", "wide_resnet50").
            layer_indices (List[int]): Layer indices for feature extraction.
            channel_indices (torch.Tensor): Pre-selected channel indices for features.
            mean (torch.Tensor): Pre-computed mean vectors of shape (N, D).
            cov_inv (torch.Tensor): Pre-computed inverse covariance of shape (N, D, D).
            device (str, optional): Target device for computation. Defaults to "cpu".

        Example:
            >>> # Create from saved statistics
            >>> lite_model = PadimLite(
            ...     backbone="resnet18",
            ...     layer_indices=[0, 1],
            ...     channel_indices=channel_idx,
            ...     mean=saved_mean,
            ...     cov_inv=saved_cov_inv,
            ...     device="cuda"
            ... )
        """

        super().__init__()
        self.device = torch.device(device)
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        self.layer_indices = layer_indices
        # keep indices small & on device at runtime
        self.register_buffer(
            "channel_indices", channel_indices.to(torch.int32).to(self.device)
        )
        # Mahalanobis holds the stats as buffers
        self.mahalanobisDistance = MahalanobisDistance(
            mean.to(self.device), cov_inv.to(self.device)
        )
        self.eval()

    @torch.no_grad()
    def predict(self, batch: torch.Tensor, export: bool = False):
        """Perform anomaly detection inference on input batch.

        Lightweight prediction method that extracts features and computes anomaly
        scores using pre-computed statistics. Optimized for inference performance
        with minimal memory overhead.

        Args:
            batch (torch.Tensor): Input images of shape (B, C, H, W).
            export (bool, optional): Use export-friendly computation paths.
                Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - image_scores (torch.Tensor): Per-image anomaly scores of shape (B,).
                - score_map (torch.Tensor): Pixel-level anomaly maps of shape (B, H, W).

        Example:
            >>> test_images = torch.randn(4, 3, 224, 224)
            >>> img_scores, score_maps = lite_model.predict(test_images)
        """

        batch = batch.to(self.device, non_blocking=True)
        emb, w, h = self.embeddings_extractor(
            batch,
            channel_indices=self.channel_indices,
            layer_hook=None,
            layer_indices=self.layer_indices,
        )
        patch_scores = self.mahalanobisDistance(emb, w, h, export)  # (B, w, h)
        score_map = F.interpolate(
            patch_scores.unsqueeze(1),
            size=batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        # # image_scores = score_map.flatten(1).max(1).values
        image_scores = patch_scores.flatten(1).amax(1)
        return image_scores, score_map


def build_padim_from_stats(stats: Dict[str, Any], device: str = "cpu") -> PadimLite:
    """Build PadimLite model from saved statistics dictionary.

    Factory function that creates a PadimLite instance from statistics saved
    by the full PaDiM model. Handles precision conversion and device placement
    automatically.

    Args:
        stats (Dict[str, Any]): Statistics dictionary containing keys:
            'mean', 'cov_inv', 'channel_indices', 'layer_indices', 'backbone'.
            Typically created by Padim.save_statistics().
        device (str, optional): Target device for the model. Defaults to "cpu".

    Returns:
        PadimLite: Initialized lightweight model ready for inference.

    Example:
        >>> # Load and create lightweight model
        >>> stats = Padim.load_statistics("model_stats.pth")
        >>> lite_model = build_padim_from_stats(stats, device="cuda")
        >>> img_scores, score_maps = lite_model.predict(test_batch)
    """

    # move/cast back to fp32 CPU first, backend/model will place to proper device
    mean = stats["mean"].float().cpu()
    cov_inv = stats["cov_inv"].float().cpu()
    ch_idx = stats["channel_indices"].to(torch.int64).cpu()
    layers = list(stats["layer_indices"])
    backbone = str(stats["backbone"])
    return PadimLite(
        backbone=backbone,
        layer_indices=layers,
        channel_indices=ch_idx,
        mean=mean,
        cov_inv=cov_inv,
        device=device,
    )
