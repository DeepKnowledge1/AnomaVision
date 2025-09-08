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
        image_scores = score_map.flatten(1).max(1).values
        return image_scores, score_map


def build_padim_from_stats(stats: Dict[str, Any], device: str = "cpu") -> PadimLite:
    """
    stats: dict with keys: mean, cov_inv, channel_indices, layer_indices, backbone
    (your save_statistics already writes this)
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
