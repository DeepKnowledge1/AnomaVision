# anodet/padim_lite.py
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..common.feature_extraction import ResnetEmbeddingsExtractor
from ..common.mahalanobis import MahalanobisDistance


class PadimLite(torch.nn.Module):
    """Minimal runtime module for PaDiM inference and drift embeddings."""

    def __init__(
        self,
        backbone: str,
        layer_indices: List[int],
        channel_indices: torch.Tensor,
        mean: torch.Tensor,
        cov_inv: torch.Tensor,
        device: str = "cpu",
        force_precision: Optional[str] = None,
    ):
        super().__init__()
        self.device = torch.device(device)

        if force_precision is None:
            self.use_fp16 = self.device.type == "cuda"
            precision_reason = f"auto-detected for {self.device.type.upper()}"
        else:
            self.use_fp16 = force_precision.lower() == "fp16"
            precision_reason = f"forced to {force_precision.upper()}"

        print(
            f"PadimLite: using {'FP16' if self.use_fp16 else 'FP32'} precision ({precision_reason})"
        )

        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        self.layer_indices = layer_indices

        if self.use_fp16 and self.device.type == "cuda":
            mean = mean.half().to(self.device)
            cov_inv = cov_inv.half().to(self.device)
            channel_indices = channel_indices.to(torch.int32).to(self.device)
        else:
            mean = mean.float().to(self.device)
            cov_inv = cov_inv.float().to(self.device)
            channel_indices = channel_indices.to(torch.int32).to(self.device)

        self.register_buffer("channel_indices", channel_indices)
        self.mahalanobisDistance = MahalanobisDistance(mean, cov_inv)

        if self.use_fp16 and self.device.type == "cuda":
            self.embeddings_extractor = self.embeddings_extractor.half()

        self.eval()

    @torch.no_grad()
    def _extract(self, batch: torch.Tensor):
        """Extract the PaDiM patch representation used for anomaly scoring."""
        batch = batch.to(self.device, non_blocking=True)
        if self.use_fp16 and self.device.type == "cuda":
            batch = batch.half()
        emb, _, _ = self.embeddings_extractor(
            batch,
            channel_indices=self.channel_indices,
            layer_hook=None,
            layer_indices=self.layer_indices,
        )
        return emb

    @torch.no_grad()
    def predict(self, batch: torch.Tensor, export: bool = False):
        batch = batch.to(self.device, non_blocking=True)
        if self.use_fp16 and self.device.type == "cuda":
            batch = batch.half()

        emb, w, h = self.embeddings_extractor(
            batch,
            channel_indices=self.channel_indices,
            layer_hook=None,
            layer_indices=self.layer_indices,
        )

        patch_scores = self.mahalanobisDistance(emb, w, h, export)
        score_map = F.interpolate(
            patch_scores.unsqueeze(1),
            size=batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        image_scores = patch_scores.flatten(1).amax(1)

        return image_scores, score_map

    def to_device(self, device: str, force_precision: Optional[str] = None):
        old_device = self.device
        old_precision = "FP16" if self.use_fp16 else "FP32"
        self.device = torch.device(device)

        if force_precision is None:
            self.use_fp16 = self.device.type == "cuda"
            precision_reason = f"auto-detected for {self.device.type.upper()}"
        else:
            self.use_fp16 = force_precision.lower() == "fp16"
            precision_reason = f"forced to {force_precision.upper()}"

        new_precision = "FP16" if self.use_fp16 else "FP32"
        print(
            f"PadimLite: moving from {old_device} ({old_precision}) to {self.device} ({new_precision}) ({precision_reason})"
        )

        if self.use_fp16 and self.device.type == "cuda":
            self = self.half().to(self.device)
        else:
            self = self.float().to(self.device)

        self.embeddings_extractor.to_device(self.device)
        if self.use_fp16 and self.device.type == "cuda":
            self.embeddings_extractor = self.embeddings_extractor.half()
        else:
            self.embeddings_extractor = self.embeddings_extractor.float()


def build_padim_from_stats(
    stats: Dict[str, Any], device: str = "cpu", force_precision: Optional[str] = None
) -> PadimLite:
    """Build PadimLite from a saved PaDiM statistics dictionary."""
    return PadimLite(
        backbone=stats["backbone"],
        layer_indices=stats["layer_indices"],
        channel_indices=stats["channel_indices"],
        mean=stats["mean"],
        cov_inv=stats["cov_inv"],
        device=device,
        force_precision=force_precision,
    )


def load_padim_lite(
    stats_path: str, device: str = "cpu", force_precision: Optional[str] = None
) -> PadimLite:
    """Load PadDiM runtime statistics and construct a PadimLite model."""
    from .padim import Padim

    stats = Padim.load_statistics(stats_path, device=device, force_fp32=None)
    return build_padim_from_stats(stats, device=device, force_precision=force_precision)
