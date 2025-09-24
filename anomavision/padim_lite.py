# anodet/padim_lite.py
from typing import Any, Dict, List, Optional

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
        mean: torch.Tensor,  # (N, D)
        cov_inv: torch.Tensor,  # (N, D, D)
        device: str = "cpu",
        force_precision: Optional[str] = None,
    ):
        """Initialize lightweight PaDiM inference module with device-aware precision.

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
            force_precision (Optional[str], optional): Force specific precision ("fp16" or "fp32").
                If None, auto-detects: FP16 for GPU, FP32 for CPU.

        Example:
            >>> # Create from saved statistics with auto-precision
            >>> lite_model = PadimLite(
            ...     backbone="resnet18",
            ...     layer_indices=[0, 1],
            ...     channel_indices=channel_idx,
            ...     mean=saved_mean,
            ...     cov_inv=saved_cov_inv,
            ...     device="cuda"  # Auto-uses FP16
            ... )
            >>> # Force FP32 even on GPU
            >>> lite_model = PadimLite(..., device="cuda", force_precision="fp32")
        """

        super().__init__()
        self.device = torch.device(device)

        # Determine precision based on device if not forced
        if force_precision is None:
            self.use_fp16 = self.device.type == "cuda"
            precision_reason = f"auto-detected for {self.device.type.upper()}"
        else:
            self.use_fp16 = force_precision.lower() == "fp16"
            precision_reason = f"forced to {force_precision.upper()}"

        print(
            f"PadimLite: using {'FP16' if self.use_fp16 else 'FP32'} precision ({precision_reason})"
        )

        # Initialize backbone
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        self.layer_indices = layer_indices

        # Convert tensors to appropriate precision and device
        if self.use_fp16 and self.device.type == "cuda":
            mean = mean.half().to(self.device)
            cov_inv = cov_inv.half().to(self.device)
            channel_indices = channel_indices.to(torch.int64).to(self.device)
        else:
            mean = mean.float().to(self.device)
            cov_inv = cov_inv.float().to(self.device)
            channel_indices = channel_indices.to(torch.int64).to(self.device)

        # Register channel indices as buffer
        self.register_buffer("channel_indices", channel_indices)

        # Initialize Mahalanobis distance with precision-converted tensors
        self.mahalanobisDistance = MahalanobisDistance(mean, cov_inv)

        # Convert backbone to matching precision if needed
        if self.use_fp16 and self.device.type == "cuda":
            self.embeddings_extractor = self.embeddings_extractor.half()

        self.eval()

    @torch.no_grad()
    def predict(self, batch: torch.Tensor, export: bool = False):
        """Perform anomaly detection inference on input batch with precision handling.

        Lightweight prediction method that extracts features and computes anomaly
        scores using pre-computed statistics. Optimized for inference performance
        with minimal memory overhead and automatic precision handling.

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

        # Move batch to device and convert precision if needed
        batch = batch.to(self.device, non_blocking=True)
        if self.use_fp16 and self.device.type == "cuda":
            batch = batch.half()

        # Extract embeddings
        emb, w, h = self.embeddings_extractor(
            batch,
            channel_indices=self.channel_indices,
            layer_hook=None,
            layer_indices=self.layer_indices,
        )

        # Compute patch scores
        patch_scores = self.mahalanobisDistance(emb, w, h, export)  # (B, w, h)

        # Upsample to input resolution
        score_map = F.interpolate(
            patch_scores.unsqueeze(1),
            size=batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Compute image-level scores
        image_scores = patch_scores.flatten(1).amax(1)

        return image_scores, score_map

    def to_device(self, device: str, force_precision: Optional[str] = None):
        """Move model to new device with optional precision conversion.

        Args:
            device (str): Target device ("cuda", "cpu", etc.).
            force_precision (Optional[str]): Force specific precision, or None to auto-detect.

        Example:
            >>> lite_model.to_device("cuda")  # Auto FP16
            >>> lite_model.to_device("cpu")   # Auto FP32
            >>> lite_model.to_device("cuda", "fp32")  # Force FP32 on GPU
        """
        old_device = self.device
        old_precision = "FP16" if self.use_fp16 else "FP32"

        self.device = torch.device(device)

        # Determine new precision
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

        # Move and convert precision
        if self.use_fp16 and self.device.type == "cuda":
            self = self.half().to(self.device)
        else:
            self = self.float().to(self.device)

        # Update embeddings extractor
        self.embeddings_extractor.to_device(self.device)
        if self.use_fp16 and self.device.type == "cuda":
            self.embeddings_extractor = self.embeddings_extractor.half()
        else:
            self.embeddings_extractor = self.embeddings_extractor.float()


def build_padim_from_stats(
    stats: Dict[str, Any], device: str = "cpu", force_precision: Optional[str] = None
) -> PadimLite:
    """Build PadimLite model from saved statistics dictionary with device-aware precision.

    Factory function that creates a PadimLite instance from statistics saved
    by the full PaDiM model. Handles precision conversion and device placement
    automatically based on target device.

    Args:
        stats (Dict[str, Any]): Statistics dictionary containing keys:
            'mean', 'cov_inv', 'channel_indices', 'layer_indices', 'backbone'.
            Typically created by Padim.save_statistics().
        device (str, optional): Target device for the model. Defaults to "cpu".
        force_precision (Optional[str], optional): Force specific precision ("fp16" or "fp32").
            If None, auto-detects: FP16 for GPU, FP32 for CPU.

    Returns:
        PadimLite: Initialized lightweight model ready for inference with optimal precision.

    Example:
        >>> # Load and create lightweight model with auto-precision
        >>> stats = Padim.load_statistics("model_stats.pth")
        >>> lite_model = build_padim_from_stats(stats, device="cuda")  # Auto FP16
        >>>
        >>> # Force specific precision
        >>> lite_model = build_padim_from_stats(stats, device="cuda", force_precision="fp32")
        >>>
        >>> # CPU automatically uses FP32
        >>> lite_model = build_padim_from_stats(stats, device="cpu")
    """

    target_device = torch.device(device)

    # Auto-detect precision if not forced
    if force_precision is None:
        use_fp16 = target_device.type == "cuda"
        precision_reason = f"auto-detected for {target_device.type.upper()}"
    else:
        use_fp16 = force_precision.lower() == "fp16"
        precision_reason = f"forced to {force_precision.upper()}"

    print(
        f"Building PadimLite: {'FP16' if use_fp16 else 'FP32'} precision on {target_device} ({precision_reason})"
    )

    # Extract components from stats (keep on CPU for now)
    mean = stats["mean"].float().cpu()
    cov_inv = stats["cov_inv"].float().cpu()
    ch_idx = stats["channel_indices"].to(torch.int64).cpu()
    layers = list(stats["layer_indices"])
    backbone = str(stats["backbone"])

    # Create model (precision conversion happens in __init__)
    return PadimLite(
        backbone=backbone,
        layer_indices=layers,
        channel_indices=ch_idx,
        mean=mean,
        cov_inv=cov_inv,
        device=device,
        force_precision=force_precision,
    )


def load_padim_lite(
    stats_path: str, device: str = "cpu", force_precision: Optional[str] = None
) -> PadimLite:
    """Load PadimLite directly from statistics file with device-aware precision.

    Convenience function that combines loading statistics and building the model
    in one step with automatic precision handling.

    Args:
        stats_path (str): Path to saved statistics file (.pth).
        device (str, optional): Target device. Defaults to "cpu".
        force_precision (Optional[str], optional): Force precision or auto-detect.

    Returns:
        PadimLite: Ready-to-use model with optimal precision for target device.

    Example:
        >>> # Auto-detect everything
        >>> model = load_padim_lite("stats.pth", device="cuda")
        >>>
        >>> # Force specific precision
        >>> model = load_padim_lite("stats.pth", device="cuda", force_precision="fp32")
    """
    # Import here to avoid circular imports
    from .padim import Padim

    # Load with device-aware precision (consistent with our new approach)
    stats = Padim.load_statistics(stats_path, device=device, force_fp32=None)

    return build_padim_from_stats(stats, device=device, force_precision=force_precision)
