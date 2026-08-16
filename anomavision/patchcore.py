"""Lightweight PatchCore anomaly detection.

This module provides a bounded-memory PatchCore implementation that follows the
public design of :mod:`anomavision.padim`: fit on a normal-image DataLoader, predict
image scores and spatial maps, and save a compact deployment artifact.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .feature_extraction import ResnetEmbeddingsExtractor


class PatchCore(torch.nn.Module):
    """Memory-bank PatchCore detector with a production-oriented footprint.

    PatchCore extracts intermediate CNN patch embeddings from normal training images,
    stores a bounded subset of those embeddings, and assigns each test patch the
    distance to its nearest normal memory-bank patch. The image score is the maximum
    patch distance; the pixel map is the patch-distance grid upsampled to the input
    resolution.

    The public methods intentionally mirror :class:`anomavision.padim.Padim`, so the
    model can be selected by the existing CLI training, inference, evaluation, and
    export workflows.

    Example:
        >>> model = PatchCore(
        ...     backbone="resnet18",
        ...     layer_indices=[0, 1],
        ...     coreset_ratio=0.1,
        ...     max_memory_patches=50000,
        ...     device=torch.device("cpu"),
        ... )
        >>> model.fit(train_loader)
        >>> image_scores, score_maps = model.predict(test_batch)

    Args:
        backbone: Feature-extraction backbone. Supported values are ``resnet18`` and
            ``wide_resnet50``.
        device: Device used for feature extraction and nearest-neighbor distance.
        layer_indices: ResNet feature stages to concatenate. Defaults to ``[0, 1]``
            to keep the lightweight model fast and compact.
        memory_bank: Optional precomputed bank with shape ``(num_patches, dim)``.
            Providing it creates a ready-to-infer model.
        coreset_ratio: Fraction of extracted normal patches to retain. Must be in
            ``(0, 1]``. Lower values reduce memory and inference time.
        max_memory_patches: Hard upper bound on retained patches. ``None`` disables
            the cap. The ultra-light default keeps at most 2,048 patches.
        patch_grid: Optional square spatial grid used to pool embeddings before the
            memory-bank search. ``14`` reduces a 224x224 ResNet stage to at most 196
            patches per image; ``None`` keeps the native feature grid.
        search_chunk_size: Number of query patches processed per nearest-neighbor
            chunk. Lower values reduce peak memory; higher values may improve GPU
            throughput.
        n_neighbors: Number of nearest neighbors. The lightweight implementation
            currently supports only ``1``.

    Raises:
        ValueError: If the coreset ratio or neighbor count is unsupported.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        device: torch.device = torch.device("cpu"),
        layer_indices: Optional[List[int]] = None,
        memory_bank: Optional[torch.Tensor] = None,
        coreset_ratio: float = 0.02,
        max_memory_patches: Optional[int] = 2048,
        patch_grid: Optional[int] = 14,
        search_chunk_size: int = 1024,
        n_neighbors: int = 1,
    ) -> None:
        super().__init__()
        if not 0 < coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in the interval (0, 1].")
        if n_neighbors != 1:
            raise ValueError("This lightweight implementation supports n_neighbors=1 only.")
        self.device = torch.device(device)
        self.backbone = backbone
        self.layer_indices = list(layer_indices or [0, 1])
        self.coreset_ratio = float(coreset_ratio)
        self.max_memory_patches = max_memory_patches
        self.patch_grid = patch_grid
        self.search_chunk_size = int(search_chunk_size)
        if self.patch_grid is not None and self.patch_grid < 1:
            raise ValueError("patch_grid must be positive or None.")
        if self.search_chunk_size < 1:
            raise ValueError("search_chunk_size must be positive.")
        self.n_neighbors = n_neighbors
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        if memory_bank is not None:
            self.register_buffer("memory_bank", memory_bank.float().to(self.device))
        else:
            self.register_buffer("memory_bank", torch.empty(0, 0, device=self.device))

    @property
    def is_fitted(self) -> bool:
        """Return whether a non-empty normal memory bank is available."""
        return self.memory_bank.ndim == 2 and self.memory_bank.shape[0] > 0

    @torch.no_grad()
    def _extract(self, batch: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Extract normalized patch embeddings for a batch.

        Args:
            batch: Input tensor with shape ``(B, C, H, W)``.

        Returns:
            A tuple ``(embeddings, width, height)``. ``embeddings`` has shape
            ``(B, width * height, feature_dim)`` and is L2-normalized per patch.

        Raises:
            RuntimeError: Propagated if the configured backbone cannot process the
                input tensor.
        """
        embeddings, width, height = self.embeddings_extractor(
            batch.to(self.device), layer_indices=self.layer_indices
        )
        embeddings = embeddings.float().reshape(batch.shape[0], width, height, -1)
        if self.patch_grid is not None and (width > self.patch_grid or height > self.patch_grid):
            embeddings = F.adaptive_avg_pool2d(
                embeddings.permute(0, 3, 1, 2), (self.patch_grid, self.patch_grid)
            ).permute(0, 2, 3, 1)
        width, height = embeddings.shape[1:3]
        return F.normalize(embeddings.reshape(batch.shape[0], width * height, -1), dim=-1), width, height

    @torch.no_grad()
    def fit(self, dataloader: torch.utils.data.DataLoader, extractions: int = 1) -> None:
        """Fit the detector from normal training images.

        The method extracts every normal patch, randomly retains the configured
        coreset, and stores it as the memory bank. No anomaly labels or gradient
        updates are required. A compact bank generally reduces both RAM/VRAM use and
        the cost of the nearest-neighbor search during inference.

        Args:
            dataloader: DataLoader yielding image tensors or ``(image, target)``
                tuples. Training images should be normal samples.
            extractions: Number of passes over the DataLoader. Values greater than
                one are useful when the loader applies random augmentations.

        Raises:
            ValueError: If the DataLoader produces no batches.

        Example:
            >>> model.fit(train_loader)
            >>> print(model.memory_bank.shape)
        """
        chunks = []
        for _ in range(extractions):
            for item in dataloader:
                batch = item[0] if isinstance(item, (tuple, list)) else item
                embeddings, _, _ = self._extract(batch)
                chunks.append(embeddings.reshape(-1, embeddings.shape[-1]).cpu())
        if not chunks:
            raise ValueError("Cannot fit PatchCore with an empty dataloader.")
        bank = torch.cat(chunks, dim=0)
        keep = max(1, int(bank.shape[0] * self.coreset_ratio))
        if self.max_memory_patches is not None:
            keep = min(keep, int(self.max_memory_patches))
        if keep < bank.shape[0]:
            indices = torch.randperm(bank.shape[0])[:keep]
            bank = bank[indices]
        self.memory_bank = bank.to(self.device)

    @torch.no_grad()
    def forward(
        self, batch: torch.Tensor, return_map: bool = True, export: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute PatchCore anomaly scores and an optional spatial score map.

        Args:
            batch: Input tensor with shape ``(B, C, H, W)`` using the same
                preprocessing as training.
            return_map: If ``True``, return a map resized to ``(H, W)``. Set to
                ``False`` when only image-level scores are needed.
            export: Retained for compatibility with PaDiM and export wrappers. The
                current distance path is already tensor-export friendly.

        Returns:
            A tuple ``(image_scores, score_map)``. ``image_scores`` has shape
            ``(B,)``. ``score_map`` has shape ``(B, H, W)`` or is ``None`` when
            ``return_map=False``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called and no memory bank was
                supplied at construction time.
        """
        if not self.is_fitted:
            raise RuntimeError("PatchCore is not fitted. Call fit() first.")
        embeddings, width, height = self._extract(batch)
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        # Embeddings are normalized, so squared cosine distance is 2 - 2 * dot.
        # Chunking avoids materializing a query-by-memory distance matrix for the
        # entire batch, which is the dominant memory cost in regular PatchCore.
        nearest_chunks = []
        for query_chunk in flat.split(self.search_chunk_size, dim=0):
            similarity = query_chunk @ self.memory_bank.transpose(0, 1)
            nearest_chunks.append((2.0 - 2.0 * similarity.amax(dim=1)).clamp_min_(0).sqrt_())
        nearest = torch.cat(nearest_chunks).reshape(batch.shape[0], width, height)
        scores = nearest.flatten(1).amax(1)
        if not return_map:
            return scores, None
        score_map = F.interpolate(
            nearest.unsqueeze(1), size=batch.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        return scores, score_map

    def predict(
        self, batch: torch.Tensor, export: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference using the standard AnomaVision prediction contract.

        Args:
            batch: Preprocessed input images with shape ``(B, C, H, W)``.
            export: Forwarded to :meth:`forward` for ONNX/TensorRT wrapper
                compatibility.

        Returns:
            ``(image_scores, score_map)`` with shapes ``(B,)`` and ``(B, H, W)``.

        Example:
            >>> scores, maps = model.predict(batch)
        """
        return self.forward(batch, return_map=True, export=export)

    def to_device(self, device: torch.device) -> None:
        """Move the extractor and memory bank to a target device.

        Args:
            device: Target PyTorch device, for example ``torch.device("cuda")`` or
                ``torch.device("cpu")``.
        """
        self.device = torch.device(device)
        self.embeddings_extractor.to_device(self.device)
        self.memory_bank = self.memory_bank.to(self.device)

    def save_statistics(self, path: str, half: Optional[bool] = False) -> None:
        """Save a compact PatchCore deployment artifact.

        The artifact contains the memory bank and the feature-extraction settings,
        but not a duplicate copy of the fitted training loop. It can be loaded with
        :func:`build_patchcore_from_stats` or through the standard PyTorch backend.

        Args:
            path: Destination ``.pth`` path.
            half: If ``True``, store the memory bank in FP16 to reduce file size.
                Defaults to FP32 for CPU-safe numerical behavior.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("PatchCore is not fitted. Call fit() first.")
        bank = self.memory_bank.detach().cpu()
        if half:
            bank = bank.half()
        torch.save(
            {
                "memory_bank": bank,
                "backbone": self.backbone,
                "layer_indices": self.layer_indices,
                "coreset_ratio": self.coreset_ratio,
                "max_memory_patches": self.max_memory_patches,
                "patch_grid": self.patch_grid,
                "search_chunk_size": self.search_chunk_size,
                "model_type": "patchcore",
                "dtype": "fp16" if half else "fp32",
            },
            path,
        )


def build_patchcore_from_stats(
    stats: Dict, device: str = "cpu", force_precision: Optional[str] = None
) -> PatchCore:
    """Build a ready-to-infer PatchCore from a compact statistics artifact.

    Args:
        stats: Dictionary created by :meth:`PatchCore.save_statistics`. It must
            contain ``memory_bank``, ``backbone``, and ``layer_indices``.
        device: Target device string such as ``"cpu"`` or ``"cuda"``.
        force_precision: Optional precision override. ``"fp16"`` is applied only
            when the target device is CUDA; CPU inference remains FP32.

    Returns:
        A fitted :class:`PatchCore` instance ready for :meth:`PatchCore.predict`.

    Raises:
        KeyError: If a required statistics key is missing.

    Example:
        >>> stats = torch.load("patchcore.pth", weights_only=False)
        >>> model = build_patchcore_from_stats(stats, device="cuda")
    """
    bank = stats["memory_bank"].float().cpu()
    model = PatchCore(
        backbone=str(stats["backbone"]),
        layer_indices=list(stats["layer_indices"]),
        memory_bank=bank,
        coreset_ratio=float(stats.get("coreset_ratio", 0.02)),
        max_memory_patches=stats.get("max_memory_patches", 2048),
        patch_grid=stats.get("patch_grid", 14),
        search_chunk_size=int(stats.get("search_chunk_size", 1024)),
        device=torch.device(device),
    )
    if force_precision == "fp16" and model.device.type == "cuda":
        model = model.half()
    return model
