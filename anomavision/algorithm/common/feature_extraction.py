"""
Provides classes and functions for extracting embedding vectors from neural networks.
"""

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)
from tqdm import tqdm

from anomavision.utils import get_logger

logger = get_logger(__name__)

BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "wide_resnet50": (wide_resnet50_2, Wide_ResNet50_2_Weights.DEFAULT),
}


class ResnetEmbeddingsExtractor(torch.nn.Module):
    """A class to hold, and extract embedding vectors from, a resnet.

    Attributes:
        backbone: The resnet from which to extract embedding vectors.

    """

    def __init__(self, backbone_name: str, device: torch.device) -> None:
        """Initialize ResNet embeddings extractor with specified backbone and device.

        Creates a ResNet-based feature extractor for anomaly detection pipelines.
        The backbone is loaded with pre-trained weights and configured for inference
        in evaluation mode.

        Args:
            backbone_name (str): Name of the ResNet architecture to use.
                Must be one of: ["resnet18", "wide_resnet50"].
            device (torch.device): Target device (CPU/GPU) for model computation.

        Raises:
            ValueError: If backbone_name is not supported.

        Example:
            >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            >>> extractor = ResnetEmbeddingsExtractor("resnet18", device)
        """

        super().__init__()

        logger.info(
            f"Initializing ResnetEmbeddingsExtractor with backbone: {backbone_name}, device: {device}"
        )

        if backbone_name not in BACKBONES:
            logger.error(
                f"Unsupported backbone: {backbone_name}. Available backbones: {list(BACKBONES.keys())}"
            )
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        model_func, weights = BACKBONES[backbone_name]
        logger.info(f"Loading {backbone_name} with weights: {weights}")
        self.backbone_name = backbone_name

        self.backbone = model_func(weights=weights, progress=True)
        self.device = device
        self.backbone.to(self.device)

        backbone_device = next(self.backbone.parameters()).device
        logger.info(f"Backbone successfully moved to device: {backbone_device}")
        print("Backbone device:", backbone_device)

        self.backbone.eval()
        self.eval()
        logger.info("Model set to evaluation mode")

    def to_device(self, device: torch.device) -> None:
        """Move the backbone model to specified device.

        Transfers the ResNet backbone and updates internal device reference.
        Useful for switching between CPU and GPU during different phases of
        model training or inference.

        Args:
            device (torch.device): Target device for the backbone model.

        Example:
            >>> extractor.to_device(torch.device("cuda"))
        """
        logger.info(f"Moving backbone to device: {device}")
        self.backbone.to(device)
        self.device = device

    def forward(
        self,
        batch: torch.Tensor,
        channel_indices: Optional[torch.Tensor] = None,
        layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Extract multi-scale feature embeddings from ResNet backbone.

        Performs forward pass through ResNet layers, extracts features from specified
        layers, optionally applies transformations, and concatenates them into unified
        embedding vectors suitable for anomaly detection.

        Args:
            batch (torch.Tensor): Input batch of images with shape (B, C, H, W).
            channel_indices (Optional[torch.Tensor]): Indices of channels to select
                from concatenated features. If None, uses all channels.
            layer_hook (Optional[Callable]): Function to apply to each layer's features
                before concatenation. Useful for normalization or transformation.
            layer_indices (Optional[List[int]]): Layer indices to extract features from.
                Valid indices are [0, 1, 2, 3] corresponding to ResNet layers.
                If None, defaults to [0, 1, 2, 3].

        Returns:
            Tuple[torch.Tensor, int, int]: A tuple containing:
                - embedding_vectors (torch.Tensor): Extracted features of shape
                (B, W*H, D) where D is the feature dimension.
                - width (int): Spatial width of the feature maps.
                - height (int): Spatial height of the feature maps.

        Example:
            >>> batch = torch.randn(4, 3, 224, 224)
            >>> embeddings, w, h = extractor(batch, layer_indices=[0, 1])
            >>> print(f"Shape: {embeddings.shape}, Spatial: {w}x{h}")
        """
        with torch.no_grad():
            x = batch
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            want = set(layer_indices or [0, 1, 2, 3])
            max_l = max(want)

            layers = []
            out1 = self.backbone.layer1(x)
            if 0 in want:
                layers.append(out1)
            if max_l >= 1:
                out2 = self.backbone.layer2(out1)
                if 1 in want:
                    layers.append(out2)
            if max_l >= 2:
                out3 = self.backbone.layer3(out2 if max_l >= 1 else out1)
                if 2 in want:
                    layers.append(out3)
            if max_l >= 3:
                out4 = self.backbone.layer4(
                    out3 if max_l >= 2 else (out2 if max_l >= 1 else out1)
                )
                if 3 in want:
                    layers.append(out4)

            if layer_indices is not None:
                layers = [layers[i] for i in layer_indices]

            if layer_hook is not None:
                layers = [layer_hook(layer) for layer in layers]

            embedding_vectors = concatenate_layers(layers)

            if channel_indices is not None:
                channel_indices = channel_indices.to(embedding_vectors.device)
                embedding_vectors = torch.index_select(
                    embedding_vectors, 1, channel_indices
                )

            batch_size, length, width, height = embedding_vectors.shape

            embedding_vectors = (
                embedding_vectors.reshape(batch_size, length, width * height)
                .permute(0, 2, 1)
                .contiguous()
            )

            return embedding_vectors, width, height

    def from_dataloader(
        self,
        dataloader: DataLoader,
        channel_indices: Optional[torch.Tensor] = None,
        layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Extract features from all batches in a DataLoader.

        Processes entire dataset through feature extraction pipeline with memory
        management optimizations. Accumulates features from all batches and returns
        concatenated result.

        Args:
            dataloader (DataLoader): PyTorch DataLoader containing image batches.
                Each batch should have shape (B, C, H, W).
            channel_indices (Optional[torch.Tensor]): Channel selection indices.
            layer_hook (Optional[Callable]): Function to apply to layer features.
            layer_indices (Optional[List[int]]): Layers to extract from.

        Returns:
            torch.Tensor: Concatenated embedding vectors from all batches with
                shape (N, W*H, D) where N is total number of images.

        Raises:
            ValueError: If batch has invalid shape (not 4D or wrong channel count).

        Example:
            >>> train_loader = DataLoader(train_dataset, batch_size=32)
            >>> features = extractor.from_dataloader(train_loader)
            >>> print(f"Extracted features shape: {features.shape}")
        """

        logger.info(
            f"Starting feature extraction from dataloader with {len(dataloader)} batches"
        )

        # Pre-allocate list to store embedding vectors
        embedding_vectors_list: List[torch.Tensor] = []

        for batch_idx, item in enumerate(tqdm(dataloader, "Feature extraction")):
            batch = item[0] if isinstance(item, (list, tuple)) else item

            batch = batch.to(self.device)
            if channel_indices is not None:
                channel_indices = channel_indices.to(self.device)

            # Validate input shape (B, C, H, W)
            if len(batch.shape) != 4:
                logger.error(
                    f"Invalid batch shape: expected 4D tensor (B,C,H,W), got {batch.shape}"
                )
                raise ValueError(f"Expected 4D tensor (B,C,H,W), got {batch.shape}")

            if batch.shape[1] != 3:
                logger.error(
                    f"Invalid number of channels: expected 3 channels (RGB), got {batch.shape[1]}"
                )
                raise ValueError(f"Expected 3 channels (RGB), got {batch.shape[1]}")

            logger.debug(f"Processing batch {batch_idx + 1}: input shape {batch.shape}")
            batch_embedding_vectors, width, height = self(
                batch,
                channel_indices=channel_indices,
                layer_hook=layer_hook,
                layer_indices=layer_indices,
            )

            # Move to CPU and detach to prevent GPU memory accumulation
            batch_embedding_vectors = batch_embedding_vectors.detach().cpu()
            embedding_vectors_list.append(batch_embedding_vectors)

            # Clear GPU cache periodically to prevent memory buildup
            if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

        # Concatenate all tensors at once (more memory efficient than incremental concat)
        embedding_vectors = torch.cat(embedding_vectors_list, dim=0)

        logger.info(
            f"Feature extraction completed. Final shape: {embedding_vectors.shape}"
        )

        return embedding_vectors


def concatenate_layers(layers: List[torch.Tensor]) -> torch.Tensor:
    """
    Resizes all feature maps to match the spatial dimensions of the first layer,
    then concatenates them along the channel dimension.

    This function is essential for multi-scale feature fusion in anomaly detection
    pipelines. It ensures all layers have consistent spatial dimensions before
    concatenation, enabling effective combination of features from different
    network depths.

    Args:
        layers (List[torch.Tensor]): A list of feature tensors of shape (B, C_i, H_i, W_i)
            where B is batch size, C_i is the number of channels for layer i,
            and H_i, W_i are the spatial dimensions for layer i.

    Returns:
        torch.Tensor: Concatenated tensor of shape (B, sum(C_i), H, W),
            where H and W are the spatial dimensions from the first layer,
            and sum(C_i) is the total channels across all input layers.

    Raises:
        ValueError: If the input list of layers is empty or if any layer has
            fewer than 2 dimensions.
        TypeError: If any element in the layers list is not a torch.Tensor.

    Example:
        >>> layer1 = torch.randn(2, 64, 56, 56)   # From early ResNet layer
        >>> layer2 = torch.randn(2, 128, 28, 28)  # From deeper ResNet layer
        >>> combined = concatenate_layers([layer1, layer2])
        >>> print(combined.shape)  # torch.Size([2, 192, 56, 56])

    Note:
        - All layers are resized to match the spatial dimensions of the first layer
        - Uses nearest neighbor interpolation for resizing
        - The order of layers in the input list determines their order in concatenation
    """

    if not layers:
        logger.error("Empty list of layers provided to concatenate_layers")
        raise ValueError("The input list of layers is empty.")

    # Validate that all layers are torch.Tensors and have at least 2 spatial dimensions
    for i, layer in enumerate(layers):
        if not isinstance(layer, torch.Tensor):
            logger.error(f"Layer at index {i} is not a torch.Tensor: {type(layer)}")
            raise TypeError(f"Layer at index {i} is not a torch.Tensor: {type(layer)}")
        if layer.dim() < 2:
            logger.error(
                f"Layer at index {i} has fewer than 2 dimensions: {layer.dim()}"
            )
            raise ValueError(
                f"Layer at index {i} has fewer than 2 dimensions: {layer.dim()}"
            )

    # Get target spatial size from the first layer
    target_size = layers[0].shape[-2:]
    logger.debug(f"Target size for layer concatenation: {target_size}")

    # Resize all layers to match the target size
    resized_layers = [
        F.interpolate(lyr, size=target_size, mode="nearest") for lyr in layers
    ]

    # Concatenate once along the channel dimension
    embedding = torch.cat(resized_layers, dim=1)
    logger.debug(f"Concatenated embedding shape: {embedding.shape}")

    return embedding
