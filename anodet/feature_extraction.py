"""
Provides classes and functions for extracting embedding vectors from neural networks.
"""

import torch
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
)
from tqdm import tqdm
from typing import List, Optional, Callable, Tuple
from torch.utils.data import DataLoader
from anodet.utils import get_logger

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
        """Construct the backbone and set appropriate mode and device

        Args:
            backbone_name: The name of the desired backbone. Must be
                one of: [resnet18, wide_resnet50].
            device: The device where to run the network.

        """

        super().__init__()

        logger.info(f"Initializing ResnetEmbeddingsExtractor with backbone: {backbone_name}, device: {device}")

        if backbone_name not in BACKBONES:
            logger.error(f"Unsupported backbone: {backbone_name}. Available backbones: {list(BACKBONES.keys())}")
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        model_func, weights = BACKBONES[backbone_name]
        logger.info(f"Loading {backbone_name} with weights: {weights}")

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
        """Perform device conversion on backone

        See pytorch docs for documentation on torch.Tensor.to

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
        """Run inference on backbone and return the embedding vectors.

        Args:
            batch: A batch of images.
            channel_indices: A list of indices with the desired channels to include in
                the embedding vectors.
            layer_hook: A function that runs on each layer of the resnet before
                concatenating them.
            layer_indices: A list of indices with the desired layers to include in the
                embedding vectors.

        Returns:
            embedding_vectors: The embedding vectors.

        """
        with torch.no_grad():
            features = []
            x = batch
            # Initial convolution layers
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            # ResNet layers
            features.append(self.backbone.layer1(x))
            features.append(self.backbone.layer2(features[-1]))
            features.append(self.backbone.layer3(features[-1]))
            features.append(self.backbone.layer4(features[-1]))

            layers = features

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
            embedding_vectors = embedding_vectors.reshape(
                batch_size, length, width * height
            )
            embedding_vectors = embedding_vectors.permute(0, 2, 1)

            # embedding_vectors = (
            #     embedding_vectors.half()
            #     if embedding_vectors.device.type != "cpu"
            #     else embedding_vectors
            # )

            return embedding_vectors, width, height

    def from_dataloader(
        self,
        dataloader: DataLoader,
        channel_indices: Optional[torch.Tensor] = None,
        layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Same as self.forward but take a dataloader instead of a tensor as argument."""

        logger.info(f"Starting feature extraction from dataloader with {len(dataloader)} batches")

        # Pre-allocate list to store embedding vectors
        embedding_vectors_list: List[torch.Tensor] = []

        for batch_idx, item in enumerate(tqdm(dataloader, "Feature extraction")):
            batch = item[0] if isinstance(item, (list, tuple)) else item

            batch = batch.to(self.device)
            if channel_indices is not None:
                channel_indices = channel_indices.to(self.device)

            # Validate input shape
            if len(batch.shape) != 4:
                logger.error(f"Invalid batch shape: expected 4D tensor (B,C,H,W), got {batch.shape}")
                raise ValueError(f"Expected 4D tensor (B,C,H,W), got {batch.shape}")

            if batch.shape[1] != 3:
                logger.error(f"Invalid number of channels: expected 3 channels (RGB), got {batch.shape[1]}")
                raise ValueError(f"Expected 3 channels (RGB), got {batch.shape[1]}")

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

        logger.info(f"Feature extraction completed. Final shape: {embedding_vectors.shape}")

        return embedding_vectors


def concatenate_layers(layers: List[torch.Tensor]) -> torch.Tensor:
    """
    Resizes all feature maps to match the spatial dimensions of the first layer,
    then concatenates them along the channel dimension.

    Args:
        layers: A list of feature tensors of shape (B, C_i, H_i, W_i)

    Returns:
        embeddings: Concatenated tensor of shape (B, sum(C_i), H, W),
                    where H and W are from the first layer.
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
            logger.error(f"Layer at index {i} has fewer than 2 dimensions: {layer.dim()}")
            raise ValueError(f"Layer at index {i} has fewer than 2 dimensions: {layer.dim()}")

    # Get target spatial size from the first layer
    target_size = layers[0].shape[-2:]

    # Resize all layers to match the target size
    resized_layers = [
        F.interpolate(l, size=target_size, mode="nearest") for l in layers
    ]

    # Concatenate once along the channel dimension
    embedding = torch.cat(resized_layers, dim=1)

    return embedding