"""
Provides classes and functions for extracting embedding vectors from neural networks.
"""

import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, wide_resnet50_2, Wide_ResNet50_2_Weights
from tqdm import tqdm
from typing import List, Optional, Callable, cast
from torch.utils.data import DataLoader


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
        assert backbone_name in ['resnet18', 'wide_resnet50']

        if backbone_name == 'resnet18':
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT, progress=True)
        elif backbone_name == 'wide_resnet50':
            self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT, progress=True)

        self.backbone.to(device)
        self.backbone.eval()
        self.eval()

    def to_device(self, device: torch.device) -> None:
        """Perform device conversion on backone

        See pytorch docs for documentation on torch.Tensor.to

        """
        self.backbone.to(device)

    def forward(self,
                batch: torch.Tensor,
                channel_indices: Optional[torch.Tensor] = None,
                layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                layer_indices: Optional[List[int]] = None
                ) -> torch.Tensor:
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
            batch = self.backbone.conv1(batch)
            batch = self.backbone.bn1(batch)
            batch = self.backbone.relu(batch)
            batch = self.backbone.maxpool(batch)
            layer1 = self.backbone.layer1(batch)
            layer2 = self.backbone.layer2(layer1)
            layer3 = self.backbone.layer3(layer2)
            layer4 = self.backbone.layer4(layer3)
            layers = [layer1, layer2, layer3, layer4]

            if layer_indices is not None:
                layers = [layers[i] for i in layer_indices]

            if layer_hook is not None:
                layers = [layer_hook(layer) for layer in layers]

            embedding_vectors = concatenate_layers(layers)

            if channel_indices is not None:
                embedding_vectors = torch.index_select(embedding_vectors, 1, channel_indices)

            batch_size, length, width, height = embedding_vectors.shape
            embedding_vectors = embedding_vectors.reshape(batch_size, length, width*height)
            embedding_vectors = embedding_vectors.permute(0, 2, 1)
            
            embedding_vectors = (
                embedding_vectors.half()
                if embedding_vectors.device.type != "cpu"
                else embedding_vectors
            )
            
            return embedding_vectors,width, height

    def from_dataloader(self,
                        dataloader: DataLoader,
                        channel_indices: Optional[torch.Tensor] = None,
                        layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                        layer_indices: Optional[List[int]] = None
                        ) -> torch.Tensor:
        """Same as self.forward but take a dataloader instead of a tensor as argument."""

        # Pre-allocate list to store embedding vectors
        embedding_vectors_list: List[torch.Tensor] = []
        
        for (batch, _, _) in tqdm(dataloader, 'Feature extraction'):
            batch_embedding_vectors,_,_ = self(batch,
                                        channel_indices=channel_indices,
                                        layer_hook=layer_hook,
                                        layer_indices=layer_indices)
            
            # Move to CPU and detach to prevent GPU memory accumulation
            batch_embedding_vectors = batch_embedding_vectors.detach().cpu()
            embedding_vectors_list.append(batch_embedding_vectors)
            
            # Clear GPU cache periodically to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all tensors at once (more memory efficient than incremental concat)
        embedding_vectors = torch.cat(embedding_vectors_list, dim=0)
                
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
        raise ValueError("The input list of layers is empty.")

    # Get target spatial size from the first layer
    target_size = layers[0].shape[-2:]

    # Resize all layers to match the target size
    resized_layers = [F.interpolate(l, size=target_size, mode='nearest') for l in layers]

    # Concatenate once along the channel dimension
    embedding = torch.cat(resized_layers, dim=1)

    return embedding




