"""
Provides classes and functions for working with PaDiM.
"""

import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
from typing import Optional, Callable, List, Tuple
from .feature_extraction import ResnetEmbeddingsExtractor
from .utils import pytorch_cov, mahalanobis, split_tensor_and_run_function
from collections import OrderedDict

from .mahalanobis import MahalanobisDistance

class Padim(torch.nn.Module):
    """A padim model with functions to train and perform inference."""

    def __init__(self, backbone: str = 'resnet18',
                 device: torch.device = torch.device('cpu'),
                 channel_indices: Optional[torch.Tensor] = None,
                 layer_indices: Optional[List[int]] = None,
                 layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 feat_dim: Optional[int] = 50) -> None:

        """Construct the model and initialize the attributes

        Args:
            backbone: The name of the desired backbone. Must be one of: [resnet18, wide_resnet50].
            device: The device where to run the model.
            channel_indices: A tensor with the desired channel indices to extract \
                from the backbone, with size (D).
            layer_indices: A list with the desired layers to extract from the backbone, \
            allowed indices are 1, 2, 3 and 4.
            layer_hook: A function that can modify the layers during extraction.
        """

        super(Padim, self).__init__()
        
        self.device = device
        # Register as a submodule for proper ONNX export
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        self.layer_indices = layer_indices
        
        print("******************************************************************", self.layer_indices)
        if self.layer_indices is None:
            self.layer_indices = [0,1]

        self.layer_hook = layer_hook
        self.to_device(self.device)
        # Register channel_indices as a buffer for ONNX compatibility
        if channel_indices is not None:
            self.register_buffer('channel_indices', channel_indices)
        else:
            if backbone == 'resnet18':
                self.net_feature_size = OrderedDict(
                    [(0, [64]), (1, [128]), (2, [256]), (3, [512])])
                
                
            elif backbone == 'wide_resnet50':
                self.net_feature_size = OrderedDict(
                    [(0, [255]), (1, [512]), (2, [1024]), (3, [2048])]
                )                
                                            
            self.register_buffer(
                "channel_indices",
                get_dims_indices(self.layer_indices, feat_dim, self.net_feature_size),
            )

    @property
    def mean(self):
        """Get the mean tensor."""
        return self._mean
    
    @property
    def cov_inv(self):
        """Get the inverse covariance tensor."""
        return self._cov_inv

    def forward(self, x: torch.Tensor,gaussian_blur: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ONNX compatibility.

        Args:
            x: A batch of input images, with dimension (B, C, H, W).

        Returns:
            image_scores: A tensor with the image level scores, with dimension (B).
            score_map: A tensor with the patch level scores, with dimension (B, H, W)
        """
        # Extract features using the embeddings extractor

        assert self.mahalanobisDistance._mean_flat is not None and self.mahalanobisDistance._cov_inv_flat is not None, \
            "Model is not trained. Please call `fit()` first."
        
        embedding_vectors ,width, height= self.embeddings_extractor(x,
                                                      channel_indices=self.channel_indices,
                                                      layer_hook=self.layer_hook,
                                                      layer_indices=self.layer_indices
                                                      )

        # Calculate Mahalanobis distance
        # patch_scores = mahalanobis(self.mean, self.cov_inv, embedding_vectors)
        patch_scores = self.mahalanobisDistance(embedding_vectors,width, height)

        # Reshape to square patches - use a more ONNX-friendly approach
        batch_size = x.shape[0]
        num_patches = embedding_vectors.shape[1]
        patch_width = int(torch.sqrt(torch.tensor(num_patches, dtype=torch.float32)).item())
        # patch_width = int(torch.sqrt(num_patches.float()).item())

        patch_scores = patch_scores.view(batch_size, patch_width, patch_width)

        # Interpolate to original image size
        score_map = F.interpolate(patch_scores.unsqueeze(1), 
                                  size=(x.shape[2], x.shape[3]),
                                  mode='bilinear', 
                                  align_corners=False)
        
        # Remove the channel dimension
        score_map = score_map.squeeze(1)

        # Apply gaussian blur - create the blur operation inline for ONNX
        # Using a simpler approach that's more ONNX-friendly
        if gaussian_blur:
            score_map = T.GaussianBlur(33, sigma=4)(score_map)     

        # Calculate image-level scores
        image_scores = torch.max(score_map.view(batch_size, -1), dim=1)[0]

        return image_scores, score_map

    def to_device(self, device: torch.device) -> None:
        """Perform device conversion on backone, mean, cov_inv and channel_indices

        Args:
            device: The device where to run the model.

        """

        self.device = device
        if self.embeddings_extractor is not None:
            self.embeddings_extractor.to_device(device)
        # Buffers are automatically moved with the module, so no need to manually move them
    def fit(self, dataloader: torch.utils.data.DataLoader, extractions: int = 1) -> None:
        """Fit the model (i.e. mean and cov_inv) to data.

        Args:
            dataloader: A pytorch dataloader, with sample dimensions (B, D, H, W), \
                containing normal images.
            extractions: Number of extractions from dataloader. Could be of interest \
                when applying random augmentations.

        """
        embedding_vectors = None
        for i in range(extractions):
            extracted_embedding_vectors = self.embeddings_extractor.from_dataloader(
                dataloader,
                channel_indices=self.channel_indices,
                layer_hook=self.layer_hook,
                layer_indices=self.layer_indices
            )
            if embedding_vectors is None:
                embedding_vectors = extracted_embedding_vectors
            else:
                embedding_vectors = torch.cat((embedding_vectors, extracted_embedding_vectors), 0)

        mean = torch.mean(embedding_vectors, dim=0)
        cov = pytorch_cov(embedding_vectors.permute(1, 0, 2), rowvar=False) \
            + 0.01 * torch.eye(embedding_vectors.shape[2])
        # Run inverse function on splitted tensor to save ram memory
        cov_inv = split_tensor_and_run_function(func=torch.inverse,
                                               tensor=cov,
                                               split_size=1)
        
        # Register as buffers for proper model state management
        # self.register_buffer('_mean', mean)
        # self.register_buffer('_cov_inv', cov_inv)
        self.mahalanobisDistance = MahalanobisDistance(mean, cov_inv)


    def predict(self, batch: torch.Tensor, gaussian_blur: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make a prediction on test images."""
        assert self.mahalanobisDistance._mean_flat is not None and self.mahalanobisDistance._cov_inv_flat is not None, \
            "Model is not trained. Please call `fit()` first."
        
        # assert self.mean is not None and self.cov_inv is not None, \
        #     "The model must be trained or provided with mean and cov_inv"
        return self(batch, gaussian_blur=gaussian_blur)

    # Optimized version with memory management
    def evaluate(self, dataloader: torch.utils.data.DataLoader) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run predict on all images in a dataloader and return the results.

        Args:
            dataloader: A pytorch dataloader, with sample dimensions (B, D, H, W), \
                containing normal images.

        Returns:
            images: An array containing all input images.
            image_classifications_target: An array containing the target \
                classifications on image level.
            masks_target: An array containing the target classifications on patch level.
            image_scores: An array containing the predicted scores on image level.
            score_maps: An array containing the predicted scores on patch level.

        """
        images_list = []
        image_classifications_target_list = []
        masks_target_list = []
        image_scores_list = []
        score_maps_list = []

        # Set model to evaluation mode
        self.eval()

        with torch.no_grad():
            for batch_idx, (batch, image_classifications, masks) in enumerate(tqdm(dataloader, desc='Inference')):
                # Move batch to device if needed
                batch = batch.to(self.device)
                
                # Get predictions
                batch_image_scores, batch_score_maps = self.predict(batch)

                # Append to lists (move to CPU to save GPU memory)
                images_list.append(batch.cpu())
                image_classifications_target_list.append(image_classifications)
                masks_target_list.append(masks)
                image_scores_list.append(batch_image_scores.cpu())
                score_maps_list.append(batch_score_maps.cpu())

                # Clear GPU cache periodically to prevent OOM
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Concatenate all tensors efficiently
        try:
            images = torch.cat(images_list, dim=0).numpy()
            image_classifications_target = torch.cat(image_classifications_target_list, dim=0).numpy()
            masks_target = torch.cat(masks_target_list, dim=0).numpy().flatten().astype(np.uint8)
            image_scores = torch.cat(image_scores_list, dim=0).numpy()
            score_maps = torch.cat(score_maps_list, dim=0).numpy().flatten()
        except RuntimeError as e:
            print(f"Error during tensor concatenation: {e}")
            print("Trying alternative approach...")
            
            # Alternative: convert to numpy first, then concatenate
            images = np.concatenate([tensor.numpy() for tensor in images_list], axis=0)
            image_classifications_target = np.concatenate([tensor.numpy() for tensor in image_classifications_target_list], axis=0)
            masks_target = np.concatenate([tensor.numpy() for tensor in masks_target_list], axis=0).flatten().astype(np.uint8)
            image_scores = np.concatenate([tensor.numpy() for tensor in image_scores_list], axis=0)
            score_maps = np.concatenate([tensor.numpy() for tensor in score_maps_list], axis=0).flatten()

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return images, image_classifications_target, masks_target, image_scores, score_maps


    # MEMORY-EFFICIENT VERSION: For very large datasets
    def evaluate_memory_efficient(self, dataloader: torch.utils.data.DataLoader) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """Memory-efficient version that processes data in chunks."""

        # Pre-allocate arrays if dataset size is known
        dataset_size = len(dataloader.dataset)
        
        # Get sample batch to determine shapes
        sample_batch, sample_class, sample_mask = next(iter(dataloader))
        sample_batch = sample_batch.to(self.device)
        sample_scores, sample_maps = self.predict(sample_batch[:1])  # Test with one sample
        
        # Calculate shapes
        img_shape = sample_batch.shape[1:]  # (C, H, W)
        map_shape = sample_maps.shape[1:]   # (H, W)
        
        # Pre-allocate numpy arrays
        images = np.zeros((dataset_size, *img_shape), dtype=np.float32)
        image_classifications_target = np.zeros(dataset_size, dtype=np.int64)
        masks_target = np.zeros((dataset_size, *sample_mask.shape[1:]), dtype=np.uint8)
        image_scores = np.zeros(dataset_size, dtype=np.float32)
        score_maps = np.zeros((dataset_size, *map_shape), dtype=np.float32)
        
        # Set model to evaluation mode
        self.eval()
        
        current_idx = 0
        
        with torch.no_grad():
            for batch_idx, (batch, image_classifications, masks) in enumerate(tqdm(dataloader, desc='Inference')):
                batch = batch.to(self.device)
                batch_size = batch.shape[0]
                
                # Get predictions
                batch_image_scores, batch_score_maps = self.predict(batch)
                
                # Fill pre-allocated arrays
                end_idx = current_idx + batch_size
                images[current_idx:end_idx] = batch.cpu().numpy()
                image_classifications_target[current_idx:end_idx] = image_classifications.numpy()
                masks_target[current_idx:end_idx] = masks.numpy()
                image_scores[current_idx:end_idx] = batch_image_scores.cpu().numpy()
                score_maps[current_idx:end_idx] = batch_score_maps.cpu().numpy()
                
                current_idx = end_idx
                
                # Clear GPU cache periodically
                if batch_idx % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Flatten masks and score_maps as expected by the original function
        masks_target = masks_target.flatten()
        score_maps = score_maps.flatten()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return images, image_classifications_target, masks_target, image_scores, score_maps


def get_dims_indices(layers, feature_dim, net_feature_size):
    random.seed(1024)
    torch.manual_seed(1024)

    total = 0
    for layer in layers:
        total += net_feature_size[layer][0]
    feature_dim = min(feature_dim, total)

    return torch.tensor(random.sample(range(0, total), feature_dim))
