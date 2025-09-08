"""
Provides classes and functions for working with PaDiM.
"""

import random
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .feature_extraction import ResnetEmbeddingsExtractor
from .mahalanobis import MahalanobisDistance
from .utils import pytorch_cov, split_tensor_and_run_function

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

        if self.layer_indices is None:
            self.layer_indices = [0, 1]

        self.layer_hook = layer_hook

        # Validate backbone early and set feature sizes (always define it)
        if backbone not in BACKBONE_FEATURE_SIZES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Available: {list(BACKBONE_FEATURE_SIZES.keys())}"
            )
        self.net_feature_size = BACKBONE_FEATURE_SIZES[backbone]

        # Register channel indices (use provided, or compute default)
        if channel_indices is not None:
            self.register_buffer("channel_indices", channel_indices.to(device))
        else:
            channel_indices_tensor = get_dims_indices(
                self.layer_indices, feat_dim, self.net_feature_size
            )
            self.register_buffer("channel_indices", channel_indices_tensor.to(device))

    @property
    def mean(self):
        """Get the mean tensor."""
        return self._mean

    @property
    def cov_inv(self):
        """Get the inverse covariance tensor."""
        return self._cov_inv

    def forward(
        self, x: torch.Tensor, return_map: bool = True, export: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        This method extracts feature embeddings from the input images using a pre-trained model,
        computes Mahalanobis distances to identify anomalous patches, and returns both per-image
        anomaly scores and optionally a detailed anomaly score map.

        Args:
            x (torch.Tensor): Input tensor containing batch of images.
                Expected shape: (B, C, H, W) where B=batch_size, C=channels,
                H=height, W=width.
            return_map (bool, optional): Whether to return the upsampled anomaly score map.
                If True, returns detailed spatial anomaly information upsampled to input resolution.
                If False, returns only per-image scores for memory efficiency.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - image_scores (torch.Tensor): Per-image anomaly scores of shape (B,).
                Higher values indicate higher anomaly likelihood. Computed as the maximum
                anomaly score across all patches in each image.
                - score_map (torch.Tensor or None): Detailed anomaly score map of shape (B, H, W)
                if return_map=True, otherwise None. The score map is upsampled from patch-level
                scores to match the input image resolution using bilinear interpolation.

        Example:
            >>> # Get both image scores and detailed anomaly map
            >>> image_scores, score_map = model.forward(input_images, return_map=True)
            >>>
            >>> # Get only image scores for memory efficiency
            >>> image_scores, _ = model.forward(input_images, return_map=False)

        Note:
            The method processes images in chunks of 1024 patches for memory efficiency
            when computing Mahalanobis distances.
        """
        embedding_vectors, w, h = self.embeddings_extractor(
            x,
            channel_indices=self.channel_indices,
            layer_hook=self.layer_hook,
            layer_indices=self.layer_indices,
        )
        patch_scores = self.mahalanobisDistance(
            features=embedding_vectors, width=w, height=h, export=export, chunk=256
        )  # (B, w, h)

        # fast image score directly from patch grid (no upsample needed)
        image_scores = patch_scores.flatten(1).amax(1)

        # if not return_map:
        #     return image_scores, None  # nothing large allocated

        score_map = F.interpolate(
            patch_scores.unsqueeze(1),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
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

    def fit(
        self, dataloader: torch.utils.data.DataLoader, extractions: int = 1
    ) -> None:
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
        # Run inverse function on splitted tensor to save ram memory
        cov_inv = split_tensor_and_run_function(
            func=torch.inverse, tensor=cov, split_size=1
        )

        # Register as buffers for proper model state management
        # self.register_buffer('_mean', mean)
        # self.register_buffer('_cov_inv', cov_inv)
        self.mahalanobisDistance = MahalanobisDistance(mean, cov_inv)

    def predict(
        self, batch: torch.Tensor, export: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make a prediction on test images."""
        assert (
            self.mahalanobisDistance._mean_flat is not None
            and self.mahalanobisDistance._cov_inv_flat is not None
        ), "Model is not trained. Please call `fit()` first."

        # assert self.mean is not None and self.cov_inv is not None, \
        #     "The model must be trained or provided with mean and cov_inv"
        return self(batch, export=export)  # (B), (B,H,W)

    # Optimized version with memory management
    def evaluate(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            for batch_idx, (batch, image_classifications, masks) in enumerate(
                tqdm(dataloader, desc="Inference")
            ):
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
            image_classifications_target = torch.cat(
                image_classifications_target_list, dim=0
            ).numpy()
            masks_target = (
                torch.cat(masks_target_list, dim=0).numpy().flatten().astype(np.uint8)
            )
            image_scores = torch.cat(image_scores_list, dim=0).numpy()
            score_maps = torch.cat(score_maps_list, dim=0).numpy().flatten()
        except RuntimeError as e:
            print(f"Error during tensor concatenation: {e}")
            print("Trying alternative approach...")

            # Alternative: convert to numpy first, then concatenate
            images = np.concatenate([tensor.numpy() for tensor in images_list], axis=0)
            image_classifications_target = np.concatenate(
                [tensor.numpy() for tensor in image_classifications_target_list], axis=0
            )
            masks_target = (
                np.concatenate([tensor.numpy() for tensor in masks_target_list], axis=0)
                .flatten()
                .astype(np.uint8)
            )
            image_scores = np.concatenate(
                [tensor.numpy() for tensor in image_scores_list], axis=0
            )
            score_maps = np.concatenate(
                [tensor.numpy() for tensor in score_maps_list], axis=0
            ).flatten()

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            images,
            image_classifications_target,
            masks_target,
            image_scores,
            score_maps,
        )

    # MEMORY-EFFICIENT VERSION: For very large datasets
    def evaluate_memory_efficient(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Memory-efficient version that processes data in chunks."""

        # Pre-allocate arrays if dataset size is known
        dataset_size = len(dataloader.dataset)

        # Get sample batch to determine shapes
        sample_batch, sample_class, sample_mask = next(iter(dataloader))
        sample_batch = sample_batch.to(self.device)
        sample_scores, sample_maps = self.predict(
            sample_batch[:1]
        )  # Test with one sample

        # Calculate shapes
        img_shape = sample_batch.shape[1:]  # (C, H, W)
        map_shape = sample_maps.shape[1:]  # (H, W)

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
            for batch_idx, (batch, image_classifications, masks) in enumerate(
                tqdm(dataloader, desc="Inference")
            ):
                batch = batch.to(self.device)
                batch_size = batch.shape[0]

                # Get predictions
                batch_image_scores, batch_score_maps = self.predict(batch)

                # Fill pre-allocated arrays
                end_idx = current_idx + batch_size
                images[current_idx:end_idx] = batch.cpu().numpy()
                image_classifications_target[current_idx:end_idx] = (
                    image_classifications.numpy()
                )
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

        return (
            images,
            image_classifications_target,
            masks_target,
            image_scores,
            score_maps,
        )

    def save_statistics(self, path: str) -> None:
        if (
            self.mahalanobisDistance._mean_flat is None
            or self.mahalanobisDistance._cov_inv_flat is None
        ):
            raise RuntimeError("Model is not trained. Call fit() first.")

        stats = {
            # move to CPU first, then cast to fp16 for smaller files
            "mean": self.mahalanobisDistance._mean_flat.detach()
            .cpu()
            .to(torch.float16),
            "cov_inv": self.mahalanobisDistance._cov_inv_flat.detach()
            .cpu()
            .to(torch.float16),
            "channel_indices": self.channel_indices.detach().cpu().to(torch.int32),
            "layer_indices": list(self.layer_indices),
            "backbone": self.embeddings_extractor.backbone_name,  # e.g. "resnet18"
            "model_version": "1.0",
        }
        torch.save(stats, path)

    @staticmethod
    def load_statistics(path: str, device: str = "cpu"):
        """Load stats dict and cast back to fp32 for use."""
        stats = torch.load(path, map_location="cpu", weights_only=False)
        stats["mean"] = stats["mean"].float().to(device)
        stats["cov_inv"] = stats["cov_inv"].float().to(device)
        stats["channel_indices"] = stats["channel_indices"].to(torch.int64).to(device)
        return stats


def get_dims_indices(layers, feature_dim, net_feature_size):
    random.seed(1024)
    torch.manual_seed(1024)

    total = 0
    for layer in layers:
        total += net_feature_size[layer][0]
    feature_dim = min(feature_dim, total)

    return torch.tensor(random.sample(range(0, total), feature_dim))
