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
        """Initialize PaDiM anomaly detection model.

        Creates a PaDiM (Patch Distribution Modeling) model for image anomaly detection.
        The model uses ResNet backbone for feature extraction and Gaussian modeling
        for anomaly scoring.

        Args:
            backbone (str, optional): ResNet architecture name. Must be one of
                ["resnet18", "wide_resnet50"]. Defaults to "resnet18".
            device (torch.device, optional): Computation device. Defaults to CPU.
            channel_indices (Optional[torch.Tensor]): Specific channel indices to use
                for feature selection. If None, randomly samples feat_dim channels.
            layer_indices (Optional[List[int]]): ResNet layers to extract features from.
                Valid indices are [0, 1, 2, 3]. Defaults to [0, 1].
            layer_hook (Optional[Callable]): Function to apply to extracted features.
            feat_dim (Optional[int]): Target feature dimension after channel selection.
                Defaults to 50.

        Raises:
            ValueError: If backbone is not supported.

        Example:
            >>> model = Padim(backbone="resnet18", device=torch.device("cuda"))
            >>> model = Padim(layer_indices=[0, 1, 2], feat_dim=100)
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
        """Perform anomaly detection inference on input images.

        Extracts features from input images, computes Mahalanobis distances to identify
        anomalous patches, and returns both per-image anomaly scores and optionally
        detailed spatial anomaly maps.

        Args:
            x (torch.Tensor): Input batch of images with shape (B, C, H, W).
            return_map (bool, optional): Whether to return upsampled anomaly score map.
                If True, returns detailed spatial anomaly information. If False,
                returns only per-image scores for memory efficiency. Defaults to True.
            export (bool, optional): If True, uses export-friendly computation paths
                for ONNX compatibility. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - image_scores (torch.Tensor): Per-image anomaly scores of shape (B,).
                Higher values indicate higher anomaly likelihood.
                - score_map (Optional[torch.Tensor]): Spatial anomaly map of shape
                (B, H, W) if return_map=True, otherwise None. Upsampled to match
                input resolution using bilinear interpolation.

        Example:
            >>> images = torch.randn(4, 3, 224, 224)
            >>> image_scores, score_map = model(images, return_map=True)
            >>> print(f"Image scores: {image_scores.shape}")  # torch.Size([4])
            >>> print(f"Score map: {score_map.shape}")        # torch.Size([4, 224, 224])
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
        """Fit the PaDiM model to normal training data.

        Extracts features from training data and computes Gaussian statistics
        (mean and covariance) for each spatial location. These statistics are
        used during inference for anomaly scoring.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing normal
                (non-anomalous) training images.
            extractions (int, optional): Number of passes through the dataloader.
                Useful when applying random augmentations to increase data diversity.
                Defaults to 1.

        Note:
            After fitting, the model stores mean vectors and inverse covariance
            matrices as MahalanobisDistance module for efficient inference.

        Example:
            >>> train_loader = DataLoader(normal_dataset, batch_size=32)
            >>> model.fit(train_loader, extractions=2)  # With augmentation
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
        """Make anomaly predictions on test images.

        Performs inference on a batch of test images, returning both image-level
        anomaly scores and pixel-level anomaly maps.

        Args:
            batch (torch.Tensor): Test images with shape (B, C, H, W).
            export (bool, optional): Use export-friendly computation for ONNX.
                Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - image_scores (torch.Tensor): Per-image anomaly scores of shape (B,).
                - score_maps (torch.Tensor): Pixel-level anomaly maps of shape (B, H, W).

        Raises:
            AssertionError: If model has not been trained (no mean/covariance available).

        Example:
            >>> test_batch = torch.randn(8, 3, 224, 224)
            >>> img_scores, score_maps = model.predict(test_batch)
        """

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

    def save_statistics(self, path: str, half: bool = False) -> None:
        """Save trained model statistics to disk.

        Saves the learned Gaussian parameters (mean, inverse covariance) and model
        configuration to a file for later loading and inference.

        Args:
            path (str): File path where to save the statistics.
            half (bool, optional): If True, saves tensors in FP16 precision for
                smaller file size while maintaining compatibility. Defaults to False.

        Raises:
            RuntimeError: If model has not been trained yet.

        Example:
            >>> model.save_statistics("padim_stats.pth", half=True)  # Compact storage
            >>> model.save_statistics("padim_stats_full.pth")       # Full precision
        """

        if (
            self.mahalanobisDistance._mean_flat is None
            or self.mahalanobisDistance._cov_inv_flat is None
        ):
            raise RuntimeError("Model is not trained. Call fit() first.")

        # Get tensors and move to CPU first
        mean_tensor = self.mahalanobisDistance._mean_flat.detach().cpu()
        cov_inv_tensor = self.mahalanobisDistance._cov_inv_flat.detach().cpu()
        channel_indices_tensor = self.channel_indices.detach().cpu()

        # Apply precision conversion if requested
        if half:
            # Convert to FP16 for storage (works on both CPU/GPU tensors after moving to CPU)
            mean_tensor = mean_tensor.half()
            cov_inv_tensor = cov_inv_tensor.half()
            # Keep indices as int64 (no need for FP16)
            dtype_info = "fp16"
        else:
            # Keep as FP32
            mean_tensor = mean_tensor.float()
            cov_inv_tensor = cov_inv_tensor.float()
            dtype_info = "fp32"

        stats = {
            "mean": mean_tensor,
            "cov_inv": cov_inv_tensor,
            "channel_indices": channel_indices_tensor,
            "layer_indices": list(self.layer_indices),
            "backbone": self.embeddings_extractor.backbone_name,
            "model_version": "1.0",
            "dtype": dtype_info,  # Track what precision was used
        }

        torch.save(stats, path)

    @staticmethod
    def load_statistics(path: str, device: str = "cpu", force_fp32: bool = True):
        """Load model statistics from disk.

        Loads previously saved Gaussian statistics and model configuration,
        with automatic precision handling for optimal inference performance.

        Args:
            path (str): Path to the saved statistics file.
            device (str, optional): Target device for loaded tensors. Defaults to "cpu".
            force_fp32 (bool, optional): If True, converts to FP32 regardless of
                saved precision for numerical stability. Defaults to True.

        Returns:
            dict: Statistics dictionary with tensors moved to specified device.

        Example:
            >>> stats = Padim.load_statistics("padim_stats.pth", device="cuda")
            >>> # Use stats to initialize model or create PadimLite
        """

        stats = torch.load(path, map_location="cpu", weights_only=False)

        # Get saved precision info (default to fp32 for backward compatibility)
        saved_dtype = stats.get("dtype", "fp32")

        # Always convert to computation precision (FP32 recommended for stability)
        if force_fp32 or saved_dtype == "fp16":
            stats["mean"] = stats["mean"].float().to(device)
            stats["cov_inv"] = stats["cov_inv"].float().to(device)
        else:
            stats["mean"] = stats["mean"].to(device)
            stats["cov_inv"] = stats["cov_inv"].to(device)

        # Handle indices (always int64)
        stats["channel_indices"] = stats["channel_indices"].to(torch.int64).to(device)

        print(f"Statistics loaded from {path}")
        print(
            f"Saved as: {saved_dtype}, loaded as: {'fp32' if force_fp32 else saved_dtype}"
        )

        return stats


def get_dims_indices(layers, feature_dim, net_feature_size):
    """
    Generate random channel indices for feature dimensionality reduction.

    This function implements random channel selection for PaDiM models,
    reducing computational complexity while maintaining anomaly detection
    performance. It ensures reproducible selection through fixed random seeds.

    Args:
        layers (List[int]): List of layer indices to extract features from
            (e.g., [0, 1] for first two ResNet layers).
        feature_dim (int): Target number of feature dimensions after reduction.
            Will be clamped to total available features if larger.
        net_feature_size (Dict[int, List[int]]): Mapping from layer index to
            list of channel counts for that layer.

    Returns:
        torch.Tensor: Tensor of randomly selected channel indices of length
            min(feature_dim, total_channels). Indices are in range [0, total_channels).

    Example:
        >>> # For ResNet18 layers [0,1] with feature_dim=100
        >>> indices = get_dims_indices([0, 1], 100, BACKBONE_FEATURE_SIZES["resnet18"])
        >>> print(indices.shape)  # torch.Size([100]) or smaller if fewer channels available

    Note:
        - Uses fixed random seed (1024) for reproducible results
        - Automatically clamps feature_dim to available channels
        - Sampling is without replacement (no duplicate indices)
    """

    random.seed(1024)
    torch.manual_seed(1024)

    total = 0
    for layer in layers:
        total += net_feature_size[layer][0]
    feature_dim = min(feature_dim, total)

    return torch.tensor(random.sample(range(0, total), feature_dim))
