"""
Provides utility functions for anomaly detection.
"""

import argparse
import logging
import os
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms as T

# Default standard transforms - kept for backward compatibility
standard_image_transform = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

standard_mask_transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
"""
Examples:
# Create transforms for 512x384 images
transform = create_image_transform(resize=(512, 384), crop_size=None)

# Create transforms with 640x480 resize and 512x512 center crop
transform = create_image_transform(resize=(640, 480), crop_size=(512, 512))

# Create transforms with proportional resize to 400px shortest edge, no crop
transform = create_image_transform(resize=400, crop_size=None)
"""


def create_image_transform(
    resize: Union[int, Tuple[int, int]] = 224,
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> T.Compose:
    """
    Create a configurable image transform pipeline.

    Args:
        resize: Size to resize to. If int, resize shortest edge. If tuple (h, w), resize to exact dimensions.
        crop_size: Size to crop to. If None, no cropping. If int, center crop to square. If tuple (h, w), crop to exact dimensions.
        normalize: Whether to apply ImageNet normalization.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        Composed transform pipeline.
    """
    transforms = []

    # Handle resize - support both single int and tuple
    if isinstance(resize, int):
        transforms.append(T.Resize(resize))
    else:
        # For tuple (h, w), resize to exact dimensions
        transforms.append(T.Resize(resize))

    # Handle cropping - support both single int and tuple, or no cropping
    if crop_size is not None:
        if isinstance(crop_size, int):
            transforms.append(T.CenterCrop(crop_size))
        else:
            # For tuple (h, w), crop to exact dimensions
            transforms.append(T.CenterCrop(crop_size))

    # Always convert to tensor
    transforms.append(T.ToTensor())

    # Optional normalization
    if normalize:
        transforms.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transforms)


def create_mask_transform(
    resize: Union[int, Tuple[int, int]] = 224,
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
) -> T.Compose:
    """
    Create a configurable mask transform pipeline.

    Args:
        resize: Size to resize to. If int, resize shortest edge. If tuple (h, w), resize to exact dimensions.
        crop_size: Size to crop to. If None, no cropping. If int, center crop to square. If tuple (h, w), crop to exact dimensions.

    Returns:
        Composed transform pipeline.
    """
    transforms = []

    # Handle resize
    if isinstance(resize, int):
        transforms.append(T.Resize(resize))
    else:
        transforms.append(T.Resize(resize))

    # Handle cropping
    if crop_size is not None:
        if isinstance(crop_size, int):
            transforms.append(T.CenterCrop(crop_size))
        else:
            transforms.append(T.CenterCrop(crop_size))

    # Convert to tensor
    transforms.append(T.ToTensor())

    return T.Compose(transforms)


def to_batch(
    images: List[np.ndarray],
    transforms: Optional[T.Compose] = None,
    device: torch.device = torch.device("cpu"),
    resize: Union[int, Tuple[int, int]] = 224,
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a list of numpy array images to a pytorch tensor batch with given transforms.

    Args:
        images: List of numpy array images.
        transforms: Optional pre-built transforms. If None, will create transforms using other parameters.
        device: Device to move tensor to.
        resize: Size to resize to if transforms is None.
        crop_size: Size to crop to if transforms is None.
        normalize: Whether to apply normalization if transforms is None.

    Returns:
        Batch tensor of processed images.
    """
    assert len(images) > 0

    # Use provided transforms or create new ones
    if transforms is None:
        transforms = create_image_transform(
            resize=resize, crop_size=crop_size, normalize=normalize
        )

    transformed_images = []
    for i, image in enumerate(images):
        image = Image.fromarray(image).convert("RGB")
        transformed_images.append(transforms(image))

    # Get dimensions from first transformed image
    first_shape = transformed_images[0].shape
    channels, height, width = first_shape

    # Create batch tensor
    batch = torch.zeros((len(images), channels, height, width))

    for i, transformed_image in enumerate(transformed_images):
        batch[i] = transformed_image

    return batch.to(device)


# From: https://github.com/pytorch/pytorch/issues/19037
def pytorch_cov(
    tensor: torch.Tensor, rowvar: bool = True, bias: bool = False
) -> torch.Tensor:
    """Estimate a covariance matrix (np.cov)."""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def mahalanobis(
    mean: torch.Tensor, cov_inv: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    """Calculate the mahalonobis distance

    Calculate the mahalanobis distance between a multivariate normal distribution
    and a point or elementwise between a set of distributions and a set of points.

    Args:
        mean: A mean vector or a set of mean vectors.
        cov_inv: A inverse of covariance matrix or a set of covariance matricies.
        batch: A point or a set of points.

    Returns:
        mahalonobis_distance: A distance or a set of distances or a set of sets of distances.

    """

    # Assert that parameters has acceptable dimensions
    assert (
        len(mean.shape) == 1 or len(mean.shape) == 2
    ), "mean must be a vector or a set of vectors (matrix)"
    assert (
        len(batch.shape) == 1 or len(batch.shape) == 2 or len(batch.shape) == 3
    ), "batch must be a vector or a set of vectors (matrix) or a set of sets of vectors (3d tensor)"
    assert (
        len(cov_inv.shape) == 2 or len(cov_inv.shape) == 3
    ), "cov_inv must be a matrix or a set of matrices (3d tensor)"

    # Standardize the dimensions
    if len(mean.shape) == 1:
        mean = mean.unsqueeze(0)
    if len(cov_inv.shape) == 2:
        cov_inv = cov_inv.unsqueeze(0)
    if len(batch.shape) == 1:
        batch = batch.unsqueeze(0)
    if len(batch.shape) == 3:
        batch = batch.reshape(batch.shape[0] * batch.shape[1], batch.shape[2])

    # Assert that parameters has acceptable shapes
    assert mean.shape[0] == cov_inv.shape[0]
    assert mean.shape[1] == cov_inv.shape[1] == cov_inv.shape[2] == batch.shape[1]
    assert batch.shape[0] % mean.shape[0] == 0

    # Set shape variables
    mini_batch_size, length = mean.shape
    batch_size = batch.shape[0]
    ratio = int(batch_size / mini_batch_size)

    # If a set of sets of distances is to be computed, expand mean and cov_inv
    if batch_size > mini_batch_size:
        mean = mean.unsqueeze(0)
        mean = mean.expand(ratio, mini_batch_size, length)
        mean = mean.reshape(batch_size, length)
        cov_inv = cov_inv.unsqueeze(0)
        cov_inv = cov_inv.expand(ratio, mini_batch_size, length, length)
        cov_inv = cov_inv.reshape(batch_size, length, length)

    # Make sure tensors are correct type
    mean = mean.float()
    cov_inv = cov_inv.float()
    batch = batch.float()

    # Calculate mahalanobis distance
    diff = mean - batch
    mult1 = torch.bmm(diff.unsqueeze(1), cov_inv)
    mult2 = torch.bmm(mult1, diff.unsqueeze(2))
    sqrt = torch.sqrt(mult2)
    mahalanobis_distance = sqrt.reshape(batch_size)

    # If a set of sets of distances is to be computed, reshape output
    if batch_size > mini_batch_size:
        mahalanobis_distance = mahalanobis_distance.reshape(ratio, mini_batch_size)

    return mahalanobis_distance


def image_score(patch_scores: torch.Tensor) -> torch.Tensor:
    """Calculate image scores from patch scores.

    Args:
        patch_scores: A batch of patch scores.

    Returns:
        image_scores: A batch of image scores.

    """

    # Calculate max value of each matrix
    image_scores = torch.max(patch_scores.reshape(patch_scores.shape[0], -1), -1).values
    return image_scores


def classification(image_scores, thresh: float):
    """Calculate image classifications from image scores.
    Args:
        image_scores (torch.Tensor | np.ndarray): A batch of image scores.
        thresh (float): A threshold value. If an image score is larger than
                        or equal to thresh it is classified as anomalous.
    Returns:
        image_classifications (same type as input): A batch of image classifications.
    """

    if isinstance(image_scores, torch.Tensor):
        image_classifications = image_scores.clone()
        image_classifications[image_classifications < thresh] = 1
        image_classifications[image_classifications >= thresh] = 0

    elif isinstance(image_scores, np.ndarray):
        image_classifications = image_scores.copy()
        image_classifications[image_classifications < thresh] = 1
        image_classifications[image_classifications >= thresh] = 0
    else:
        raise TypeError("image_scores must be a torch.Tensor or numpy.ndarray")

    return image_classifications


def rename_files(source_path: str, destination_path: Optional[str] = None) -> None:
    """Rename all files in a directory path with increasing integer name.
    Ex. 0001.png, 0002.png ...
    Write files to destination path if argument is given.

    Args:
        source_path: Path to folder.
        destination_path: Path to folder.

    """
    for count, filename in enumerate(os.listdir(source_path), 1):
        file_source_path = os.path.join(source_path, filename)
        file_extension = os.path.splitext(filename)[1]

        new_name = str(count).zfill(4) + file_extension
        if destination_path:
            new_destination = os.path.join(destination_path, new_name)
        else:
            new_destination = os.path.join(source_path, new_name)

        os.rename(file_source_path, new_destination)


def split_tensor_and_run_function(
    func: Callable[[torch.Tensor], List],
    tensor: torch.Tensor,
    split_size: Union[int, List],
) -> torch.Tensor:
    """Splits the tensor into chunks in given split_size and run a function on each chunk.

    Args:
        func: Function to be run on a chunk of tensor.
        tensor: Tensor to split.
        split_size: Size of a single chunk or list of sizes for each chunk.

    Returns:
        output_tensor: Tensor of same size as input tensor

    """
    tensors_list = []
    for sub_tensor in torch.split(tensor, split_size):
        tensors_list.append(func(sub_tensor))

    output_tensor = torch.cat(tensors_list)

    return output_tensor


# Global flag to ensure logging is configured only once
_logging_configured = False
_log_filename = None


def setup_logging(log_level="INFO"):
    """
    Setup logging configuration - call once at application startup.
    Returns the root logger.
    """
    global _logging_configured, _log_filename

    # Only configure logging once
    if not _logging_configured:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Create timestamp for log filename (only once)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_filename = f"logs/detect_{timestamp}.log"

        # Clear any existing handlers on the root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Configure root logger (this affects all loggers)
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(_log_filename),
                logging.StreamHandler(),  # Also log to console
            ],
            force=True,  # Override any existing configuration
        )

        _logging_configured = True

        # Log the initialization message
        init_logger = logging.getLogger(__name__)
        init_logger.info(f"Logging initialized. Log file: {_log_filename}")

    # Return the root logger configured with the specified level
    return logging.getLogger()


def get_logger(name=None):
    """
    Get a logger for a specific module.
    Call setup_logging() first to configure the logging system.

    Args:
        name: Logger name (use __name__ from calling module)
    """
    if not _logging_configured:
        setup_logging()  # Auto-configure if not done yet

    return logging.getLogger(name)


def save_args_to_yaml__(args, filename="config.yml"):
    with open(filename, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def save_args_to_yaml(args, path, organize_structure=False):
    """
    Save a configuration to YAML.

    - Accepts argparse.Namespace, dict, dataclass, EasyDict, or any object with __dict__.
    - If organize_structure=True, writes a *sectioned* config with:
        common, training, inference, export  (inference/export derived from training/preprocess)

    Args:
        args: Configuration to save (effective training config recommended)
        path: Output file path
        organize_structure: If True, reorganize flat config into structured sections
    """
    from dataclasses import asdict, is_dataclass

    import numpy as np
    import yaml

    try:
        from easydict import EasyDict  # optional
    except Exception:
        EasyDict = None

    def make_yaml_serializable(obj):
        """Recursively convert obj to YAML-serializable types."""
        if isinstance(obj, (np.generic,)):  # numpy scalar
            return obj.item()
        if isinstance(obj, np.ndarray):  # numpy array
            return obj.tolist()
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        try:
            from pathlib import Path

            if isinstance(obj, Path):
                return str(obj)
        except Exception:
            pass
        if isinstance(obj, tuple):
            return [make_yaml_serializable(x) for x in obj]
        if isinstance(obj, list):
            return [make_yaml_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: make_yaml_serializable(v) for k, v in obj.items()}
        if is_dataclass(obj):
            return make_yaml_serializable(asdict(obj))
        if hasattr(obj, "__dict__"):  # argparse.Namespace / any object
            return make_yaml_serializable(vars(obj))
        return obj

    def organize_config(flat_config: dict) -> dict:
        """Split a flat training config into sections and add inference/export defaults."""
        flat_config = dict(flat_config)  # shallow copy

        # keys we mirror into inference/export
        preprocess_keys = ["resize", "crop_size", "normalize", "norm_mean", "norm_std"]

        common_keys = {
            "dataset_path",
            "class_name",
            "resize",
            "crop_size",
            "normalize",
            "norm_mean",
            "norm_std",
            "backbone",
            "feat_dim",
            "layer_indices",
            "model",
            "model_data_path",
            "device",
            "log_level",
            "enable_visualization",
            "save_visualizations",
            "viz_alpha",
            "viz_padding",
            "viz_color",
        }

        training_keys = {
            "batch_size",
            "output_model",
            "run_name",
            "epochs",
            "learning_rate",
            "weight_decay",
            "momentum",
        }

        organized = {"common": {}, "training": {}}

        # distribute keys
        for key, value in flat_config.items():
            if key in common_keys:
                organized["common"][key] = value
            elif key in training_keys:
                organized["training"][key] = value
            else:
                # default unknown-but-useful keys to common
                organized["common"][key] = value

        # build inference section (derived from preprocess + sensible runtime defaults)
        base_pre = {
            k: organized["common"].get(k, flat_config.get(k)) for k in preprocess_keys
        }
        inference = {
            "batch_size": flat_config.get("infer_batch_size", 1),
            "num_workers": flat_config.get("num_workers", 0),
            "pin_memory": flat_config.get("pin_memory", False),
            "enable_visualization": organized["common"].get(
                "enable_visualization", True
            ),
            "save_visualizations": organized["common"].get(
                "save_visualizations", False
            ),
            "viz_alpha": organized["common"].get("viz_alpha", 0.6),
            "viz_padding": organized["common"].get("viz_padding", 2),
            "viz_color": organized["common"].get("viz_color", [255, 0, 0]),
            "thresh": flat_config.get("thresh", None),
        }
        inference.update({k: v for k, v in base_pre.items() if v is not None})
        organized["inference"] = inference

        # build export section (shares preprocess, adds export knobs)
        export = {
            "format": flat_config.get(
                "export_format", "onnx"
            ),  # or 'torchscript', 'openvino', ...
            "opset": flat_config.get("opset", 17),
            "dynamic_batch": flat_config.get("dynamic_batch", False),
        }
        export.update({k: v for k, v in base_pre.items() if v is not None})
        organized["export"] = export

        # optional evaluation stub
        if "evaluation" not in organized:
            organized["evaluation"] = {
                "metrics": flat_config.get("metrics", ["auroc", "pixel_auroc"]),
                "val_batch_size": flat_config.get(
                    "val_batch_size", organized["training"].get("batch_size", 8)
                ),
            }

        return organized

    # Normalize input → dict
    if isinstance(args, dict):
        data = args
    elif EasyDict is not None and isinstance(args, EasyDict):
        data = dict(args)
    else:
        if is_dataclass(args):
            data = asdict(args)
        elif hasattr(args, "__dict__"):
            data = vars(args)
        else:
            raise TypeError(
                "save_args_to_yaml expects a dict, EasyDict, argparse.Namespace, dataclass, "
                "or an object with __dict__"
            )

    data = make_yaml_serializable(data)

    if organize_structure:
        data = organize_config(data)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def merge_config(args: argparse.Namespace, config: dict) -> dict:
    """
    Merge command-line arguments with YAML config.
    Args override config values if they are not None.
    """

    merged = config.copy()
    for key, value in vars(args).items():
        if (
            value is not None and key != "config"
        ):  # only override if user provided value
            merged[key] = value
    return merged


def easydict_to_dict(d):
    from easydict import EasyDict

    if isinstance(d, EasyDict):
        d = {k: easydict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        d = [easydict_to_dict(v) for v in d]
    return d


def yaml_save(file="data.yaml", data={}):
    config_dict = easydict_to_dict(data)

    # Save to YAML
    with open(file, "w") as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)


def save_args_to_yaml(config: dict, output_path: str):
    """Save dictionary as YAML file."""
    config = easydict_to_dict(config)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def adaptive_gaussian_blur(input_array, kernel_size=33, sigma=4):
    """
    Apply Gaussian blur using PyTorch if available, otherwise fallback to NumPy/SciPy
    Input should be either numpy array or torch tensor
    Handles batched inputs correctly: (B, H, W) or (B, C, H, W)
    """
    # Check if input is a torch tensor
    try:
        import torch
        import torchvision.transforms as T

        if torch.is_tensor(input_array):
            # Handle different input shapes
            if input_array.dim() == 2:
                # Single image (H, W) - add batch and channel dims
                input_reshaped = input_array.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                blurred = T.GaussianBlur(kernel_size, sigma=sigma)(input_reshaped)
                return blurred.squeeze(0).squeeze(0)  # Back to (H, W)

            elif input_array.dim() == 3:
                # Batch of images (B, H, W) - add channel dim
                input_reshaped = input_array.unsqueeze(1)  # (B, 1, H, W)
                blurred = T.GaussianBlur(kernel_size, sigma=sigma)(input_reshaped)
                return blurred.squeeze(1)  # Back to (B, H, W)

            elif input_array.dim() == 4:
                # Already in correct format (B, C, H, W) or (1, C, H, W)
                return T.GaussianBlur(kernel_size, sigma=sigma)(input_array)
            else:
                raise ValueError(f"Unsupported tensor dimensions: {input_array.dim()}")

    except ImportError:
        pass

    # Input is numpy array or PyTorch not available
    # Convert torch tensor to numpy if needed
    if hasattr(input_array, "detach"):
        input_array = input_array.detach().cpu().numpy()

    # Use NumPy/SciPy implementation
    try:
        from scipy.ndimage import gaussian_filter

        truncate = (kernel_size - 1) / (2 * sigma)

        # Handle different numpy array shapes
        if input_array.ndim == 2:
            # Single image (H, W)
            return gaussian_filter(input_array, sigma=sigma, truncate=truncate)

        elif input_array.ndim == 3:
            # Batch of images (B, H, W) - process each image separately
            blurred_batch = []
            for i in range(input_array.shape[0]):
                blurred_img = gaussian_filter(
                    input_array[i], sigma=sigma, truncate=truncate
                )
                blurred_batch.append(blurred_img)
            return np.stack(blurred_batch, axis=0)

        elif input_array.ndim == 4:
            # Batch with channels (B, C, H, W) - process each image and channel
            blurred_batch = []
            for b in range(input_array.shape[0]):
                blurred_channels = []
                for c in range(input_array.shape[1]):
                    blurred_channel = gaussian_filter(
                        input_array[b, c], sigma=sigma, truncate=truncate
                    )
                    blurred_channels.append(blurred_channel)
                blurred_batch.append(np.stack(blurred_channels, axis=0))
            return np.stack(blurred_batch, axis=0)
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {input_array.ndim}")

    except ImportError:
        raise ImportError("SciPy is required when PyTorch is not available")
