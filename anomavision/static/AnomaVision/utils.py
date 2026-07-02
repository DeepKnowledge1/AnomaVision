"""
Provides utility functions for anomaly detection.
"""

from typing import List

import cv2
import numpy as np

from scipy.ndimage import gaussian_filter


def to_batch(images: List[np.ndarray]) -> np.ndarray:
    resized = [cv2.resize(img, (224, 224)) for img in images]
    normalized = [
        (img / 255.0 - np.array([0.485, 0.456, 0.406]))
        / np.array([0.229, 0.224, 0.225])
        for img in resized
    ]
    batch = np.stack([img.transpose(2, 0, 1) for img in normalized])  # CHW
    return batch.astype(np.float32)


def classification(image_scores: np.ndarray, thresh: float) -> np.ndarray:
    """
    Classify images as anomalous (0) or normal (1) based on threshold.

    Args:
        image_scores: A 1D array of image anomaly scores.
        thresh: Threshold value to determine anomaly.

    Returns:
        An array of classifications: 0 = anomaly, 1 = normal.
    """
    return np.where(image_scores < thresh, 1, 0)


def adaptive_gaussian_blur(input_array, kernel_size=33, sigma=4):
    """
    Apply Gaussian blur using NumPy/SciPy.
    Input should be a numpy array.
    Handles batched inputs correctly: (H, W), (B, H, W), or (B, C, H, W).

    Args:
        input_array: Numpy array of the score maps.
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred numpy array with the same shape as input.
    """
    # Ensure input is a numpy array
    if not isinstance(input_array, np.ndarray):
        input_array = np.array(input_array)

    # Calculate truncate radius to match PyTorch's GaussianBlur behavior
    truncate = (kernel_size - 1) / (2 * sigma)

    if input_array.ndim == 2:
        # Single image (H, W)
        return gaussian_filter(input_array, sigma=sigma, truncate=truncate)

    elif input_array.ndim == 3:
        # Batch of images (B, H, W) - blur spatial dims only
        return np.array([
            gaussian_filter(input_array[i], sigma=sigma, truncate=truncate)
            for i in range(input_array.shape[0])
        ])

    elif input_array.ndim == 4:
        # Batch with channels (B, C, H, W) - blur spatial dims only
        return np.array([
            [
                gaussian_filter(input_array[b, c], sigma=sigma, truncate=truncate)
                for c in range(input_array.shape[1])
            ]
            for b in range(input_array.shape[0])
        ])
    else:
        raise ValueError(f"Unsupported numpy array dimensions: {input_array.ndim}")
