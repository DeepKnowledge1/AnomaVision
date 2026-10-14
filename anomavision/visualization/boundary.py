from typing import Tuple, Union

import cv2
import numpy as np
import torch
from skimage.segmentation import find_boundaries

from .frame import frame_by_anomalies
from .utils import composite_image, to_numpy


def framed_boundary_images(
    images: Union[np.ndarray, torch.Tensor],
    patch_classifications: Union[np.ndarray, torch.Tensor],
    image_classifications: Union[np.ndarray, torch.Tensor],
    padding: int = 30,
    boundary_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Draw boundaries around masked areas on images and adds
    a frame around the image that indicates if a boundary was drawn.

    Args:
        images: Images on which to draw boundaries.
        patch_classifications: anomaly classifications about the images.
        image_classifications: information about, if the images have anomalies
        padding: the thickness of the border around the images.
        boundary_color: Color of boundaries.

    Returns:
        b_image: Image with boundaries.

    """

    images = to_numpy(images).copy()
    masks = to_numpy(patch_classifications).copy()
    image_classifications = to_numpy(image_classifications).copy()

    b_images = boundary_images(images, masks, boundary_color=boundary_color)
    framed_b_images = frame_by_anomalies(
        b_images, image_classifications, padding=padding
    )

    return np.array(framed_b_images)


def boundary_images(
    images: Union[np.ndarray, torch.Tensor],
    patch_classifications: Union[np.ndarray, torch.Tensor],
    boundary_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Draw boundaries around masked areas on images and adds
    a frame around the image that indicates if a boundary was drawn.

    Args:
        images: Images on which to draw boundaries.
        patch_classifications: anomaly classifications about the images.
        boundary_color: Color of boundaries.

    Returns:
        b_image: Image with boundaries.

    """

    images = to_numpy(images).copy()
    masks = to_numpy(patch_classifications).copy()

    b_images = [
        boundary_image(image, masks[i], boundary_color=boundary_color)
        for i, image in enumerate(images)
    ]

    return np.array(b_images)


def boundary_image(
    image: Union[np.ndarray, torch.Tensor],
    patch_classification: Union[np.ndarray, torch.Tensor],
    boundary_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Draw boundaries around masked areas on image.

    Args:
        image: Image on which to draw boundaries.
        patch_classification: Mask defining the areas.
        boundary_color: Color of boundaries.

    Returns:
        b_image: Image with boundaries.

    """

    image = to_numpy(image).copy()
    mask = np.squeeze(to_numpy(patch_classification).copy())

    if mask.ndim != 2:
        raise ValueError(
            f"patch_classification must be a 2D mask after squeezing; got shape {mask.shape}"
        )

    # Resize the localization mask itself, not its one-pixel boundary.
    # This preserves the defect region and lets OpenCV trace a visible
    # contour at the final image resolution.
    binary_mask = (mask > 0.5).astype(np.uint8)

    if binary_mask.shape != image.shape[:2]:
        binary_mask = cv2.resize(
            binary_mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Fill tiny gaps introduced by patch/grid localization while keeping
    # separate defects as separate regions.
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    b_image = image.copy()
    if contours:
        cv2.drawContours(
            b_image,
            contours,
            contourIdx=-1,
            color=tuple(int(v) for v in boundary_color),
            thickness=max(2, min(image.shape[:2]) // 150),
        )

    return b_image
