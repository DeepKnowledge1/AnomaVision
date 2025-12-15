from typing import Optional, Tuple, Union, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from anomavision.datasets.StreamSource import StreamSource
from anomavision.utils import (
    create_image_transform,
    create_mask_transform,
)


class StreamDataset(IterableDataset):
    """
    Streaming dataset with preprocessing identical to AnodetDataset.

    Yields:
        batch: torch.Tensor (C, H, W)
        image: np.ndarray (H, W, C) RGB
        image_classification: int
        mask: torch.Tensor (1, H, W)
    """

    def __init__(
        self,
        source: StreamSource,
        image_transforms=None,
        mask_transforms=None,
        resize: Union[int, Tuple[int, int]] = 224,
        crop_size: Optional[Union[int, Tuple[int, int]]] = 224,
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        max_frames: Optional[int] = None,
    ):
        self.source = source
        self.max_frames = max_frames

        # === Image preprocessing (same logic as AnodetDataset) ===
        if image_transforms is not None:
            self.image_transforms = image_transforms
        else:
            self.image_transforms = create_image_transform(
                resize=resize,
                crop_size=crop_size,
                normalize=normalize,
                mean=mean,
                std=std,
            )

        # === Mask preprocessing ===
        if mask_transforms is not None:
            self.mask_transforms = mask_transforms
        else:
            self.mask_transforms = create_mask_transform(
                resize=resize,
                crop_size=crop_size,
            )

    def __iter__(self):
        frame_count = 0

        while self.source.is_connected():
            if self.max_frames is not None and frame_count >= self.max_frames:
                break

            frame = self.source.read_frame()
            if frame is None:
                continue

            # frame: np.ndarray (H, W, C) RGB
            image_np = frame

            # Convert to PIL (same as offline)
            image_pil = Image.fromarray(image_np)

            # Apply preprocessing
            batch = self.image_transforms(image_pil)

            # Streaming = no GT anomaly
            image_classification = 0

            # Empty mask (same shape logic as AnodetDataset)
            _, H, W = batch.shape
            mask = torch.zeros((1, H, W), dtype=torch.float32)

            frame_count += 1
            yield batch, image_np, image_classification, mask
