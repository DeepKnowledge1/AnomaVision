import os
from typing import List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from ..utils import (
    create_image_transform,
    create_mask_transform,
    standard_image_transform,
    standard_mask_transform,
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in [
        "png",
        "jpg",
        "jpeg",
    ]


class AnodetDataset(Dataset):

    def __init__(
        self,
        image_directory_path: str,
        mask_directory_path: Optional[str] = None,
        image_transforms: Optional[T.Compose] = None,
        mask_transforms: Optional[T.Compose] = None,
        resize: Union[int, Tuple[int, int]] = 224,
        crop_size: Optional[Union[int, Tuple[int, int]]] = 224,
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        """
        Args:
            image_directory_path: Path to directory containing images
            mask_directory_path: Optional path to directory containing masks
            image_transforms: Optional pre-built image transforms. If None, creates from other parameters.
            mask_transforms: Optional pre-built mask transforms. If None, creates from other parameters.
            resize: Size to resize to. If int, resize shortest edge. If tuple (h, w), resize to exact dimensions.
            crop_size: Size to crop to. If None, no cropping. If int, center crop to square. If tuple (h, w), crop to exact dimensions.
            normalize: Whether to apply ImageNet normalization to images.
            mean: Mean values for normalization.
            std: Standard deviation values for normalization.
        """

        # Use provided transforms or create new ones with configurable parameters
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

        if mask_transforms is not None:
            self.mask_transforms = mask_transforms
        else:
            self.mask_transforms = create_mask_transform(
                resize=resize, crop_size=crop_size
            )

        # Load image paths
        self.image_directory_path = image_directory_path
        self.image_paths = []
        for file in os.listdir(self.image_directory_path):
            filename = os.fsdecode(file)
            if allowed_file(filename):
                self.image_paths.append(
                    os.path.join(self.image_directory_path, filename)
                )

        # Load mask paths if mask_directory_path argument is given
        self.mask_directory_path = mask_directory_path
        self.mask_paths = []
        if self.mask_directory_path is not None:
            for file in os.listdir(self.mask_directory_path):
                filename = os.fsdecode(file)
                if allowed_file(filename):
                    self.mask_paths.append(
                        os.path.join(self.mask_directory_path, filename)
                    )

            assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        batch = self.image_transforms(image)
        image = np.array(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Load mask if mask_directory_path argument is given
        if self.mask_directory_path is not None:
            mask = Image.open(self.mask_paths[idx])
            mask = self.mask_transforms(mask)
            image_classification = 0
        else:
            mask = torch.zeros([1, batch.shape[1], batch.shape[2]])
            image_classification = 1

        return batch, image, image_classification, mask
