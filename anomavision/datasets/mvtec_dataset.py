import os
from typing import List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils import (
    create_image_transform,
    create_mask_transform,
    standard_image_transform,
    standard_mask_transform,
)

# URL =
# 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        class_name,
        is_train=True,
        image_transforms=None,
        mask_transforms=None,
        resize: Union[int, Tuple[int, int]] = 224,
        crop_size: Optional[Union[int, Tuple[int, int]]] = 224,
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        """
        Args:
            dataset_path: Path to MVTec dataset root
            class_name: Name of the class (must be in CLASS_NAMES)
            is_train: Whether to load training or test data
            image_transforms: Optional pre-built image transforms. If None, creates from other parameters.
            mask_transforms: Optional pre-built mask transforms. If None, creates from other parameters.
            resize: Size to resize to. If int, resize shortest edge. If tuple (h, w), resize to exact dimensions.
            crop_size: Size to crop to. If None, no cropping. If int, center crop to square. If tuple (h, w), crop to exact dimensions.
            normalize: Whether to apply ImageNet normalization to images.
            mean: Mean values for normalization.
            std: Standard deviation values for normalization.
        """

        assert class_name in CLASS_NAMES, "class_name: {}, should be in {}".format(
            class_name, CLASS_NAMES
        )
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

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

    def __getitem__(self, idx):
        batch, image_classification, mask = self.x[idx], self.y[idx], self.mask[idx]

        batch = Image.open(batch).convert("RGB")
        image = np.array(batch)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        batch = self.image_transforms(batch)

        if image_classification == 0:
            mask = torch.zeros([1, batch.shape[1], batch.shape[2]])
        else:
            mask = Image.open(mask)
            mask = self.mask_transforms(mask)

        return batch, image, image_classification, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".png")
                ]
            )
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == "good":
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list
                ]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + "_mask.png")
                    for img_fname in img_fname_list
                ]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), "number of x and y should be same"

        return list(x), list(y), list(mask)
