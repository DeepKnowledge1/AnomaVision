import os
from typing import List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils import create_image_transform, create_mask_transform

valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get_class_names(
    dataset_path: str, valid_classes: Optional[List[str]] = None
) -> List[str]:
    """
    Discover valid class names inside dataset_path (MVTec-style format).

    A valid class is any folder inside dataset_path that contains
    at least a 'train/' or 'test/' subfolder.

    Args:
        dataset_path: Root of the dataset.
        valid_classes: Optional list to restrict to known classes.

    Returns:
        Sorted list of valid class names.
    """
    classes = []
    for d in os.listdir(dataset_path):
        d_path = os.path.join(dataset_path, d)
        if not os.path.isdir(d_path):
            continue
        # Accept only if it has train/ or test/
        if any(os.path.isdir(os.path.join(d_path, sub)) for sub in ["train", "test"]):
            classes.append(d)

    if valid_classes is not None:
        classes = [c for c in classes if c in valid_classes]

    return sorted(classes)


class MVTecDataset(Dataset):
    """
    Generic dataset loader for datasets following the MVTec format.

    Compatible with:
      - MVTec AD
      - VISA
      - BTAD
      - Any dataset with the same directory structure
    """

    def __init__(
        self,
        dataset_path: str,
        class_name: str,
        is_train: bool = True,
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
            dataset_path: Path to dataset root (MVTec-style format).
            class_name: Name of the class (must exist in dataset).
            is_train: Load training or test split.
            image_transforms: Optional custom image transforms.
            mask_transforms: Optional custom mask transforms.
            resize: Resize shortest edge (int) or to exact (h, w).
            crop_size: Center crop or exact crop size.
            normalize: Apply ImageNet normalization.
            mean: Mean for normalization.
            std: Std for normalization.
        """

        available_classes = get_class_names(dataset_path)
        assert class_name in available_classes, (
            f"class_name '{class_name}' not found in dataset. "
            f"Available: {available_classes}"
        )

        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train

        # Load samples
        self.x, self.y, self.mask = self.load_dataset_folder()

        # Transforms
        self.image_transforms = image_transforms or create_image_transform(
            resize=resize, crop_size=crop_size, normalize=normalize, mean=mean, std=std
        )
        self.mask_transforms = mask_transforms or create_mask_transform(
            resize=resize, crop_size=crop_size
        )

    def __getitem__(self, idx):
        img_path, label, mask_path = self.x[idx], self.y[idx], self.mask[idx]

        img_pil = Image.open(img_path).convert("RGB")
        img_np = cv.cvtColor(np.array(img_pil), cv.COLOR_BGR2RGB)
        img_tensor = self.image_transforms(img_pil)

        if label == 0 or mask_path is None:
            mask = torch.zeros([1, img_tensor.shape[1], img_tensor.shape[2]])
        else:
            mask = Image.open(mask_path)
            mask = self.mask_transforms(mask)

        return img_tensor, img_np, label, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")

        if not os.path.isdir(img_dir):
            raise RuntimeError(f"Missing directory: {img_dir}")

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted(
                os.path.join(img_type_dir, f)
                for f in os.listdir(img_type_dir)
                if f.lower().endswith(valid_exts)
            )
            x.extend(img_fpath_list)

            if img_type == "good":
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list
                ]

                gt_fpath_list = []
                for fname in img_fname_list:
                    candidates = [
                        os.path.join(gt_type_dir, f"{fname}_mask.png"),
                        os.path.join(gt_type_dir, f"{fname}.png"),
                        os.path.join(gt_type_dir, f"{fname}_gt.png"),
                    ]
                    gt_file = next((c for c in candidates if os.path.exists(c)), None)
                    gt_fpath_list.append(gt_file)

                mask.extend(gt_fpath_list)

        assert len(x) == len(y), "number of images and labels must match"

        return list(x), list(y), list(mask)
