
from anomavision.datasets.StreamSource import StreamSource
from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image






class StreamDataset(IterableDataset):
    """
    Real-time streaming dataset for any StreamSource (e.g. WebcamSource).
    Yields (image_tensor, image_np, label, mask).
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
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        # resize / crop / normalize args kept for consistency if you plug in your own transforms

    def __iter__(self):
        """
        Iterate through frames from the stream.
        Yields:
            image_tensor: transformed tensor (if transforms are provided)
            image_np: original numpy RGB frame
            label: int (default 0)
            mask: tensor mask (default zeros, shape [1, H, W])
        """
        self.source.connect()
        frame_count = 0

        try:
            while self.source.is_connected():
                if self.max_frames is not None and frame_count >= self.max_frames:
                    break

                frame = self.source.read_frame()
                if frame is None:
                    break

                image_np = frame  # np.ndarray (H, W, C), RGB

                # Convert to PIL for typical torchvision-style transforms
                img_pil = Image.fromarray(image_np)

                if self.image_transforms is not None:
                    image_tensor = self.image_transforms(img_pil)
                else:
                    # Fallback: convert to tensor manually (C, H, W), [0,1]
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

                # Default label & mask for streaming (no anomaly / no mask)
                label = 0
                _, H, W = image_tensor.shape
                if self.mask_transforms is not None:
                    # If you later have masks, use mask_transforms here
                    mask = self.mask_transforms(Image.fromarray(np.zeros((H, W), dtype=np.uint8)))
                else:
                    mask = torch.zeros((1, H, W), dtype=torch.float32)

                frame_count += 1
                yield image_tensor, image_np, label, mask

        finally:
            self.source.disconnect()

