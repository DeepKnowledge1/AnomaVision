from anomavision.datasets.StreamSource import StreamSource

from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image


class WebcamSource(StreamSource):
    def __init__(self, camera_id: int = 0):
        """
        Args:
            camera_id: Index of the camera for cv2.VideoCapture (usually 0).
        """
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> None:
        """Open the camera device."""
        if self.cap is not None and self.cap.isOpened():
            return  # already connected

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            # Clean up and raise a clear error
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"Failed to open webcam with id {self.camera_id}")

    def read_frame(self):
        """
        Read a single frame from the webcam.

        Returns:
            frame_rgb: np.ndarray (H, W, C) in RGB format, or None if read fails.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            return None

        # OpenCV returns BGR; convert to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def disconnect(self) -> None:
        """Release the camera."""
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None

    def is_connected(self) -> bool:
        """Return True if the camera is currently opened."""
        return self.cap is not None and self.cap.isOpened()

