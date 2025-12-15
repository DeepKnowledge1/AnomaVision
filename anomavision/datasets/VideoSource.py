
from anomavision.datasets.StreamSource import StreamSource

from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image


class VideoSource(StreamSource):
    def __init__(self, video_path: str, loop: bool = False):
        """
        Args:
            video_path: Path to the video file.
            loop: If True, restart from the beginning when the video ends.
        """
        self.video_path = video_path
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_index: int = 0
        self.num_frames: Optional[int] = None

    def connect(self):
        """Open the video file."""
        if self.cap is not None and self.cap.isOpened():
            return  # already opened

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Optionally store total number of frames (may be -1 on some codecs)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.num_frames = total if total > 0 else None
        self.frame_index = 0

    def _restart_if_looping(self) -> bool:
        """
        When loop=True and we reach the end of the video,
        restart from the first frame. Returns True if restarted,
        False if not (e.g., loop is False or restart failed).
        """
        if not self.loop or self.cap is None:
            return False

        # Seek back to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_index = 0
        return True

    def read_frame(self):
        """
        Read a single frame from the video.

        Returns:
            frame_rgb: np.ndarray (H, W, C) in RGB format,
                       or None if no more frames / read failed.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            # End of file or read error
            if self._restart_if_looping():
                ret, frame_bgr = self.cap.read()
                if not ret or frame_bgr is None:
                    return None
            else:
                return None

        self.frame_index += 1

        # OpenCV gives BGR; convert to RGB to match typical PyTorch pipelines
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def disconnect(self):
        """Release the video capture."""
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None

    def is_connected(self) -> bool:
        """Check if the video capture is opened."""
        return self.cap is not None and self.cap.isOpened()
