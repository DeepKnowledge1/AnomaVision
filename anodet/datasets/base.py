"""
Real-time Dataset Classes for Anomaly Detection
Supports MQTT, Webcam, Video, and TCP data sources
"""

import json
import os
import queue
import socket
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils import create_image_transform, create_mask_transform


class BaseRealtimeDataset(Dataset, ABC):
    """Base class for real-time data sources"""

    def __init__(
        self,
        buffer_size: int = 100,
        image_transforms=None,
        resize: Union[int, Tuple[int, int]] = 224,
        crop_size: Optional[Union[int, Tuple[int, int]]] = 224,
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        timeout: float = 1.0,
    ):
        """
        Args:
            buffer_size: Size of the circular buffer for storing frames
            timeout: Timeout for getting new frames
        """
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self._running = False
        self._thread = None

        # Image transforms (same as existing datasets)
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

    @abstractmethod
    def _capture_loop(self):
        """Abstract method for capturing frames in background thread"""
        pass

    def start(self):
        """Start the data capture thread"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the data capture thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_frame(self):
        """Get the latest frame (non-blocking)"""
        try:
            return self.frame_buffer.get_nowait()
        except queue.Empty:
            return None

    def __len__(self):
        """Return current buffer size"""
        return self.frame_buffer.qsize()

    def __getitem__(self, idx):
        """Get frame for inference"""
        frame_data = self.get_frame()
        if frame_data is None:
            # Return empty tensors if no frame available
            dummy_tensor = torch.zeros((3, 224, 224))
            return dummy_tensor, np.zeros((224, 224, 3)), 1, torch.zeros((1, 224, 224))

        image, metadata = frame_data

        # Process image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image

        batch = self.image_transforms(image_pil)
        image_np = np.array(image_pil)

        # No masks for real-time data
        mask = torch.zeros([1, batch.shape[1], batch.shape[2]])
        image_classification = 1  # Unknown classification for real-time

        return batch, image_np, image_classification, mask
