"""
Refactored Real-time Dataset Classes for Anomaly Detection
Eliminates code duplication and improves error handling
"""

import os
import cv2
import json
import socket
import threading
import queue
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List, Any, Dict
from contextlib import contextmanager
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

from ..utils import create_image_transform


from anodet.utils import get_logger
logger = get_logger(__name__)

class RealtimeDatasetError(Exception):
    """Base exception for real-time dataset errors"""
    pass


class ConnectionError(RealtimeDatasetError):
    """Raised when connection cannot be established or is lost"""
    pass


class FrameProcessingError(RealtimeDatasetError):
    """Raised when frame processing fails"""
    pass


class BaseRealtimeDataset(Dataset, ABC):
    """Base class for real-time data sources with common functionality"""

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
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            buffer_size: Size of the circular buffer for storing frames
            timeout: Timeout for getting new frames
            max_retries: Maximum connection retry attempts
            retry_delay: Delay between retry attempts
        """
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Performance monitoring
        self._frame_count = 0
        self._start_time = None
        self._last_frame_time = None

        # Setup image transforms
        self._setup_transforms(image_transforms, resize, crop_size, normalize, mean, std)

    def _setup_transforms(self, image_transforms, resize, crop_size, normalize, mean, std):
        """Setup image transforms with consistent parameters"""
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

    @abstractmethod
    def _get_source_info(self) -> Dict[str, Any]:
        """Return source-specific metadata"""
        pass

    def start(self):
        """Start the data capture thread"""
        with self._lock:
            if not self._running:
                logger.info(f"Starting {self.__class__.__name__} capture")
                self._running = True
                self._start_time = time.time()
                self._thread = threading.Thread(target=self._safe_capture_loop, daemon=True)
                self._thread.start()

    def stop(self):
        """Stop the data capture thread"""
        with self._lock:
            if self._running:
                logger.info(f"Stopping {self.__class__.__name__} capture")
                self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Thread did not terminate cleanly")

    def _safe_capture_loop(self):
        """Wrapper for capture loop with error handling"""
        retry_count = 0

        while self._running and retry_count <= self.max_retries:
            try:
                self._capture_loop()
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                logger.error(f"Capture error (attempt {retry_count}/{self.max_retries + 1}): {e}")

                if retry_count <= self.max_retries and self._running:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Max retries exceeded, stopping capture")
                    self._running = False

    def _add_frame_to_buffer(self, frame: np.ndarray, metadata: Dict[str, Any]):
        """Thread-safe method to add frame to buffer"""
        if not self._running:
            return

        try:
            # Update performance metrics
            current_time = time.time()
            self._frame_count += 1
            self._last_frame_time = current_time

            # Add frame timestamp
            metadata.update({
                'capture_time': current_time,
                'frame_number': self._frame_count
            })

            # Add to buffer (remove old frames if full)
            try:
                self.frame_buffer.put_nowait((frame, metadata))
            except queue.Full:
                try:
                    # Remove oldest frame
                    self.frame_buffer.get_nowait()
                    self.frame_buffer.put_nowait((frame, metadata))
                except queue.Empty:
                    # Buffer became empty between operations, try again
                    self.frame_buffer.put_nowait((frame, metadata))

        except Exception as e:
            logger.error(f"Error adding frame to buffer: {e}")

    def get_frame(self):
        """Get the latest frame (non-blocking)"""
        try:
            return self.frame_buffer.get_nowait()
        except queue.Empty:
            return None

    def _process_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Convert frame to the expected format (reduces code duplication)"""
        if frame is None:
            raise FrameProcessingError("Frame is None")

        try:
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # Convert to PIL Image
            image_pil = Image.fromarray(frame_rgb)

            # Apply transforms
            batch = self.image_transforms(image_pil)

            return batch, frame_rgb

        except Exception as e:
            raise FrameProcessingError(f"Failed to process frame: {e}")

    def _create_dummy_data(self) -> Tuple[torch.Tensor, np.ndarray, int, torch.Tensor]:
        """Create dummy data when no frame is available"""
        dummy_tensor = torch.zeros(3, 224, 224)
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        image_classification = 1
        mask = torch.zeros(1, 224, 224)
        return dummy_tensor, dummy_image, image_classification, mask

    def __len__(self):
        """Return current buffer size"""
        return self.frame_buffer.qsize()

    def __getitem__(self, idx):
        """Get frame for inference with consistent format"""
        frame_data = self.get_frame()
        if frame_data is None:
            return self._create_dummy_data()

        frame, metadata = frame_data

        try:
            batch, frame_rgb = self._process_frame(frame)

            # No anomaly masks for real-time data
            image_classification = 1
            mask = torch.zeros(1, batch.shape[1], batch.shape[2])

            return batch, frame_rgb, image_classification, mask

        except FrameProcessingError as e:
            logger.warning(f"Frame processing failed: {e}")
            return self._create_dummy_data()

    def __iter__(self):
        """Make dataset iterable - continuously yield frames"""
        while self._running:
            try:
                yield self.__getitem__(0)
                time.sleep(0.001)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                logger.error(f"Error in iteration: {e}")
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        current_time = time.time()
        uptime = current_time - self._start_time if self._start_time else 0
        fps = self._frame_count / uptime if uptime > 0 else 0

        return {
            'frame_count': self._frame_count,
            'uptime_seconds': uptime,
            'fps': fps,
            'buffer_size': self.frame_buffer.qsize(),
            'is_running': self._running,
            'last_frame_time': self._last_frame_time,
            **self._get_source_info()
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return {
            'class_name': self.__class__.__name__,
            'buffer_size': self.buffer_size,
            'timeout': self.timeout,
            'is_realtime': True,
            **self.get_stats()
        }

    def close(self):
        """Clean up resources"""
        logger.info(f"Closing {self.__class__.__name__}")
        self.stop()

        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup


class StaticDatasetMixin:
    """Mixin for static datasets to provide consistent interface"""

    def start(self):
        """No-op for static datasets"""
        pass

    def stop(self):
        """No-op for static datasets"""
        pass

    def close(self):
        """No-op for static datasets"""
        pass

    def is_realtime(self) -> bool:
        """Static datasets are not real-time"""
        return False

    def get_frame(self):
        """Not applicable for static datasets"""
        logger.warning("get_frame() not available for static datasets")
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Basic stats for static datasets"""
        return {
            'dataset_size': len(self),
            'is_running': False,
            'is_realtime': False
        }
