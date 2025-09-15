import queue
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .base import BaseRealtimeDataset


class WebcamDataset(BaseRealtimeDataset):
    """Dataset for webcam input"""

    def __init__(self, camera_id: int = 0, fps_limit: Optional[int] = None, **kwargs):
        """
        Args:
            camera_id: Camera device ID (0 for default webcam)
            fps_limit: Limit capture FPS (None for unlimited)
        """
        super().__init__(**kwargs)
        self.camera_id = camera_id
        self.fps_limit = fps_limit
        self.cap = None

    def __len__(self):
        """Return buffer size as dataset length"""
        return self.buffer_size

    def __iter__(self):
        """Make dataset iterable - continuously yield frames"""
        while self._running:
            try:
                batch, image, classification, mask = self.__getitem__(0)
                yield batch, image, classification, mask
            except Exception as e:
                print(f"Error in iteration: {e}")
                break

    def __getitem__(self, idx):
        """Get current frame with same format as AnodetDataset"""
        frame_data = self.get_frame()
        if frame_data is None:
            # Return dummy data if no frame available
            dummy_tensor = torch.zeros(3, 224, 224)
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            return dummy_tensor, dummy_image, 1, torch.zeros(1, 224, 224)

        frame, metadata = frame_data

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        # Apply transforms if available
        if hasattr(self, "image_transforms") and self.image_transforms:
            batch = self.image_transforms(image_pil)
        else:
            # Basic transform if none provided
            batch = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

        # Match AnodetDataset return format: (batch, image, classification, mask)
        image_classification = 1  # No anomaly mask for real-time
        mask = torch.zeros([1, batch.shape[1], batch.shape[2]])

        return batch, frame_rgb, image_classification, mask

    def _capture_loop(self):
        """Capture frames from webcam"""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam {self.camera_id}. Check camera permissions and availability."
            )

        # Test if we can actually read frames
        ret, test_frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError(f"Webcam {self.camera_id} opened but cannot read frames")

        import time

        frame_delay = 1.0 / self.fps_limit if self.fps_limit else 0

        while self._running:
            ret, frame = self.cap.read()
            if ret:
                metadata = {
                    "timestamp": time.time(),
                    "source": f"webcam_{self.camera_id}",
                }

                # Add to buffer (remove old frames if full)
                try:
                    self.frame_buffer.put_nowait((frame, metadata))
                except queue.Full:
                    try:
                        self.frame_buffer.get_nowait()  # Remove oldest
                        self.frame_buffer.put_nowait((frame, metadata))
                    except queue.Empty:
                        pass

                if frame_delay > 0:
                    time.sleep(frame_delay)

        if self.cap:
            self.cap.release()
