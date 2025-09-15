import queue

import cv2
import numpy as np
import torch
from PIL import Image

from .base import BaseRealtimeDataset


class VideoDataset(BaseRealtimeDataset):
    """Dataset for video file input"""

    def __init__(self, video_path: str, loop: bool = True, **kwargs):
        """
        Args:
            video_path: Path to video file
            loop: Whether to loop video when it ends
        """
        super().__init__(**kwargs)
        self.video_path = video_path
        self.loop = loop
        self.cap = None

    def _capture_loop(self):
        """Capture frames from video file"""
        import time

        while self._running:
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video file {self.video_path}")

            frame_count = 0
            while self._running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if self.loop:
                        break  # Restart video
                    else:
                        self._running = False
                        break

                metadata = {
                    "timestamp": time.time(),
                    "frame_number": frame_count,
                    "source": self.video_path,
                }

                try:
                    self.frame_buffer.put_nowait((frame, metadata))
                except queue.Full:
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait((frame, metadata))
                    except queue.Empty:
                        pass

                frame_count += 1
                time.sleep(0.033)  # ~30 FPS

            self.cap.release()
            if not self.loop:
                break

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
