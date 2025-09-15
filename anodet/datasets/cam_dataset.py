"""
Refactored dataset implementations with eliminated code duplication
"""

import cv2
import time
from typing import Optional, Dict, Any

from .base import BaseRealtimeDataset, ConnectionError

from anodet.utils import get_logger
logger = get_logger(__name__)

class WebcamDataset(BaseRealtimeDataset):
    """Dataset for webcam input"""

    def __init__(
        self,
        camera_id: int = 0,
        fps_limit: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            camera_id: Camera device ID (0 for default webcam)
            fps_limit: Limit capture FPS (None for unlimited)
        """
        super().__init__(**kwargs)
        self.camera_id = camera_id
        self.fps_limit = fps_limit
        self.cap = None

    def _get_source_info(self) -> Dict[str, Any]:
        """Return webcam-specific metadata"""
        info = {'camera_id': self.camera_id, 'fps_limit': self.fps_limit}

        if self.cap and self.cap.isOpened():
            try:
                info.update({
                    'frame_width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'frame_height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'actual_fps': self.cap.get(cv2.CAP_PROP_FPS),
                })
            except Exception as e:
                logger.debug(f"Could not get camera properties: {e}")

        return info

    def _capture_loop(self):
        """Capture frames from webcam"""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot open webcam {self.camera_id}")

        # Test frame read
        ret, test_frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise ConnectionError(f"Webcam {self.camera_id} opened but cannot read frames")

        # Calculate frame delay
        frame_delay = 1.0 / self.fps_limit if self.fps_limit else 0

        try:
            while self._running:
                ret, frame = self.cap.read()
                if ret:
                    metadata = {
                        "source": f"webcam_{self.camera_id}",
                        "fps_limit": self.fps_limit
                    }
                    self._add_frame_to_buffer(frame, metadata)

                    if frame_delay > 0:
                        time.sleep(frame_delay)
                else:
                    logger.warning("Failed to read frame from webcam")
                    time.sleep(0.1)  # Brief pause before retrying
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None

    def close(self):
        """Clean up webcam resources"""
        super().close()
        if self.cap:
            self.cap.release()
            self.cap = None

