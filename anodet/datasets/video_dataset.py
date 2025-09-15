import queue

import cv2
import numpy as np
import torch
from PIL import Image
import time
from typing import Optional, Dict, Any

from .base import BaseRealtimeDataset

from anodet.utils import get_logger
logger = get_logger(__name__)


class VideoDataset(BaseRealtimeDataset):
    """Dataset for video file input"""

    def __init__(
        self,
        video_path: str,
        loop: bool = True,
        playback_speed: float = 1.0,
        **kwargs
    ):
        """
        Args:
            video_path: Path to video file
            loop: Whether to loop video when it ends
            playback_speed: Playback speed multiplier (1.0 = normal speed)
        """
        super().__init__(**kwargs)
        self.video_path = video_path
        self.loop = loop
        self.playback_speed = playback_speed
        self.cap = None
        self._total_frames = 0
        self._current_frame = 0

    def _get_source_info(self) -> Dict[str, Any]:
        """Return video-specific metadata"""
        info = {
            'video_path': self.video_path,
            'loop': self.loop,
            'playback_speed': self.playback_speed,
            'current_frame': self._current_frame,
            'total_frames': self._total_frames
        }

        if self.cap and self.cap.isOpened():
            try:
                info.update({
                    'frame_width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'frame_height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'original_fps': self.cap.get(cv2.CAP_PROP_FPS),
                })
            except Exception as e:
                logger.debug(f"Could not get video properties: {e}")

        return info

    def _capture_loop(self):
        """Capture frames from video file"""
        while self._running:
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                raise ConnectionError(f"Cannot open video file {self.video_path}")

            # Get video properties
            original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_delay = (1.0 / original_fps) / self.playback_speed

            self._current_frame = 0

            try:
                while self._running and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        if self.loop:
                            logger.info("Video ended, looping...")
                            break  # Break inner loop to restart video
                        else:
                            logger.info("Video ended")
                            self._running = False
                            break

                    metadata = {
                        "source": self.video_path,
                        "frame_number": self._current_frame,
                        "total_frames": self._total_frames,
                        "progress": self._current_frame / max(self._total_frames, 1)
                    }

                    self._add_frame_to_buffer(frame, metadata)
                    self._current_frame += 1

                    if frame_delay > 0:
                        time.sleep(frame_delay)
            finally:
                if self.cap:
                    self.cap.release()

            if not self.loop:
                break

    def close(self):
        """Clean up video resources"""
        super().close()
        if self.cap:
            self.cap.release()
            self.cap = None

