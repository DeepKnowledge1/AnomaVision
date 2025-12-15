from anomavision.datasets.StreamSource import StreamSource

import cv2
import threading
import queue
import time
import numpy as np
from typing import Optional


class WebcamSource(StreamSource):
    """
    Webcam source with internal frame queue and background capture thread.

    - Capture runs continuously in a separate thread
    - read_frame() consumes from a bounded queue
    - If queue is full, oldest frame is dropped (real-time behavior)
    """

    def __init__(
        self,
        camera_id: int = 0,
        queue_size: int = 5,
        capture_fps: Optional[float] = None,
    ):
        """
        Args:
            camera_id: OpenCV camera index
            queue_size: Max number of frames to buffer
            capture_fps: Optional FPS limit for capture thread
        """
        self.camera_id = camera_id
        self.queue_size = queue_size
        self.capture_fps = capture_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected = False

    # ===================== LIFECYCLE =====================

    def connect(self) -> None:
        if self._connected:
            return

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"Failed to open webcam with id {self.camera_id}")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self._thread.start()

        self._connected = True

    def disconnect(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None

        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ===================== CAPTURE THREAD =====================

    def _capture_loop(self) -> None:
        sleep_time = 0.0
        if self.capture_fps and self.capture_fps > 0:
            sleep_time = 1.0 / self.capture_fps

        while not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame_bgr = self.cap.read()
            if not ret or frame_bgr is None:
                time.sleep(0.01)
                continue

            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Queue management: drop oldest if full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame_rgb)
            except queue.Full:
                pass

            if sleep_time > 0:
                time.sleep(sleep_time)

    # ===================== CONSUMER =====================

    def read_frame(self):
        """
        Returns:
            np.ndarray (H, W, C) in RGB format,
            or None if no frame available.
        """
        if not self._connected:
            return None

        try:
            return self.frame_queue.get(timeout=0.5)
        except queue.Empty:
            return None
