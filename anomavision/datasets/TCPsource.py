from anomavision.datasets.StreamSource import StreamSource

from typing import Optional
import socket
import threading
import time
from queue import Queue, Empty, Full

import cv2
import numpy as np


class TCPSource(StreamSource):
    """
    TCP source that receives length-prefixed image messages.

    Protocol (per frame):
        [4 bytes big-endian uint32: payload_length] [payload_length bytes: image data]
    Payload is JPEG/PNG image bytes decodable by OpenCV.

    Upgraded design:
      - Background receiver thread reads frames continuously
      - Frames are buffered into a bounded queue
      - When queue is full, oldest frame is dropped (real-time behavior)
      - read_frame() consumes from the queue
    """

    def __init__(
        self,
        host: str,
        port: int,
        recv_timeout: Optional[float] = 1.0,
        header_size: int = 4,
        max_message_size: Optional[int] = None,
        max_queue_size: int = 10,
        read_timeout: Optional[float] = 1.0,
        connect_timeout: float = 5.0,
    ):
        """
        Args:
            host: Server host/IP to connect to.
            port: Server port to connect to.
            recv_timeout: Socket timeout in seconds for recv operations (in receiver thread).
            header_size: Number of bytes used for the length header (default 4).
            max_message_size: Optional upper bound on payload size (bytes).
            max_queue_size: Max buffered decoded frames.
            read_timeout: Seconds to wait for a frame in read_frame().
            connect_timeout: Seconds to wait for TCP connect completion.
        """
        self.host = host
        self.port = port
        self.recv_timeout = recv_timeout
        self.header_size = header_size
        self.max_message_size = max_message_size

        self.max_queue_size = max_queue_size
        self.read_timeout = read_timeout
        self.connect_timeout = connect_timeout

        self.socket: Optional[socket.socket] = None

        self._frame_queue: Queue = Queue(maxsize=max_queue_size)

        self._lock = threading.Lock()
        self._connected: bool = False
        self._stop_event = threading.Event()

        self._thread: Optional[threading.Thread] = None

    # ---------- Internal helpers ----------

    def _set_connected(self, v: bool) -> None:
        with self._lock:
            self._connected = v

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """
        Receive exactly n bytes from the socket, or return None on failure/EOF/stop.
        Runs inside receiver thread.
        """
        if self.socket is None:
            return None

        data = bytearray()
        while len(data) < n and not self._stop_event.is_set():
            try:
                chunk = self.socket.recv(n - len(data))
            except socket.timeout:
                # receiver thread keeps trying (network jitter)
                continue
            except OSError:
                return None

            if not chunk:
                return None  # peer closed

            data.extend(chunk)

        if self._stop_event.is_set():
            return None

        return bytes(data) if len(data) == n else None

    def _enqueue_drop_oldest(self, frame_rgb: np.ndarray) -> None:
        """
        Enqueue frame. If full, drop oldest. Robust against races.
        """
        while True:
            try:
                self._frame_queue.put_nowait(frame_rgb)
                return
            except Full:
                try:
                    _ = self._frame_queue.get_nowait()
                except Empty:
                    # extremely unlikely race, retry put
                    pass

    def _receiver_loop(self) -> None:
        """
        Continuously read frames from TCP, decode, and enqueue.
        """
        try:
            while not self._stop_event.is_set() and self.socket is not None:
                # 1) Read header
                header = self._recv_exact(self.header_size)
                if header is None:
                    break

                payload_len = int.from_bytes(header, byteorder="big", signed=False)

                if payload_len <= 0:
                    continue

                if self.max_message_size is not None and payload_len > self.max_message_size:
                    # Drain and skip (best-effort)
                    _ = self._recv_exact(payload_len)
                    continue

                # 2) Read payload
                payload = self._recv_exact(payload_len)
                if payload is None:
                    break

                # 3) Decode
                data = np.frombuffer(payload, np.uint8)
                frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self._enqueue_drop_oldest(frame_rgb)

        finally:
            # Mark disconnected on thread exit
            self._set_connected(False)

    # ---------- StreamSource interface ----------

    def connect(self) -> None:
        """Create the socket, connect, and start receiver thread."""
        if self.is_connected():
            return

        self._stop_event.clear()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.recv_timeout is not None:
            s.settimeout(self.recv_timeout)

        # Connect with timeout safety
        s.settimeout(self.connect_timeout)
        try:
            s.connect((self.host, self.port))
        except Exception as e:
            try:
                s.close()
            except Exception:
                pass
            raise RuntimeError(f"Failed to connect to {self.host}:{self.port}") from e

        # After connect, restore recv_timeout for steady-state recv
        if self.recv_timeout is not None:
            s.settimeout(self.recv_timeout)

        self.socket = s
        self._set_connected(True)

        # Start receiver thread
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()

    def read_frame(self):
        """
        Returns:
            frame_rgb: np.ndarray (H, W, C) in RGB format,
            or None if no frame available before read_timeout or disconnected.
        """
        if not self.is_connected():
            return None

        try:
            if self.read_timeout is None:
                return self._frame_queue.get()
            return self._frame_queue.get(timeout=self.read_timeout)
        except Empty:
            return None

    def disconnect(self) -> None:
        """Stop receiver thread and close socket."""
        self._stop_event.set()

        # Close socket to unblock recv
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None

        # Join thread briefly
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        self._set_connected(False)

        # Clear buffered frames
        try:
            while True:
                self._frame_queue.get_nowait()
        except Empty:
            pass

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected and not self._stop_event.is_set() and self.socket is not None
