from anomavision.datasets.StreamSource import StreamSource

from abc import ABC, abstractmethod
from typing import Optional
import socket

import cv2
import numpy as np



class TCPSource(StreamSource):
    """
    TCP source that receives length-prefixed image messages.

    Protocol (per frame):
        [4 bytes big-endian uint32: payload_length] [payload_length bytes: image data]

    Where the payload is a JPEG/PNG-encoded image that can be decoded with OpenCV.
    """

    def __init__(
        self,
        host: str,
        port: int,
        recv_timeout: Optional[float] = 1.0,
        header_size: int = 4,
        max_message_size: Optional[int] = None,
    ):
        """
        Args:
            host: Server host/IP to connect to.
            port: Server port to connect to.
            recv_timeout: Socket timeout in seconds for recv operations.
            header_size: Number of bytes used for the length header (default 4).
            max_message_size: Optional upper bound on payload size (bytes).
        """
        self.host = host
        self.port = port
        self.recv_timeout = recv_timeout
        self.header_size = header_size
        self.max_message_size = max_message_size

        self.socket: Optional[socket.socket] = None
        self._connected: bool = False

    # ---------- Internal helpers ----------

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """
        Receive exactly n bytes from the socket, or return None on failure/EOF.
        """
        if self.socket is None:
            return None

        data = bytearray()
        while len(data) < n:
            try:
                chunk = self.socket.recv(n - len(data))
            except (socket.timeout, socket.error):
                return None
            if not chunk:
                # Connection closed by peer
                return None
            data.extend(chunk)
        return bytes(data)

    # ---------- StreamSource interface ----------

    def connect(self):
        """Create the socket and connect to the remote host."""
        if self.socket is not None and self._connected:
            return  # already connected

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.recv_timeout is not None:
            self.socket.settimeout(self.recv_timeout)

        try:
            self.socket.connect((self.host, self.port))
        except Exception as e:
            self.socket.close()
            self.socket = None
            self._connected = False
            raise RuntimeError(f"Failed to connect to {self.host}:{self.port}") from e

        self._connected = True

    def read_frame(self):
        """
        Read a single frame from the TCP stream.

        Returns:
            frame_rgb: np.ndarray (H, W, C) in RGB format, or None on error/EOF/timeout.
        """
        if not self.is_connected():
            return None

        # 1) Read header (length)
        header = self._recv_exact(self.header_size)
        if header is None:
            return None

        # Interpret header as big-endian uint32
        payload_len = int.from_bytes(header, byteorder="big", signed=False)

        # Optional safety check
        if self.max_message_size is not None and payload_len > self.max_message_size:
            # Drain this message and skip it
            _ = self._recv_exact(payload_len)
            return None

        if payload_len == 0:
            return None

        # 2) Read payload
        payload = self._recv_exact(payload_len)
        if payload is None:
            return None

        # 3) Decode image
        data = np.frombuffer(payload, np.uint8)
        frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def disconnect(self):
        """Close the socket connection."""
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                # Socket might already be closed or not connected
                pass
            self.socket.close()
            self.socket = None

        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self.socket is not None
