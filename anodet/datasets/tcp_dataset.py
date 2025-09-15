import queue
import socket
import struct
import cv2
import numpy as np
from .base import BaseRealtimeDataset
from typing import Dict, Any
import time
from anodet.utils import get_logger

logger = get_logger(__name__)

class TCPDataset(BaseRealtimeDataset):
    """Dataset for TCP socket image data with proper message framing"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        buffer_size_tcp: int = 4096,
        socket_timeout: float = 5.0,
        use_length_prefix: bool = True,
        **kwargs
    ):
        """
        Args:
            host: TCP server hostname
            port: TCP server port
            buffer_size_tcp: TCP receive buffer size
            socket_timeout: Socket timeout in seconds
            use_length_prefix: Whether to expect length-prefixed messages
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.buffer_size_tcp = buffer_size_tcp
        self.socket_timeout = socket_timeout
        self.use_length_prefix = use_length_prefix
        self.sock = None
        self._bytes_received = 0

    def _get_source_info(self) -> Dict[str, Any]:
        """Return TCP-specific metadata"""
        return {
            'host': self.host,
            'port': self.port,
            'buffer_size_tcp': self.buffer_size_tcp,
            'bytes_received': self._bytes_received,
            'connected': self.sock is not None
        }

    def _receive_exact(self, size: int) -> bytes:
        """Receive exactly 'size' bytes from socket"""
        data = b''
        while len(data) < size:
            try:
                chunk = self.sock.recv(size - len(data))
                if not chunk:
                    raise ConnectionError("Socket connection closed")
                data += chunk
            except socket.timeout:
                if not self._running:
                    break
                continue
        return data

    def _receive_length_prefixed_message(self) -> bytes:
        """Receive a length-prefixed message (4 bytes length + data)"""
        # First, receive the length (4 bytes, big-endian unsigned int)
        length_data = self._receive_exact(4)
        if len(length_data) != 4:
            raise ValueError("Could not receive message length")

        message_length = struct.unpack('>I', length_data)[0]

        # Validate message length
        if message_length <= 0 or message_length > 10 * 1024 * 1024:  # Max 10MB
            raise ValueError(f"Invalid message length: {message_length}")

        # Receive the actual message
        return self._receive_exact(message_length)

    def _receive_delimited_message(self, delimiter: bytes = b'\r\n\r\n') -> bytes:
        """Receive data until delimiter is found"""
        data = b''
        while delimiter not in data:
            try:
                chunk = self.sock.recv(self.buffer_size_tcp)
                if not chunk:
                    break
                data += chunk
            except socket.timeout:
                if not self._running:
                    break
                continue

        # Remove delimiter if found
        if delimiter in data:
            data = data.split(delimiter)[0]

        return data

    def _decode_image_data(self, data: bytes) -> np.ndarray:
        """Try to decode image data with multiple formats"""
        if not data:
            return None

        try:
            # Try direct OpenCV decode
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                return frame

            # Try assuming it's raw image data with header info
            # This is useful if the sender includes metadata
            if len(data) > 12:  # Minimum header size
                # Try to find image start markers
                jpeg_marker = data.find(b'\xff\xd8\xff')
                png_marker = data.find(b'\x89PNG')

                if jpeg_marker >= 0:
                    image_data = data[jpeg_marker:]
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame

                elif png_marker >= 0:
                    image_data = data[png_marker:]
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame

            logger.debug(f"Could not decode {len(data)} bytes as image")
            return None

        except Exception as e:
            logger.debug(f"Image decode error: {e}")
            return None

    def _capture_loop(self):
        """Connect to TCP server and receive image data"""
        while self._running:
            self.sock = None
            try:
                # Create and configure socket
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.socket_timeout)

                # Connect to server
                logger.info(f"Connecting to TCP server {self.host}:{self.port}")
                self.sock.connect((self.host, self.port))
                logger.info("TCP connection established")

                # Receive data loop
                while self._running:
                    try:
                        if self.use_length_prefix:
                            # Receive length-prefixed message
                            data = self._receive_length_prefixed_message()
                        else:
                            # Try to receive complete message with delimiter
                            data = self._receive_delimited_message()

                            # Fallback: receive single chunk (original behavior)
                            if not data:
                                data = self.sock.recv(self.buffer_size_tcp)
                                if not data:
                                    logger.warning("TCP connection closed by server")
                                    break

                        self._bytes_received += len(data)

                        # Decode image data
                        frame = self._decode_image_data(data)

                        if frame is not None:
                            metadata = {
                                "source": f"tcp_{self.host}_{self.port}",
                                "bytes_received": self._bytes_received,
                                "data_size": len(data),
                                "frame_shape": frame.shape
                            }
                            self._add_frame_to_buffer(frame, metadata)
                        else:
                            logger.debug(f"Could not decode TCP image data ({len(data)} bytes)")
                            # Log first few bytes for debugging
                            if len(data) > 0:
                                header_bytes = data[:min(20, len(data))]
                                logger.debug(f"Data header: {header_bytes.hex()}")

                    except socket.timeout:
                        if self._running:
                            logger.debug("TCP receive timeout, continuing...")
                        continue
                    except (socket.error, ValueError, ConnectionError) as e:
                        logger.warning(f"TCP receive error: {e}")
                        break

            except socket.error as e:
                if self._running:
                    logger.error(f"TCP connection error: {e}")
                    time.sleep(self.retry_delay)
            except Exception as e:
                if self._running:
                    logger.error(f"Unexpected TCP error: {e}")
                    time.sleep(self.retry_delay)
            finally:
                self._cleanup_socket()

    def _cleanup_socket(self):
        """Clean up socket resources"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None

    def close(self):
        """Clean up TCP resources"""
        super().close()
        self._cleanup_socket()
