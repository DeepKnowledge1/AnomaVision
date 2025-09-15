import queue
import socket

import cv2
import numpy as np

from .base import BaseRealtimeDataset
from typing import Dict, Any
import time
from anodet.utils import get_logger
logger = get_logger(__name__)

class TCPDataset(BaseRealtimeDataset):
    """Dataset for TCP socket image data"""

    def __init__(
        self,
        host: str,
        port: int,
        buffer_size_tcp: int = 4096,
        socket_timeout: float = 5.0,
        **kwargs
    ):
        """
        Args:
            host: TCP server hostname
            port: TCP server port
            buffer_size_tcp: TCP receive buffer size
            socket_timeout: Socket timeout in seconds
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.buffer_size_tcp = buffer_size_tcp
        self.socket_timeout = socket_timeout
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
                        data = self.sock.recv(self.buffer_size_tcp)
                        if not data:
                            logger.warning("TCP connection closed by server")
                            break

                        self._bytes_received += len(data)

                        # Decode image data
                        nparr = np.frombuffer(data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            metadata = {
                                "source": f"tcp_{self.host}_{self.port}",
                                "bytes_received": self._bytes_received,
                                "data_size": len(data)
                            }
                            self._add_frame_to_buffer(frame, metadata)
                        else:
                            logger.debug("Could not decode TCP image data")

                    except socket.timeout:
                        if self._running:
                            logger.debug("TCP receive timeout, continuing...")
                        continue
                    except socket.error as e:
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
