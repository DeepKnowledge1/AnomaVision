import queue
import socket

import cv2
import numpy as np

from .base import BaseRealtimeDataset


class TCPDataset(BaseRealtimeDataset):
    """Dataset for TCP socket image data"""

    def __init__(self, host: str, port: int, buffer_size_tcp: int = 4096, **kwargs):
        """
        Args:
            host: TCP server hostname
            port: TCP server port
            buffer_size_tcp: TCP receive buffer size
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.buffer_size_tcp = buffer_size_tcp
        self.sock = None

    def _capture_loop(self):
        """Connect to TCP server and receive image data"""
        import time

        while self._running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0)
                self.sock.connect((self.host, self.port))

                while self._running:
                    # Implement your TCP protocol here
                    # This is a simple example - adapt to your specific format
                    data = self.sock.recv(self.buffer_size_tcp)
                    if not data:
                        break

                    # Convert received data to image (implement based on your protocol)
                    # Example: expecting raw image bytes
                    nparr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        metadata = {
                            "timestamp": time.time(),
                            "source": f"tcp_{self.host}_{self.port}",
                        }

                        try:
                            self.frame_buffer.put_nowait((frame, metadata))
                        except queue.Full:
                            try:
                                self.frame_buffer.get_nowait()
                                self.frame_buffer.put_nowait((frame, metadata))
                            except queue.Empty:
                                pass

            except Exception as e:
                print(f"TCP connection error: {e}")
                time.sleep(1.0)  # Wait before reconnecting

            finally:
                if self.sock:
                    self.sock.close()

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
