import queue
import time
from typing import Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import torch
from PIL import Image

from .base import BaseRealtimeDataset


class MQTTDataset(BaseRealtimeDataset):
    """Dataset for MQTT image messages"""

    def __init__(
        self,
        broker_host: str,
        broker_port: int = 1883,
        topic: str = "camera/image",
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            topic: MQTT topic to subscribe to
        """
        super().__init__(**kwargs)
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.username = username
        self.password = password
        self.client = None

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            # Assume message contains base64 encoded image or raw image bytes
            image_data = msg.payload

            # Convert bytes to numpy array (implement based on your MQTT format)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                metadata = {
                    "timestamp": time.time(),
                    "topic": msg.topic,
                    "source": "mqtt",
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
            print(f"Error processing MQTT message: {e}")

    def _capture_loop(self):
        """Start MQTT client and listen for messages"""
        self.client = mqtt.Client()

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_message = self._on_message
        self.client.connect(self.broker_host, self.broker_port, 60)
        self.client.subscribe(self.topic)

        while self._running:
            self.client.loop(timeout=1.0)

        self.client.disconnect()

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
