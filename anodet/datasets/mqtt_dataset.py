import queue
import time
from typing import Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import torch
from PIL import Image
from typing import Dict, Any
from .base import BaseRealtimeDataset

from anodet.utils import get_logger
logger = get_logger(__name__)

class MQTTDataset(BaseRealtimeDataset):
    """Dataset for MQTT image messages"""

    def __init__(
        self,
        broker_host: str,
        broker_port: int = 1883,
        topic: str = "camera/image",
        username: Optional[str] = None,
        password: Optional[str] = None,
        qos: int = 0,
        keep_alive: int = 60,
        **kwargs
    ):
        """
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            topic: MQTT topic to subscribe to
            username: MQTT username (optional)
            password: MQTT password (optional)
            qos: Quality of service level
            keep_alive: Keep alive interval
        """
        super().__init__(**kwargs)
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.username = username
        self.password = password
        self.qos = qos
        self.keep_alive = keep_alive
        self.client = None
        self._message_count = 0

    def _get_source_info(self) -> Dict[str, Any]:
        """Return MQTT-specific metadata"""
        return {
            'broker_host': self.broker_host,
            'broker_port': self.broker_port,
            'topic': self.topic,
            'qos': self.qos,
            'message_count': self._message_count,
            'connected': self.client.is_connected() if self.client else False
        }

    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker {self.broker_host}:{self.broker_port}")
            client.subscribe(self.topic, self.qos)
        else:
            logger.error(f"Failed to connect to MQTT broker, return code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection"""
        logger.info(f"Disconnected from MQTT broker, return code {rc}")

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            # Decode image data
            image_data = msg.payload
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                self._message_count += 1
                metadata = {
                    "topic": msg.topic,
                    "source": "mqtt",
                    "message_count": self._message_count,
                    "payload_size": len(image_data)
                }
                self._add_frame_to_buffer(frame, metadata)
            else:
                logger.warning("Could not decode MQTT image data")

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def _capture_loop(self):
        """Start MQTT client and listen for messages"""
        self.client = mqtt.Client()

        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Set credentials if provided
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        try:
            # Connect to broker
            self.client.connect(self.broker_host, self.broker_port, self.keep_alive)

            # Start network loop
            while self._running:
                self.client.loop(timeout=1.0)

        except Exception as e:
            raise ConnectionError(f"MQTT connection failed: {e}")
        finally:
            if self.client:
                try:
                    self.client.disconnect()
                except:
                    pass

    def close(self):
        """Clean up MQTT resources"""
        super().close()
        if self.client:
            try:
                self.client.disconnect()
                self.client = None
            except:
                pass

