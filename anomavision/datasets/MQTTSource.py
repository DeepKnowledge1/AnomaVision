
from anomavision.datasets.StreamSource import StreamSource

from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image
try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("paho-mqtt is required for MQTTSource. Install via 'pip install paho-mqtt'.")


import threading
from queue import Queue, Empty




class MQTTSource(StreamSource):
    def __init__(
        self,
        broker: str,
        port: int,
        topic: str,
        client_id: Optional[str] = None,
        keepalive: int = 60,
        qos: int = 0,
        max_queue_size: int = 10,
        read_timeout: Optional[float] = 1.0,
    ):
        """
        Args:
            broker: MQTT broker hostname/IP.
            port: MQTT broker port.
            topic: Topic to subscribe to (e.g. 'camera/frames').
            client_id: Optional MQTT client ID.
            keepalive: Keepalive interval in seconds.
            qos: MQTT QoS level (0, 1, or 2).
            max_queue_size: Max buffered frames.
            read_timeout: Seconds to wait for a frame in read_frame().
        """
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.keepalive = keepalive
        self.qos = qos
        self.read_timeout = read_timeout

        self.client: Optional[mqtt.Client] = None
        self._connected = False

        self._frame_queue: Queue = Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()

    # ---------- MQTT callbacks ----------

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            with self._lock:
                self._connected = True
            client.subscribe(self.topic, qos=self.qos)
        else:
            # connection failed
            with self._lock:
                self._connected = False

    def _on_disconnect(self, client, userdata, rc):
        with self._lock:
            self._connected = False

    def _on_message(self, client, userdata, msg):
        """
        Called when a new MQTT message arrives on the subscribed topic.
        Assumes payload is an encoded image (JPEG/PNG).
        """
        payload = msg.payload
        if not payload:
            return

        # Decode bytes -> OpenCV image
        data = np.frombuffer(payload, np.uint8)
        frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Non-blocking put (drop oldest if full)
        if self._frame_queue.full():
            try:
                _ = self._frame_queue.get_nowait()
            except Empty:
                pass
        self._frame_queue.put_nowait(frame_rgb)

    # ---------- StreamSource interface ----------

    def connect(self):
        """Connect to the MQTT broker and start background loop."""
        if self.client is not None and self.is_connected():
            return  # already connected

        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.client.connect(self.broker, self.port, self.keepalive)
        # Start network loop in background thread
        self.client.loop_start()

    def read_frame(self):
        """
        Read the latest available frame from the MQTT buffer.

        Returns:
            np.ndarray (H, W, C) in RGB format, or None if
            - not connected, or
            - no frame arrives before read_timeout.
        """
        if not self.is_connected():
            return None

        try:
            frame = self._frame_queue.get(
                timeout=self.read_timeout
            ) if self.read_timeout is not None else self._frame_queue.get()
        except Empty:
            return None

        return frame

    def disconnect(self):
        """Disconnect from the broker and stop the background loop."""
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None

        with self._lock:
            self._connected = False

        # Clear buffered frames
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected
