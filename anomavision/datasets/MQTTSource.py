from anomavision.datasets.StreamSource import StreamSource

from typing import Optional
import threading
import time
from queue import Queue, Empty, Full

import cv2
import numpy as np

try:
    import paho.mqtt.client as mqtt
except ImportError as e:
    mqtt = None


class MQTTSource(StreamSource):
    """
    MQTT stream source with an internal bounded queue.

    - Background network loop handled by paho-mqtt (loop_start)
    - Incoming frames buffered in a bounded queue
    - When queue is full, oldest frame is dropped (real-time behavior)
    """

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
        connect_timeout: float = 5.0,
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
            connect_timeout: Seconds to wait for successful connect().
        """
        if mqtt is None:
            raise ImportError(
                "paho-mqtt is required for MQTTSource. Install via: pip install paho-mqtt"
            )

        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.keepalive = keepalive
        self.qos = qos
        self.read_timeout = read_timeout
        self.connect_timeout = connect_timeout

        self.client: Optional[mqtt.Client] = None

        self._frame_queue: Queue = Queue(maxsize=max_queue_size)

        self._lock = threading.Lock()
        self._connected = False
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()  # signals successful on_connect

    # ---------- MQTT callbacks ----------

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            with self._lock:
                self._connected = True
            self._connected_event.set()
            client.subscribe(self.topic, qos=self.qos)
        else:
            with self._lock:
                self._connected = False
            # Still set event so connect() can return quickly (failed)
            self._connected_event.set()

    def _on_disconnect(self, client, userdata, rc):
        with self._lock:
            self._connected = False

    def _on_message(self, client, userdata, msg):
        """
        Assumes payload is an encoded image (JPEG/PNG).
        Decodes to RGB np.ndarray and enqueues.
        """
        if self._stop_event.is_set():
            return

        payload = msg.payload
        if not payload:
            return

        data = np.frombuffer(payload, np.uint8)
        frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Drop-oldest strategy (robust against races)
        while True:
            try:
                self._frame_queue.put_nowait(frame_rgb)
                break
            except Full:
                try:
                    _ = self._frame_queue.get_nowait()
                except Empty:
                    # extremely unlikely race; just retry put
                    pass

    # ---------- StreamSource interface ----------

    def connect(self) -> None:
        """Connect to the MQTT broker and start background loop."""
        if self.is_connected():
            return

        self._stop_event.clear()
        self._connected_event.clear()

        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Connect + start network loop (runs callbacks in background thread)
        self.client.connect(self.broker, self.port, self.keepalive)
        self.client.loop_start()

        # Wait briefly for connection result (success or fail)
        self._connected_event.wait(timeout=self.connect_timeout)

        # If not connected, stop loop to avoid a dangling thread
        if not self.is_connected():
            self.disconnect()
            raise RuntimeError(
                f"Failed to connect to MQTT broker {self.broker}:{self.port} (topic={self.topic})"
            )

    def read_frame(self):
        """
        Returns:
            np.ndarray (H, W, C) in RGB format, or None if no frame available.
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
        """Disconnect from the broker and stop the background loop."""
        self._stop_event.set()

        if self.client is not None:
            try:
                self.client.loop_stop()
            except Exception:
                pass
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client = None

        with self._lock:
            self._connected = False

        # Clear buffered frames
        try:
            while True:
                self._frame_queue.get_nowait()
        except Empty:
            pass

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected and not self._stop_event.is_set()
