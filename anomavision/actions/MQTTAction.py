import json
import threading
from typing import Any, Dict, Optional

from anomavision.actions.ActionBase import ActionBase

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


class MQTTAction(ActionBase):
    """Publish AnomaVision inspection results to an MQTT broker.

    The action owns one MQTT client connection. A separate connection manager
    can be introduced later when a source and action need to share a broker
    connection; keeping ownership local here makes the action easy to test
    and use independently.
    """

    def __init__(
        self,
        broker: str,
        topic: str,
        port: int = 1883,
        client_id: Optional[str] = None,
        keepalive: int = 60,
        qos: int = 1,
        retain: bool = False,
        connect_timeout: float = 5.0,
    ):
        if mqtt is None:
            raise ImportError(
                "paho-mqtt is required for MQTTAction. "
                "Install it with: pip install paho-mqtt"
            )

        if qos not in (0, 1, 2):
            raise ValueError("MQTT QoS must be 0, 1, or 2")

        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.keepalive = keepalive
        self.qos = qos
        self.retain = retain
        self.connect_timeout = connect_timeout

        self.client: Optional[mqtt.Client] = None
        self._lock = threading.Lock()
        self._connected = False
        self._connected_event = threading.Event()

    def _on_connect(self, client, userdata, flags, rc, *args):
        with self._lock:
            self._connected = rc == 0
        self._connected_event.set()

    def _on_disconnect(self, client, userdata, rc, *args):
        with self._lock:
            self._connected = False

    def connect(self) -> bool:
        """Connect to the MQTT broker and start the network loop."""
        if self.is_connected():
            return True

        self._connected_event.clear()
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect(self.broker, self.port, self.keepalive)
            self.client.loop_start()
        except Exception:
            self.disconnect()
            raise

        if not self._connected_event.wait(timeout=self.connect_timeout):
            self.disconnect()
            raise TimeoutError(
                f"Timed out connecting to MQTT broker "
                f"{self.broker}:{self.port}"
            )

        if not self.is_connected():
            self.disconnect()
            raise ConnectionError(
                f"Failed to connect to MQTT broker "
                f"{self.broker}:{self.port}"
            )

        return True

    def execute(self, result: Dict[str, Any]) -> bool:
        """Publish an inspection result as JSON.

        Raises:
            RuntimeError: If the action is not connected or publishing fails.
            TypeError: If the result cannot be JSON encoded.
        """
        if not self.is_connected() or self.client is None:
            raise RuntimeError("MQTTAction is not connected")

        payload = json.dumps(result, default=str, separators=(",", ":"))
        info = self.client.publish(
            self.topic,
            payload=payload,
            qos=self.qos,
            retain=self.retain,
        )

        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"MQTT publish failed with return code {info.rc}"
            )

        return True

    def disconnect(self) -> None:
        """Stop the MQTT network loop and close the broker connection."""
        client = self.client
        self.client = None

        if client is not None:
            try:
                client.loop_stop()
            finally:
                try:
                    client.disconnect()
                except Exception:
                    pass

        with self._lock:
            self._connected = False
        self._connected_event.clear()

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected
