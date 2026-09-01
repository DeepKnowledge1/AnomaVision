"""OPC UA action for publishing inspection decisions to industrial systems."""

from typing import Any, Dict, Optional

from anomavision.actions.ActionBase import ActionBase


class OPCUAAction(ActionBase):
    """Publish inspection results to configured OPC UA nodes.

    The optional ``opcua`` dependency is imported lazily so users who only use
    video/offline inference do not need an OPC UA client installed.

    Configuration example::

        {
            "type": "opcua",
            "endpoint": "opc.tcp://192.168.1.50:4840",
            "decision_node": "ns=2;s=Line1.InspectionResult",
            "score_node": "ns=2;s=Line1.AnomalyScore"
        }
    """

    def __init__(
        self,
        endpoint: str,
        decision_node: str,
        score_node: Optional[str] = None,
        timeout: float = 5.0,
        decision_values: Optional[Dict[str, Any]] = None,
        client: Any = None,
    ) -> None:
        if not endpoint:
            raise ValueError("OPC UA endpoint is required")
        if not decision_node:
            raise ValueError("OPC UA decision_node is required")

        self.endpoint = endpoint
        self.decision_node = decision_node
        self.score_node = score_node
        self.timeout = float(timeout)
        self.decision_values = decision_values or {
            "PASS": False,
            "FAIL": True,
            "UNKNOWN": False,
        }
        self._client = client
        self._connected = False
        self._decision = None
        self._score = None

    def connect(self) -> bool:
        """Connect to the OPC UA server and resolve configured nodes."""
        if self._client is None:
            try:
                from opcua import Client
            except ImportError as exc:
                raise RuntimeError(
                    "OPC UA support requires the optional 'opcua' package"
                ) from exc
            self._client = Client(self.endpoint, timeout=self.timeout)

        try:
            self._client.connect()
            self._decision = self._client.get_node(self.decision_node)
            self._score = (
                self._client.get_node(self.score_node) if self.score_node else None
            )
            self._connected = True
            return True
        except Exception:
            self._connected = False
            try:
                self._client.disconnect()
            except Exception:
                pass
            raise

    def execute(self, result: dict) -> bool:
        """Write the inspection decision and optional anomaly score."""
        if not self._connected or self._decision is None:
            raise RuntimeError("OPC UA action is not connected")

        decision = str(result.get("decision", "UNKNOWN")).upper()
        if decision not in self.decision_values:
            raise ValueError(f"Unsupported inspection decision: {decision}")

        self._decision.set_value(self.decision_values[decision])

        if self._score is not None and result.get("anomaly_score") is not None:
            self._score.set_value(float(result["anomaly_score"]))

        return True

    def disconnect(self) -> None:
        """Disconnect from the OPC UA server."""
        if self._client is not None:
            try:
                self._client.disconnect()
            finally:
                self._connected = False
                self._decision = None
                self._score = None
