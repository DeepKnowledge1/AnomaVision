import pytest

from anomavision.actions.ActionFactory import ActionFactory
from anomavision.actions.OPCUAAction import OPCUAAction


class FakeNode:
    def __init__(self):
        self.values = []

    def set_value(self, value):
        self.values.append(value)


class FakeClient:
    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.nodes = {
            "decision": FakeNode(),
            "score": FakeNode(),
        }

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.disconnected = True

    def get_node(self, node_id):
        return self.nodes[node_id]


def test_opcua_action_connect_execute_disconnect():
    client = FakeClient()
    action = OPCUAAction(
        endpoint="opc.tcp://localhost:4840",
        decision_node="decision",
        score_node="score",
        client=client,
    )

    assert action.connect() is True
    assert action.execute({"decision": "FAIL", "anomaly_score": 0.91}) is True
    assert client.nodes["decision"].values == [True]
    assert client.nodes["score"].values == [0.91]

    action.disconnect()
    assert client.disconnected is True


def test_opcua_action_rejects_unknown_decision():
    client = FakeClient()
    action = OPCUAAction(
        endpoint="opc.tcp://localhost:4840",
        decision_node="decision",
        client=client,
    )
    action.connect()

    with pytest.raises(ValueError, match="Unsupported inspection decision"):
        action.execute({"decision": "MAYBE", "anomaly_score": 0.2})


def test_factory_creates_opcua_action():
    action = ActionFactory.create(
        {
            "type": "opcua",
            "endpoint": "opc.tcp://localhost:4840",
            "decision_node": "decision",
        }
    )
    assert isinstance(action, OPCUAAction)
