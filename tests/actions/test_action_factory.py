import pytest

from anomavision.actions.ActionBase import ActionBase
from anomavision.actions.ActionFactory import ActionFactory


class DummyAction(ActionBase):
    def __init__(self, value=1):
        self.value = value
        self.connected = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def execute(self, result):
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected


def test_register_and_create_custom_action():
    ActionFactory.register("dummy", DummyAction)

    action = ActionFactory.create({"type": "dummy", "value": 42})

    assert isinstance(action, DummyAction)
    assert action.value == 42


def test_create_requires_type():
    with pytest.raises(ValueError, match="requires a 'type'"):
        ActionFactory.create({})


def test_create_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown Action type"):
        ActionFactory.create({"type": "does-not-exist"})


def test_create_all():
    actions = ActionFactory.create_all(
        [
            {"type": "dummy", "value": 1},
            {"type": "dummy", "value": 2},
        ]
    )

    assert [action.value for action in actions] == [1, 2]
