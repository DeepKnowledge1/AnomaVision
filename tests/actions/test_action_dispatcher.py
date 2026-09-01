from anomavision.actions.ActionBase import ActionBase
from anomavision.actions.ActionDispatcher import ActionDispatcher


class RecordingAction(ActionBase):
    def __init__(self, should_fail=False):
        self.connected = False
        self.executed = []
        self.disconnected = False
        self.should_fail = should_fail

    def connect(self):
        self.connected = True
        return True

    def execute(self, result):
        if self.should_fail:
            raise RuntimeError("simulated action failure")
        self.executed.append(result)
        return True

    def disconnect(self):
        self.disconnected = True
        self.connected = False

    def is_connected(self):
        return self.connected


def test_dispatcher_connect_execute_and_disconnect():
    first = RecordingAction()
    second = RecordingAction()
    dispatcher = ActionDispatcher([first, second])

    dispatcher.connect_all()
    result = {"decision": "FAIL", "anomaly_score": 0.95}
    assert dispatcher.execute_all(result) == [True, True]
    assert first.executed == [result]
    assert second.executed == [result]

    dispatcher.disconnect_all()
    assert first.disconnected
    assert second.disconnected


def test_dispatcher_isolates_action_failure():
    failing = RecordingAction(should_fail=True)
    working = RecordingAction()
    dispatcher = ActionDispatcher([failing, working])

    assert dispatcher.execute_all({"decision": "FAIL"}) == [False, True]
    assert len(working.executed) == 1


def test_dispatcher_rolls_back_connections_on_startup_failure():
    working = RecordingAction()

    class FailingConnectAction(RecordingAction):
        def connect(self):
            raise RuntimeError("connection failed")

    dispatcher = ActionDispatcher([working, FailingConnectAction()])

    try:
        dispatcher.connect_all()
    except RuntimeError:
        pass
    else:
        raise AssertionError("connect_all should raise")

    assert working.disconnected
