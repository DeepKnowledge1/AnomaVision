import logging
from typing import Iterable, List

from anomavision.actions.ActionBase import ActionBase


class ActionDispatcher:
    """Execute configured actions without coupling the inference pipeline to them."""

    def __init__(self, actions: Iterable[ActionBase], logger=None):
        self.actions: List[ActionBase] = list(actions)
        self.logger = logger or logging.getLogger(__name__)

    def connect_all(self) -> None:
        """Connect all actions before processing starts."""
        connected = []
        try:
            for action in self.actions:
                action.connect()
                connected.append(action)
        except Exception:
            # Roll back already-connected actions so startup is deterministic.
            for action in reversed(connected):
                try:
                    action.disconnect()
                except Exception:
                    self.logger.exception("Failed to disconnect action during rollback")
            raise

    def execute_all(self, result) -> List[bool]:
        """Execute every action and isolate failures between integrations.

        A failing PLC/MQTT integration must not terminate image inference.
        The returned list contains one success flag per configured action.
        """
        statuses = []
        for action in self.actions:
            try:
                statuses.append(bool(action.execute(result)))
            except Exception:
                statuses.append(False)
                self.logger.exception(
                    "Action %s failed while processing inspection result",
                    action.__class__.__name__,
                )
        return statuses

    def disconnect_all(self) -> None:
        """Disconnect all actions, attempting cleanup even after failures."""
        for action in reversed(self.actions):
            try:
                action.disconnect()
            except Exception:
                self.logger.exception(
                    "Failed to disconnect action %s",
                    action.__class__.__name__,
                )
