import logging
from typing import Iterable, List

from anomavision.actions.ActionBase import ActionBase


class ActionDispatcher:
    """Execute configured industrial actions without coupling them to inference."""

    def __init__(self, actions: Iterable[ActionBase], logger=None, fail_fast: bool = True):
        self.actions: List[ActionBase] = list(actions)
        self.logger = logger or logging.getLogger(__name__)
        self.fail_fast = fail_fast
        self._connected = set()

    def connect_all(self) -> None:
        """Connect actions, optionally continuing when an integration is unavailable."""
        connected = []
        for action in self.actions:
            try:
                action.connect()
                connected.append(action)
                self._connected.add(id(action))
            except Exception:
                self.logger.exception(
                    "Action %s failed to connect; %s",
                    action.__class__.__name__,
                    "continuing because fail_fast is disabled"
                    if not self.fail_fast else "aborting startup",
                )
                if self.fail_fast:
                    for connected_action in reversed(connected):
                        try:
                            connected_action.disconnect()
                        except Exception:
                            self.logger.exception(
                                "Failed to disconnect action during rollback"
                            )
                    self._connected.clear()
                    raise

    def execute_all(self, result) -> List[bool]:
        """Execute every available action and isolate integration failures."""
        statuses = []
        for action in self.actions:
            if id(action) not in self._connected and not action.is_connected():
                statuses.append(False)
                self.logger.warning(
                    "Skipping disconnected action %s",
                    action.__class__.__name__,
                )
                continue
            try:
                statuses.append(bool(action.execute(result)))
            except Exception:
                statuses.append(False)
                self.logger.exception(
                    "Action %s failed while processing inspection result",
                    action.__class__.__name__,
                )
                if self.fail_fast:
                    raise
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
        self._connected.clear()
