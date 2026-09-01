from abc import ABC, abstractmethod
from typing import Any, Dict


class ActionBase(ABC):
    """Base interface for actions executed from an AnomaVision result."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish the connection required by the action."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, result: Dict[str, Any]) -> bool:
        """Execute the action using an inference/inspection result."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources owned by the action."""
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the action is currently connected."""
        raise NotImplementedError
