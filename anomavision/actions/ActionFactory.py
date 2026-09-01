from typing import Any, Dict, Iterable, List

from anomavision.actions.ActionBase import ActionBase
from anomavision.actions.MQTTAction import MQTTAction


class ActionFactory:
    """Build industrial actions from configuration dictionaries.

    Example:
        {"type": "mqtt", "broker": "localhost", "topic": "factory/results"}
    """

    _ACTIONS = {
        "mqtt": MQTTAction,
    }

    @classmethod
    def create(cls, config: Dict[str, Any]) -> ActionBase:
        if not isinstance(config, dict):
            raise TypeError("Action config must be a dictionary")

        action_type = config.get("type")
        if not action_type:
            raise ValueError("Action config requires a 'type' field")

        action_type = str(action_type).lower()
        action_class = cls._ACTIONS.get(action_type)
        if action_class is None:
            supported = ", ".join(sorted(cls._ACTIONS))
            raise ValueError(
                f"Unknown Action type: {action_type}. "
                f"Supported types: {supported}"
            )

        kwargs = {key: value for key, value in config.items() if key != "type"}
        return action_class(**kwargs)

    @classmethod
    def create_all(cls, configs: Iterable[Dict[str, Any]]) -> List[ActionBase]:
        """Create a list of actions from a sequence of configurations."""
        return [cls.create(config) for config in configs]

    @classmethod
    def register(cls, action_type: str, action_class: type[ActionBase]) -> None:
        """Register a custom action implementation.

        This lets downstream projects add integrations without modifying the
        factory itself.
        """
        if not action_type:
            raise ValueError("action_type cannot be empty")
        if not issubclass(action_class, ActionBase):
            raise TypeError("action_class must inherit from ActionBase")
        cls._ACTIONS[action_type.lower()] = action_class
