"""Helpers for routing existing AnomaVision inference results to actions."""

import time
from typing import Any, Iterable, Optional

import numpy as np

from anomavision.actions.ActionDispatcher import ActionDispatcher
from anomavision.actions.ActionFactory import ActionFactory
from anomavision.inspection.InspectionResult import InspectionResult


def create_action_dispatcher(actions_config, logger=None) -> Optional[ActionDispatcher]:
    """Create and connect configured actions, or return ``None`` when disabled."""
    if not actions_config:
        return None

    actions = ActionFactory.create_all(actions_config)
    dispatcher = ActionDispatcher(actions, logger=logger)
    dispatcher.connect_all()
    return dispatcher


def dispatch_inference_results(
    dispatcher: Optional[ActionDispatcher],
    image_scores: Iterable[Any],
    classifications: Iterable[Any],
    *,
    source_id: str,
    model_name: str,
    model_version: Optional[str] = None,
    frame_start: int = 0,
    inference_time_ms: Optional[float] = None,
    images=None,
) -> None:
    """Dispatch already-classified inference results without changing inference logic."""
    if dispatcher is None:
        return

    for index, (score, classification) in enumerate(zip(image_scores, classifications)):
        is_anomaly = bool(np.asarray(classification).item())
        result = InspectionResult(
            source_id=source_id,
            frame_id=frame_start + index,
            is_anomaly=is_anomaly,
            anomaly_score=float(np.asarray(score).item()),
            model_name=model_name,
            model_version=model_version,
            inference_time_ms=inference_time_ms,
            decision="FAIL" if is_anomaly else "PASS",
            event_id=f"{source_id}-{time.time_ns()}-{index}",
        ).to_dict()

        if images is not None:
            result["image"] = images[index]

        dispatcher.execute_all(result)
