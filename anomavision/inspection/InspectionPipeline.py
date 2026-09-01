"""Reusable inference-to-action pipeline components."""

import logging
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from anomavision.actions.ActionDispatcher import ActionDispatcher
from anomavision.inspection.DecisionEngine import DecisionEngine
from anomavision.inspection.InspectionResult import InspectionResult


class InspectionPipeline:
    """Convert model predictions into normalized results and actions.

    The pipeline is deliberately independent of a particular model backend or
    data source. Existing ``StreamSource``/``StreamDataset`` implementations
    can feed it, while ``ActionDispatcher`` handles industrial outputs.
    """

    def __init__(
        self,
        actions: Iterable[Any] = (),
        threshold: Optional[float] = None,
        source_id: str = "unknown",
        model_name: str = "unknown",
        model_version: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.source_id = source_id
        self.model_name = model_name
        self.model_version = model_version
        self.decision_engine = (
            DecisionEngine(threshold) if threshold is not None else None
        )
        self.action_dispatcher = ActionDispatcher(actions, logger=self.logger)
        self._connected = False
        self._frame_id = 0

    @property
    def has_actions(self) -> bool:
        return bool(self.action_dispatcher.actions)

    def connect(self) -> None:
        """Connect configured actions before production processing starts."""
        if not self.has_actions:
            return
        self.action_dispatcher.connect_all()
        self._connected = True

    def process(
        self,
        anomaly_score: Any,
        is_anomaly: Optional[Any] = None,
        inference_time_ms: Optional[float] = None,
        frame_id: Optional[int] = None,
    ) -> InspectionResult:
        """Create an inspection result and dispatch it to configured actions."""
        score = float(np.asarray(anomaly_score).reshape(-1)[0])

        if self.decision_engine is not None:
            decision = self.decision_engine.evaluate(score)
            anomaly = decision == "FAIL"
        else:
            anomaly = bool(np.asarray(is_anomaly).reshape(-1)[0]) if is_anomaly is not None else False
            decision = "FAIL" if anomaly else "PASS"

        current_frame_id = self._frame_id if frame_id is None else frame_id
        if frame_id is None:
            self._frame_id += 1

        result = InspectionResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source_id=self.source_id,
            frame_id=current_frame_id,
            is_anomaly=anomaly,
            anomaly_score=score,
            model_name=self.model_name,
            model_version=self.model_version,
            inference_time_ms=inference_time_ms,
            decision=decision,
            event_id=uuid.uuid4().hex,
        )

        if self.has_actions:
            statuses = self.action_dispatcher.execute_all(result.to_dict())
            if not all(statuses):
                self.logger.warning(
                    "One or more actions failed for inspection event %s",
                    result.event_id,
                )

        return result

    def process_batch(
        self,
        anomaly_scores: Any,
        classifications: Optional[Any] = None,
        inference_time_ms: Optional[float] = None,
        start_frame_id: Optional[int] = None,
    ) -> List[InspectionResult]:
        """Process a batch while preserving one event per image."""
        scores = np.asarray(anomaly_scores).reshape(-1)
        flags = None if classifications is None else np.asarray(classifications).reshape(-1)
        frame_id = start_frame_id
        results = []

        for index, score in enumerate(scores):
            flag = None if flags is None else flags[index]
            result = self.process(
                anomaly_score=score,
                is_anomaly=flag,
                inference_time_ms=inference_time_ms,
                frame_id=frame_id,
            )
            results.append(result)
            if frame_id is not None:
                frame_id += 1

        return results

    def disconnect(self) -> None:
        """Disconnect all actions and release external resources."""
        if self.has_actions:
            self.action_dispatcher.disconnect_all()
        self._connected = False
