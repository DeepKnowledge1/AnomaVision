import numpy as np

from anomavision.inspection.InspectionPipeline import InspectionPipeline


class RecordingAction:
    def __init__(self):
        self.connected = False
        self.results = []

    def connect(self):
        self.connected = True
        return True

    def execute(self, result):
        self.results.append(result)
        return True

    def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected


def test_process_creates_result_and_dispatches_action():
    action = RecordingAction()
    pipeline = InspectionPipeline(
        actions=[action],
        threshold=0.8,
        source_id="camera-01",
        model_name="padim",
    )

    pipeline.connect()
    result = pipeline.process(0.91, inference_time_ms=12.5)
    pipeline.disconnect()

    assert result.decision == "FAIL"
    assert result.is_anomaly is True
    assert result.anomaly_score == 0.91
    assert result.source_id == "camera-01"
    assert result.model_name == "padim"
    assert result.event_id
    assert len(action.results) == 1
    assert action.results[0]["decision"] == "FAIL"
    assert action.connected is False


def test_process_batch_creates_one_event_per_score():
    action = RecordingAction()
    pipeline = InspectionPipeline(actions=[action], threshold=0.5)
    pipeline.connect()

    results = pipeline.process_batch(
        np.array([0.1, 0.8, 0.2]),
        inference_time_ms=10.0,
        start_frame_id=100,
    )

    assert [result.frame_id for result in results] == [100, 101, 102]
    assert [result.decision for result in results] == ["PASS", "FAIL", "PASS"]
    assert len(action.results) == 3

    pipeline.disconnect()


def test_pipeline_without_actions_still_produces_results():
    pipeline = InspectionPipeline(threshold=0.5)
    result = pipeline.process(0.2)

    assert result.decision == "PASS"
    assert result.is_anomaly is False
