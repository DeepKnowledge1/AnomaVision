import numpy as np
import pytest

from anomavision.drift_runtime import InferenceDriftRuntime, input_drift_features
from anomavision.production_monitor import ProductionDriftMonitor


class FakeModel:
    def _extract(self, batch):
        # B x patches x features, matching the PatchCore extraction contract.
        return batch, 1, 1


class NumpyModel:
    def _extract(self, batch):
        # Exercise backends/tests that expose NumPy features instead of tensors.
        return np.asarray(batch), 1, 1


class NoEmbeddingModel:
    def extract_drift_embeddings(self, batch):
        raise NotImplementedError


def test_monitor_warms_up_then_evaluates():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, size=(200, 4))
    monitor = ProductionDriftMonitor(
        reference, window_size=50, min_samples=10, evaluation_interval=10
    )
    assert monitor.update(rng.normal(0, 1, size=(5, 4))) is None
    report = monitor.update(rng.normal(0, 1, size=(5, 4)))
    assert report is not None
    assert report.status == "stable"
    assert monitor.status().ready is True


def test_monitor_detects_shift_and_bounds_window():
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 1, size=(300, 3))
    monitor = ProductionDriftMonitor(
        reference,
        window_size=20,
        min_samples=10,
        evaluation_interval=10,
        threshold=0.20,
    )
    report = monitor.update(rng.normal(4, 1, size=(30, 3)))
    assert report is not None
    assert report.status == "drift"
    assert monitor.status().window_fill == 20
    assert monitor.status().samples_seen == 30


def test_monitor_reset():
    rng = np.random.default_rng(1)
    monitor = ProductionDriftMonitor(
        rng.normal(size=(50, 2)), min_samples=5, window_size=10
    )
    monitor.update(rng.normal(size=(5, 2)))
    monitor.reset()
    status = monitor.status()
    assert status.samples_seen == 0
    assert status.window_fill == 0
    assert status.status == "warming_up"


def test_runtime_uses_model_representation_and_returns_report():
    rng = np.random.default_rng(9)
    reference = rng.normal(size=(100, 3))
    monitor = ProductionDriftMonitor(
        reference, window_size=20, min_samples=10, evaluation_interval=10
    )
    runtime = InferenceDriftRuntime(monitor, FakeModel())

    assert runtime.update(rng.normal(size=(5, 1, 3))) is None
    result = runtime.update(rng.normal(size=(5, 1, 3)))
    assert result is not None
    assert result["status"] in {"stable", "drift"}


def test_runtime_accepts_numpy_feature_extractor():
    rng = np.random.default_rng(10)
    reference = rng.normal(size=(30, 3))
    monitor = ProductionDriftMonitor(
        reference, window_size=10, min_samples=5, evaluation_interval=5
    )
    runtime = InferenceDriftRuntime(monitor, NumpyModel())

    result = runtime.update(rng.normal(size=(5, 1, 3)))
    assert result is not None
    assert result["feature_dimensions"] == 3


def test_runtime_falls_back_to_input_statistics_without_model_embeddings():
    batch = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    features = input_drift_features(batch)
    assert features.shape == (2, 15)
    assert np.isfinite(features).all()

    reference = np.repeat(features, 5, axis=0)
    monitor = ProductionDriftMonitor(
        reference, window_size=10, min_samples=5, evaluation_interval=5
    )
    runtime = InferenceDriftRuntime(monitor, NoEmbeddingModel())
    result = runtime.update(
        batch[:5] if batch.shape[0] >= 5 else np.tile(batch, (3, 1, 1, 1))
    )
    assert result is not None
    assert result["feature_dimensions"] == 15


def test_runtime_rejects_models_without_feature_extractor():
    rng = np.random.default_rng(3)
    monitor = ProductionDriftMonitor(
        rng.normal(size=(20, 2)), min_samples=2, window_size=5
    )
    runtime = InferenceDriftRuntime(monitor, object())
    with pytest.raises(ValueError, match="image batch"):
        runtime.update(np.zeros((2, 2)))


def test_monitor_rejects_invalid_production_embeddings():
    monitor = ProductionDriftMonitor(np.ones((20, 4)), min_samples=2, window_size=5)
    with pytest.raises(ValueError, match="embedding dimension"):
        monitor.update(np.ones((2, 3)))
    with pytest.raises(ValueError, match="NaN"):
        monitor.update(np.full((2, 4), np.nan))
