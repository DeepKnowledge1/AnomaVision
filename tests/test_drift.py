import json

import numpy as np
import pytest

from anomavision.drift import DriftMonitor, load_embeddings, save_report


def test_identical_distributions_are_stable():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(500, 8))
    report = DriftMonitor(reference).compare(reference.copy())

    assert report.status == "stable"
    assert report.psi < 1e-9
    assert report.mean_shift < 1e-9
    assert report.std_shift < 1e-9
    assert report.cosine_shift < 1e-9


def test_shifted_distribution_is_detected():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(500, 8))
    current = rng.normal(loc=2.0, scale=1.0, size=(500, 8))
    report = DriftMonitor(reference).compare(current)

    assert report.status == "drift"
    assert report.psi >= report.threshold
    assert "feature_distribution_shift" in report.warnings


def test_dimension_mismatch_is_rejected():
    monitor = DriftMonitor(np.ones((10, 4)))
    with pytest.raises(ValueError, match="feature dimension"):
        monitor.compare(np.ones((10, 3)))


def test_report_round_trip(tmp_path):
    reference = np.arange(60, dtype=float).reshape(20, 3)
    current = reference + 0.01
    report = DriftMonitor(reference).compare(current)
    path = tmp_path / "drift.json"
    save_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == report.status
    assert payload["feature_dimensions"] == 3


def test_embedding_loaders(tmp_path):
    values = np.arange(20, dtype=float).reshape(10, 2)
    npy = tmp_path / "features.npy"
    npz = tmp_path / "features.npz"
    np.save(npy, values)
    np.savez(npz, embeddings=values)

    np.testing.assert_array_equal(load_embeddings(npy), values)
    np.testing.assert_array_equal(load_embeddings(npz), values)
