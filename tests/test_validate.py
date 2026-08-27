"""Unit tests for cross-backend validation metrics."""

import numpy as np

from anomavision.validate import _compare_pair, _relative_error


def test_relative_error_handles_zero_reference():
    """Relative error should remain finite when the reference is zero."""
    assert _relative_error(0.0, 0.0) == 0.0
    assert np.isfinite(_relative_error(1.0, 0.0))


def test_compare_pair_passes_for_identical_outputs():
    """Identical scores and maps must pass validation."""
    scores = np.array([0.1, 0.9], dtype=np.float32)
    maps = np.array(
        [
            [[0.0, 0.1], [0.2, 0.3]],
            [[0.4, 0.5], [0.6, 0.7]],
        ],
        dtype=np.float32,
    )

    result = _compare_pair(
        "pt",
        "onnx",
        scores,
        scores.copy(),
        maps,
        maps.copy(),
        1e-3,
        1e-2,
        1e-3,
        1e-2,
    )

    assert result["pass"] is True
    assert result["score"]["max_absolute_error"] == 0.0
    assert result["heatmap"]["mae"] == 0.0


def test_compare_pair_fails_outside_tolerance():
    """Large score and heatmap differences must fail validation."""
    scores_a = np.array([0.1, 0.9], dtype=np.float32)
    scores_b = np.array([0.2, 1.1], dtype=np.float32)
    maps_a = np.zeros((2, 2, 2), dtype=np.float32)
    maps_b = np.ones((2, 2, 2), dtype=np.float32)

    result = _compare_pair(
        "pt",
        "onnx",
        scores_a,
        scores_b,
        maps_a,
        maps_b,
        1e-3,
        1e-2,
        1e-3,
        1e-2,
    )

    assert result["pass"] is False
    assert result["score"]["pass"] is False
    assert result["heatmap"]["pass"] is False
