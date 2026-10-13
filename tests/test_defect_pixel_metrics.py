"""Strict defect-pixel count tests for anomaly maps.

Unit tests verify exact threshold semantics. The integration test runs the trained
PaDiM fixture through both PyTorch and ONNX and checks the resulting pixel masks
and counts at a fixed threshold.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from anomavision.export import ModelExporter
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.utils import get_logger


LOGGER = get_logger(__name__)
DEFECT_THRESHOLD = 0.5


def count_defect_pixels(anomaly_map: np.ndarray, threshold: float) -> int:
    """Count pixels strictly above the configured anomaly threshold."""
    anomaly_map = np.asarray(anomaly_map)
    if anomaly_map.ndim != 2:
        raise ValueError(f"Expected a 2D anomaly map, got shape {anomaly_map.shape}")
    if not np.isfinite(anomaly_map).all():
        raise ValueError("Anomaly map contains non-finite values.")
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    return int(np.count_nonzero(anomaly_map > threshold))


def test_exact_defect_pixel_count_for_known_region():
    anomaly_map = np.zeros((32, 32), dtype=np.float32)
    anomaly_map[5:15, 8:18] = 0.9

    assert count_defect_pixels(anomaly_map, threshold=0.5) == 100


def test_pixels_equal_to_threshold_are_not_counted_as_defects():
    anomaly_map = np.array([[0.0, 0.5, 0.51, 0.9]], dtype=np.float32)

    assert count_defect_pixels(anomaly_map, threshold=0.5) == 2


def test_normal_map_has_zero_defect_pixels():
    anomaly_map = np.full((224, 224), 0.1, dtype=np.float32)

    assert count_defect_pixels(anomaly_map, threshold=0.5) == 0


def test_non_finite_anomaly_map_fails_closed():
    anomaly_map = np.array([[0.1, np.nan]], dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        count_defect_pixels(anomaly_map, threshold=0.5)


def test_padim_pt_and_onnx_defect_pixel_counts_match(
    tmp_path: Path, trained_padim_model, sample_batch
):
    """Compare pixel-level defect decisions from real PaDiM inference backends."""
    batch, _, _, _ = sample_batch
    batch = batch.cpu()
    assert batch.ndim == 4 and batch.shape[0] > 0

    pt_path = tmp_path / "padim_pixel_count.pt"
    torch.save(trained_padim_model.eval(), pt_path)
    exporter = ModelExporter(pt_path, tmp_path, LOGGER, device="cpu")
    onnx_path = exporter.export_onnx(
        input_shape=tuple(batch.shape),
        output_name="padim_pixel_count.onnx",
        opset_version=18,
        dynamic_batch=False,
        force_precision="fp32",
    )
    assert onnx_path is not None and onnx_path.is_file()
    onnx.checker.check_model(str(onnx_path))

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    assert session.get_providers()[0] == "CPUExecutionProvider"

    pt_wrapper = ModelWrapper(str(pt_path), device="cpu")
    onnx_wrapper = ModelWrapper(str(onnx_path), device="cpu")
    try:
        _, pt_maps = pt_wrapper.predict(batch)
        _, onnx_maps = onnx_wrapper.predict(batch)

        assert pt_maps.shape == onnx_maps.shape
        assert pt_maps.shape == (
            batch.shape[0],
            batch.shape[2],
            batch.shape[3],
        )
        assert np.isfinite(pt_maps).all()
        assert np.isfinite(onnx_maps).all()

        # Pixel counts are computed per image, not pooled across the batch.
        pt_masks = pt_maps > DEFECT_THRESHOLD
        onnx_masks = onnx_maps > DEFECT_THRESHOLD
        pt_counts = np.count_nonzero(pt_masks, axis=(1, 2))
        onnx_counts = np.count_nonzero(onnx_masks, axis=(1, 2))

        np.testing.assert_array_equal(
            onnx_counts,
            pt_counts,
            err_msg=(
                "ONNX and PyTorch disagree on the number of defect pixels "
                f"at threshold {DEFECT_THRESHOLD}"
            ),
        )
        np.testing.assert_array_equal(
            onnx_masks,
            pt_masks,
            err_msg=(
                "ONNX and PyTorch produce different pixel-level defect masks "
                f"at threshold {DEFECT_THRESHOLD}"
            ),
        )

        assert np.all(pt_counts >= 0)
        assert np.all(pt_counts <= batch.shape[2] * batch.shape[3])
    finally:
        pt_wrapper.close()
        onnx_wrapper.close()
