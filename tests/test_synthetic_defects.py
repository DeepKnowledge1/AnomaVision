"""Tests for the lightweight Synthetic Defect Studio engine."""

import numpy as np
import pytest
from PIL import Image

from anomavision.synthetic_defects import (
    generate_synthetic_dataset,
    generate_synthetic_defect,
)


def _normal_image() -> Image.Image:
    pixels = np.full((64, 64, 3), 128, dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_generation_is_deterministic_and_returns_exact_mask():
    first = generate_synthetic_defect(_normal_image(), "scratch", "medium", seed=7)
    second = generate_synthetic_defect(_normal_image(), "scratch", "medium", seed=7)

    assert np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
    assert np.array_equal(np.asarray(first[1]), np.asarray(second[1]))
    assert first[2] == second[2]
    assert first[1].mode == "L"
    assert np.asarray(first[1]).shape == (64, 64)
    assert np.asarray(first[1]).max() == 255


@pytest.mark.parametrize(
    "defect_type", ["scratch", "crack", "stain", "dent", "hole", "cutpaste"]
)
def test_all_defect_types_change_pixels_and_produce_mask(defect_type):
    source = np.asarray(_normal_image())
    defective, mask, metadata = generate_synthetic_defect(
        source, defect_type=defect_type, severity="high", seed=12
    )

    assert metadata["defect_type"] == defect_type
    assert np.asarray(mask).sum() > 0
    assert not np.array_equal(source, np.asarray(defective))


def test_invalid_defect_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported defect_type"):
        generate_synthetic_defect(_normal_image(), defect_type="rust")


def test_dataset_export_writes_manifest_and_normal_masks(tmp_path):
    source_dir = tmp_path / "normal"
    source_dir.mkdir()
    _normal_image().save(source_dir / "part.png")

    summary = generate_synthetic_dataset(
        source_dir,
        tmp_path / "dataset",
        defect_types=["scratch", "cutpaste"],
        copies_per_type=1,
        val_ratio=0.0,
        seed=4,
    )

    assert summary["normal_samples"] == 1
    assert summary["anomaly_samples"] == 2
    manifest = (tmp_path / "dataset" / "manifest.jsonl").read_text()
    assert manifest.count("\n") == 3
    normal_mask = next((tmp_path / "dataset" / "masks").rglob("normal/*.png"))
    assert np.asarray(Image.open(normal_mask)).max() == 0
