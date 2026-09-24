from pathlib import Path

import pytest

from apps.studio.services.training_service import (
    create_training_config,
    normalize_dataset_source,
)


def test_normalize_class_folder(tmp_path: Path):
    class_dir = tmp_path / "bottle"
    (class_dir / "train" / "good").mkdir(parents=True)

    dataset_path, class_name = normalize_dataset_source(str(class_dir))

    assert dataset_path == class_dir
    assert class_name == "bottle"


def test_normalize_train_good_folder(tmp_path: Path):
    good = tmp_path / "bottle" / "train" / "good"
    good.mkdir(parents=True)

    dataset_path, class_name = normalize_dataset_source(str(good))

    assert dataset_path == tmp_path
    assert class_name == "bottle"


def test_rejects_unstructured_folder(tmp_path: Path):
    (tmp_path / "images").mkdir()
    with pytest.raises(ValueError, match="train/good"):
        normalize_dataset_source(str(tmp_path / "images"))


def test_create_training_config(tmp_path: Path):
    class_dir = tmp_path / "bottle"
    (class_dir / "train" / "good").mkdir(parents=True)

    config = create_training_config(
        tmp_path / "project",
        str(class_dir),
        "patchcore",
        batch_size=4,
    )

    text = config.read_text(encoding="utf-8")
    assert "algorithm: patchcore" in text
    assert "batch_size: 4" in text
    assert "class_name: bottle" in text
