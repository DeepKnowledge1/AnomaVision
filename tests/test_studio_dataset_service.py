from pathlib import Path

from PIL import Image

from apps.studio.services.dataset_service import inspect_dataset


def test_inspect_dataset_reports_images_and_duplicates(tmp_path: Path):
    image = Image.new("RGB", (640, 480), "white")
    image.save(tmp_path / "one.png")
    (tmp_path / "copy.png").write_bytes((tmp_path / "one.png").read_bytes())

    report = inspect_dataset(tmp_path)

    assert report["image_count"] == 2
    assert report["resolutions"]["640 × 480"] == 2
    assert report["duplicate_count"] == 1
    assert report["failed_count"] == 0


def test_inspect_dataset_reports_invalid_image(tmp_path: Path):
    (tmp_path / "broken.jpg").write_bytes(b"not an image")

    report = inspect_dataset(tmp_path)

    assert report["image_count"] == 0
    assert report["failed_count"] == 1
