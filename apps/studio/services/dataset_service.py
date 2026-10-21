"""Dataset discovery and quality inspection for AnomaVision Studio."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageStat

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(root: Path | str, recursive: bool = True) -> list[Path]:
    root = Path(root).expanduser()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_previews(root: Path) -> list[dict[str, str]]:
    """Return up to two small previews from the dataset's training split."""
    train_root = root / "train"
    if not train_root.is_dir():
        return []
    candidates = sorted(
        path for path in train_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    previews: list[dict[str, str]] = []
    for path in candidates[:2]:
        try:
            with Image.open(path) as image:
                image.thumbnail((320, 240))
                preview = image.convert("RGB")
                buffer = io.BytesIO()
                preview.save(buffer, format="JPEG", quality=78, optimize=True)
            previews.append({
                "name": path.name,
                "path": str(path.relative_to(root)),
                "data_url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            })
        except Exception:
            continue
    return previews


def _split_summary(root: Path) -> dict[str, Any] | None:
    """Summarize standard train/test class folders when present (e.g. MVTec AD)."""
    train_root, test_root = root / "train", root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        return None

    def count_images(folder: Path) -> int:
        return sum(
            1 for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    test_classes = [
        {"name": folder.name, "image_count": count_images(folder)}
        for folder in sorted(test_root.iterdir())
        if folder.is_dir()
    ]
    return {
        "training_count": count_images(train_root),
        "testing_count": count_images(test_root),
        "test_class_count": len(test_classes),
        "test_classes": test_classes,
        "training_previews": _training_previews(root),
    }


def inspect_dataset(
    root: Path | str,
    recursive: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    paths = discover_images(root, recursive=recursive)
    total = len(paths)
    if progress_callback:
        progress_callback(0, total, f"Found {total} image file(s); starting inspection")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    hashes: dict[str, list[str]] = {}

    for index, path in enumerate(paths, start=1):
        relative = str(path.relative_to(Path(root).expanduser()))
        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                stat = ImageStat.Stat(image.convert("L"))
                brightness = float(stat.mean[0])
            file_hash = _sha256(path)
            record = {
                "path": relative,
                "name": path.name,
                "width": width,
                "height": height,
                "mode": mode,
                "bytes": path.stat().st_size,
                "brightness": round(brightness, 2),
                "sha256": file_hash,
            }
            records.append(record)
            hashes.setdefault(file_hash, []).append(relative)
        except Exception as exc:
            failures.append({"path": relative, "error": str(exc)})
        finally:
            if progress_callback:
                progress_callback(index, total, f"Checked {index} of {total}: {path.name}")

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    resolutions: dict[str, int] = {}
    for record in records:
        key = f'{record["width"]} × {record["height"]}'
        resolutions[key] = resolutions.get(key, 0) + 1

    root_path = Path(root).expanduser()
    split_summary = _split_summary(root_path)
    return {
        "root": str(root_path),
        "split_summary": split_summary,
        "image_count": len(records),
        "valid_count": len(records),
        "failed_count": len(failures),
        "duplicate_count": sum(len(group) - 1 for group in duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "resolutions": dict(
            sorted(resolutions.items(), key=lambda item: item[1], reverse=True)
        ),
        "records": records,
        "failures": failures,
    }


def save_dataset_manifest(destination: Path | str, report: dict[str, Any]) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination
