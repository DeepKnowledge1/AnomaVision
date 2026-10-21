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

    # Dataset quality signals are advisory; they never change model behavior.
    image_records = [
        item for item in records
        if not any(part.lower() in {"ground_truth", "masks", "mask"} for part in Path(item["path"]).parts)
    ]
    train_hashes: dict[str, list[str]] = {}
    test_hashes: dict[str, list[str]] = {}
    for item in image_records:
        parts = [part.lower() for part in Path(item["path"]).parts]
        if "train" in parts:
            train_hashes.setdefault(item["sha256"], []).append(item["path"])
        if "test" in parts:
            test_hashes.setdefault(item["sha256"], []).append(item["path"])
    leakage_groups = [
        {"train": train_paths, "test": test_hashes[digest]}
        for digest, train_paths in train_hashes.items()
        if digest in test_hashes
    ]

    test_root = root_path / "test"
    ground_truth_root = root_path / "ground_truth"
    defect_test_images: list[Path] = []
    missing_masks: list[str] = []
    empty_masks: list[str] = []
    invalid_mask_labels: list[str] = []
    mask_check_available = test_root.is_dir() and any(
        child.is_dir() and child.name.lower() not in {"good", "normal"}
        for child in test_root.iterdir()
    )
    if test_root.is_dir():
        for class_dir in sorted(test_root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.lower() in {"good", "normal"}:
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                defect_test_images.append(image_path)
                mask_dir = ground_truth_root / class_dir.name
                candidates = [
                    mask_dir / f"{image_path.stem}_mask{suffix}"
                    for suffix in IMAGE_EXTENSIONS
                ]
                mask_path = next((candidate for candidate in candidates if candidate.is_file()), None)
                if mask_path is None:
                    if mask_check_available:
                        missing_masks.append(str(image_path.relative_to(root_path)))
                    continue
                try:
                    with Image.open(mask_path) as mask:
                        mask.load()
                        mask_values = mask.convert("L")
                        if mask_values.getbbox() is None:
                            empty_masks.append(str(mask_path.relative_to(root_path)))
                        values = set(mask_values.getdata())
                        if not values.issubset({0, 1, 255}):
                            invalid_mask_labels.append(str(mask_path.relative_to(root_path)))
                except Exception:
                    invalid_mask_labels.append(str(mask_path.relative_to(root_path)))

    dimensions = [(item["width"], item["height"]) for item in image_records]
    modes: dict[str, int] = {}
    for item in image_records:
        modes[item["mode"]] = modes.get(item["mode"], 0) + 1
    dimension_summary = {
        "min_width": min((w for w, _ in dimensions), default=None),
        "max_width": max((w for w, _ in dimensions), default=None),
        "avg_width": round(sum(w for w, _ in dimensions) / len(dimensions), 1) if dimensions else None,
        "min_height": min((h for _, h in dimensions), default=None),
        "max_height": max((h for _, h in dimensions), default=None),
        "avg_height": round(sum(h for _, h in dimensions) / len(dimensions), 1) if dimensions else None,
    }
    image_hashes: dict[str, int] = {}
    for item in image_records:
        image_hashes[item["sha256"]] = image_hashes.get(item["sha256"], 0) + 1
    duplicate_image_count = sum(count - 1 for count in image_hashes.values() if count > 1)
    test_classes = split_summary["test_classes"] if split_summary else []
    defect_classes = [item for item in test_classes if item["name"].lower() not in {"good", "normal"}]
    sparse_classes = [item for item in defect_classes if item["image_count"] < 10]
    normal_count = sum(item["image_count"] for item in test_classes if item["name"].lower() in {"good", "normal"})
    defect_count = sum(item["image_count"] for item in defect_classes)
    quality_summary = {
        "dataset_class": root_path.name,
        "total_images": len(image_records),
        "normal_test_images": normal_count,
        "defect_test_images": defect_count,
        "defect_categories": len(defect_classes),
        "corrupted_images": len(failures),
        "duplicate_images": duplicate_image_count,
        "train_test_leakage_groups": len(leakage_groups),
        "leakage_groups": leakage_groups[:10],
        "mask_check_available": mask_check_available,
        "missing_masks": len(missing_masks),
        "missing_mask_files": missing_masks[:10],
        "empty_masks": len(empty_masks),
        "empty_mask_files": empty_masks[:10],
        "invalid_masks": len(invalid_mask_labels),
        "invalid_mask_files": invalid_mask_labels[:10],
        "image_modes": modes,
        "image_dimensions": dimension_summary,
        "sparse_defect_classes": sparse_classes,
        "preprocessing_compatible": all(item["width"] > 0 and item["height"] > 0 for item in image_records),
        "preprocessing_note": "Images are readable and can be resized; model-specific normalization is applied by the existing pipeline.",
        "near_duplicates_checked": False,
    }
    return {
        "root": str(root_path),
        "split_summary": split_summary,
        "quality_summary": quality_summary,
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
