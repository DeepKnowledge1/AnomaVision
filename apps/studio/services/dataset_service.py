"""Dataset discovery and quality inspection for AnomaVision Studio."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(root: Path | str, recursive: bool = True) -> list[Path]:
    root = Path(root).expanduser()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(root: Path | str, recursive: bool = True) -> dict[str, Any]:
    paths = discover_images(root, recursive=recursive)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    hashes: dict[str, list[str]] = {}

    for path in paths:
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

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    resolutions: dict[str, int] = {}
    for record in records:
        key = f'{record["width"]} × {record["height"]}'
        resolutions[key] = resolutions.get(key, 0) + 1

    return {
        "root": str(Path(root).expanduser()),
        "image_count": len(records),
        "valid_count": len(records),
        "failed_count": len(failures),
        "duplicate_count": sum(len(group) - 1 for group in duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "resolutions": dict(sorted(resolutions.items(), key=lambda item: item[1], reverse=True)),
        "records": records,
        "failures": failures,
    }


def save_dataset_manifest(destination: Path | str, report: dict[str, Any]) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination
