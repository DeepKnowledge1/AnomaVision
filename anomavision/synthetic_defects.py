"""Deterministic synthetic defect generation for demos and controlled testing.

The studio intentionally uses lightweight Pillow drawing primitives instead of a
large generative model. Every generated sample returns the modified image and an
exact binary mask describing the pixels changed by the synthetic defect.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _seed_value(seed: int | str) -> int:
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _severity_value(severity: str | float) -> float:
    if isinstance(severity, str):
        return {"low": 0.3, "medium": 0.55, "high": 0.85}.get(severity.lower(), 0.55)
    return float(np.clip(float(severity), 0.1, 1.0))


def _geometry(
    width: int, height: int, rng: np.random.Generator
) -> Tuple[int, int, int]:
    margin = max(8, min(width, height) // 8)
    cx = int(rng.integers(margin, max(margin + 1, width - margin)))
    cy = int(rng.integers(margin, max(margin + 1, height - margin)))
    radius = max(4, int(min(width, height) * rng.uniform(0.06, 0.16)))
    return cx, cy, radius


def _line_points(cx: int, cy: int, radius: int, rng: np.random.Generator):
    angle = float(rng.uniform(0, 2 * np.pi))
    length = radius * 2.5
    x0 = int(cx - np.cos(angle) * length)
    y0 = int(cy - np.sin(angle) * length)
    x1 = int(cx + np.cos(angle) * length)
    y1 = int(cy + np.sin(angle) * length)
    return [(x0, y0), (cx, cy), (x1, y1)]


def _blend_color(
    base: np.ndarray, color: Tuple[int, int, int], alpha: float
) -> np.ndarray:
    overlay = np.empty_like(base)
    overlay[...] = np.asarray(color, dtype=np.uint8)
    return np.clip(
        base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def generate_synthetic_defect(
    image: Image.Image | np.ndarray,
    defect_type: str = "scratch",
    severity: str | float = "medium",
    seed: int | str = 42,
) -> Tuple[Image.Image, Image.Image, Dict[str, object]]:
    """Apply one deterministic synthetic defect and return image, mask, metadata."""
    if isinstance(image, Image.Image):
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        source = np.asarray(image, dtype=np.uint8)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("image must be an RGB image")

    defect = defect_type.strip().lower().replace(" ", "_")
    supported = {"scratch", "crack", "stain", "dent", "hole", "cutpaste"}
    if defect not in supported:
        raise ValueError(f"unsupported defect_type: {defect_type}")

    strength = _severity_value(severity)
    rng = np.random.default_rng(_seed_value(seed))
    height, width = source.shape[:2]
    if min(width, height) < 16:
        raise ValueError("image must be at least 16x16 pixels")
    cx, cy, radius = _geometry(width, height, rng)
    mask_layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_layer)
    result = source.copy()

    if defect in {"scratch", "crack"}:
        width_px = max(1, int(1 + strength * min(width, height) / 45))
        points = _line_points(cx, cy, radius, rng)
        draw.line(points, fill=255, width=width_px + 2)
        if defect == "crack":
            for branch in range(2):
                offset = int((branch - 0.5) * radius * 0.7)
                branch_points = [
                    (cx, cy),
                    (cx + offset, cy - radius),
                    (cx + offset * 2, cy),
                ]
                draw.line(branch_points, fill=255, width=max(1, width_px))
        mask = np.asarray(mask_layer, dtype=np.uint8) > 0
        result = _blend_color(result, (20, 20, 20), 0.45 + 0.4 * strength)
        result[~mask] = source[~mask]

    elif defect == "stain":
        box = (cx - radius, cy - radius // 2, cx + radius, cy + radius // 2)
        draw.ellipse(box, fill=255)
        mask = np.asarray(mask_layer, dtype=np.uint8) > 0
        blurred = mask_layer.filter(ImageFilter.GaussianBlur(max(1, radius // 5)))
        alpha = np.asarray(blurred, dtype=np.float32) / 255.0
        alpha *= 0.35 + 0.45 * strength
        result = _blend_color(result, (102, 68, 42), alpha[..., None])

    elif defect == "dent":
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(box, fill=255)
        mask = np.asarray(mask_layer, dtype=np.uint8) > 0
        yy, xx = np.ogrid[:height, :width]
        distance = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(1, radius**2)
        dent_alpha = np.clip(1.0 - distance, 0.0, 1.0) * (0.25 + 0.5 * strength)
        result = _blend_color(result, (35, 35, 35), dent_alpha[..., None])
        rim = np.logical_and(mask, distance > 0.55)
        result[rim] = _blend_color(result[rim], (245, 245, 245), 0.25)

    elif defect == "cutpaste":
        patch_radius = max(4, int(radius * (0.7 + 0.8 * strength)))
        src_x = int(
            rng.integers(patch_radius, max(patch_radius + 1, width - patch_radius))
        )
        src_y = int(
            rng.integers(patch_radius, max(patch_radius + 1, height - patch_radius))
        )
        dst_x = int(
            rng.integers(patch_radius, max(patch_radius + 1, width - patch_radius))
        )
        dst_y = int(
            rng.integers(patch_radius, max(patch_radius + 1, height - patch_radius))
        )
        src_box = (
            src_x - patch_radius,
            src_y - patch_radius,
            src_x + patch_radius,
            src_y + patch_radius,
        )
        dst_box = (
            dst_x - patch_radius,
            dst_y - patch_radius,
            dst_x + patch_radius,
            dst_y + patch_radius,
        )
        patch = Image.fromarray(source).crop(src_box)
        draw.rectangle(dst_box, fill=255)
        pasted = np.asarray(patch.resize((2 * patch_radius, 2 * patch_radius))).astype(
            np.float32
        )
        pasted = np.clip(
            pasted * (1.0 - 0.25 * strength) + 255.0 * 0.1 * strength,
            0,
            255,
        ).astype(np.uint8)
        y0, y1 = dst_y - patch_radius, dst_y + patch_radius
        x0, x1 = dst_x - patch_radius, dst_x + patch_radius
        alpha = 0.35 + 0.55 * strength
        result[y0:y1, x0:x1] = np.clip(
            result[y0:y1, x0:x1].astype(np.float32) * (1.0 - alpha)
            + pasted.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

    else:  # hole
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(box, fill=255)
        mask = np.asarray(mask_layer, dtype=np.uint8) > 0
        result[mask] = _blend_color(result[mask], (8, 8, 8), 0.8)
        ring = Image.new("L", (width, height), 0)
        ImageDraw.Draw(ring).ellipse(box, outline=255, width=max(1, radius // 8))
        ring_mask = np.asarray(ring, dtype=np.uint8) > 0
        result[ring_mask] = _blend_color(result[ring_mask], (220, 220, 220), 0.35)

    mask_image = mask_layer.convert("L")
    metadata: Dict[str, object] = {
        "defect_type": defect,
        "severity": severity,
        "seed": seed,
        "image_size": [width, height],
        "mask_area_percent": round(float(np.asarray(mask_image).mean() / 255 * 100), 3),
    }
    return Image.fromarray(result, mode="RGB"), mask_image, metadata


def save_studio_outputs(
    defective: Image.Image, mask: Image.Image, metadata: Dict[str, object]
) -> list[str]:
    """Save a generated sample and return paths suitable for a UI file output."""
    output_dir = Path(tempfile.mkdtemp(prefix="anomavision_synthetic_"))
    defect_path = output_dir / "synthetic_defect.png"
    mask_path = output_dir / "ground_truth_mask.png"
    metadata_path = output_dir / "metadata.json"
    defective.save(defect_path)
    mask.save(mask_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return [str(defect_path), str(mask_path), str(metadata_path)]


def generate_synthetic_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    defect_types: list[str] | tuple[str, ...] = (
        "scratch",
        "crack",
        "stain",
        "dent",
        "hole",
        "cutpaste",
    ),
    severity: str = "medium",
    copies_per_type: int = 1,
    seed: int = 42,
    val_ratio: float = 0.2,
    max_samples: int = 10_000,
) -> Dict[str, object]:
    """Create a reproducible normal/anomaly dataset with masks and manifest.

    The output layout is ``images/{train,val}/{normal,anomaly}`` and
    ``masks/{train,val}/{normal,anomaly}``. Normal masks are all-black; anomaly
    masks are exact binary masks generated with the synthetic image.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise ValueError(f"input_dir does not exist or is not a directory: {input_dir}")
    if copies_per_type < 1 or copies_per_type > 100:
        raise ValueError("copies_per_type must be between 1 and 100")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in the range [0, 1)")
    normalized_types = [item.strip().lower().replace(" ", "_") for item in defect_types]
    supported = {"scratch", "crack", "stain", "dent", "hole", "cutpaste"}
    unknown = set(normalized_types) - supported
    if unknown:
        raise ValueError(f"unsupported defect types: {sorted(unknown)}")

    source_files = sorted(
        path
        for path in input_path.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    if not source_files:
        raise ValueError(f"no supported images found in {input_dir}")

    output_path.mkdir(parents=True, exist_ok=True)
    records: list[Dict[str, object]] = []
    sample_count = 0
    rng = np.random.default_rng(_seed_value(seed))

    for source_index, source_file in enumerate(source_files):
        split = "val" if rng.random() < val_ratio else "train"
        source = Image.open(source_file).convert("RGB")
        image_id = f"{source_index:06d}_{source_file.stem}"
        normal_rel = Path("images") / split / "normal" / f"{image_id}.png"
        normal_mask_rel = Path("masks") / split / "normal" / f"{image_id}.png"
        normal_path = output_path / normal_rel
        normal_mask_path = output_path / normal_mask_rel
        normal_path.parent.mkdir(parents=True, exist_ok=True)
        normal_mask_path.parent.mkdir(parents=True, exist_ok=True)
        source.save(normal_path)
        Image.new("L", source.size, 0).save(normal_mask_path)
        records.append(
            {
                "image": normal_rel.as_posix(),
                "mask": normal_mask_rel.as_posix(),
                "label": "normal",
                "defect_type": None,
                "severity": None,
                "seed": seed,
                "source": str(source_file),
            }
        )

        for defect_type in normalized_types:
            for copy_index in range(copies_per_type):
                sample_count += 1
                if sample_count > max_samples:
                    raise ValueError(
                        f"sample limit exceeded ({max_samples}); reduce copies_per_type or input size"
                    )
                sample_seed = int(seed) + source_index * 100_003 + copy_index * 1_009
                defective, mask, metadata = generate_synthetic_defect(
                    source,
                    defect_type=defect_type,
                    severity=severity,
                    seed=sample_seed,
                )
                defect_id = f"{image_id}_{defect_type}_{copy_index:03d}"
                image_rel = Path("images") / split / "anomaly" / f"{defect_id}.png"
                mask_rel = Path("masks") / split / "anomaly" / f"{defect_id}.png"
                image_path = output_path / image_rel
                mask_path = output_path / mask_rel
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                defective.save(image_path)
                mask.save(mask_path)
                records.append(
                    {
                        "image": image_rel.as_posix(),
                        "mask": mask_rel.as_posix(),
                        "label": "anomaly",
                        "source": str(source_file),
                        **metadata,
                    }
                )

    manifest_path = output_path / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "format": "anomavision-synthetic-v1",
        "source_dir": str(input_path),
        "samples": len(records),
        "normal_samples": sum(record["label"] == "normal" for record in records),
        "anomaly_samples": sum(record["label"] == "anomaly" for record in records),
        "defect_types": normalized_types,
        "severity": severity,
        "seed": seed,
        "val_ratio": val_ratio,
        "manifest": "manifest.jsonl",
    }
    (output_path / "dataset_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
