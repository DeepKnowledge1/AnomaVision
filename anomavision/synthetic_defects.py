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
        return {"low": 0.3, "medium": 0.55, "high": 0.85}.get(
            severity.lower(), 0.55
        )
    return float(np.clip(float(severity), 0.1, 1.0))


def _geometry(width: int, height: int, rng: np.random.Generator) -> Tuple[int, int, int]:
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


def _blend_color(base: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
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
    supported = {"scratch", "crack", "stain", "dent", "hole"}
    if defect not in supported:
        raise ValueError(f"unsupported defect_type: {defect_type}")

    strength = _severity_value(severity)
    rng = np.random.default_rng(_seed_value(seed))
    height, width = source.shape[:2]
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
                branch_points = [(cx, cy), (cx + offset, cy - radius), (cx + offset * 2, cy)]
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
