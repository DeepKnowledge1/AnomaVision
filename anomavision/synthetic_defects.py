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
        values = {"low": 0.3, "medium": 0.55, "high": 0.85}
        normalized = severity.strip().lower()
        if normalized not in values:
            raise ValueError("severity must be low, medium, or high")
        return values[normalized]
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


def _surface_aware_realism(
    source: np.ndarray,
    rendered: np.ndarray,
    mask_layer: Image.Image,
    strength: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Image.Image]:
    """Soften synthetic edges and modulate appearance by the host surface.

    The pass is deliberately deterministic and CPU-only. It avoids flat painted
    colors by preserving local luminance/texture and introduces imperfect,
    sub-pixel boundaries typical of industrial camera images.
    """
    height, width = source.shape[:2]
    hard = np.asarray(mask_layer, dtype=np.float32) / 255.0
    grid_h = max(4, height // 18)
    grid_w = max(4, width // 18)
    noise_small = rng.random((grid_h, grid_w), dtype=np.float32)
    noise = (
        np.asarray(
            Image.fromarray((noise_small * 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.BICUBIC
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    organic = np.clip(hard * (0.72 + 0.56 * noise), 0.0, 1.0)
    organic = (
        np.asarray(
            Image.fromarray((organic * 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(max(1, int(min(width, height) * 0.004)))
            ),
            dtype=np.float32,
        )
        / 255.0
    )

    source_float = source.astype(np.float32)
    rendered_float = rendered.astype(np.float32)
    surface = np.asarray(
        Image.fromarray(source).filter(
            ImageFilter.GaussianBlur(max(2, min(width, height) // 40))
        ),
        dtype=np.float32,
    )
    local_luma = source_float.mean(axis=2) - surface.mean(axis=2)
    delta = rendered_float - source_float
    texture_gain = 0.82 + 0.18 * np.clip(1.0 - local_luma / 64.0, 0.65, 1.35)
    blended = source_float + delta * organic[..., None] * texture_gain[..., None]

    # A faint, irregular surrounding response prevents sticker-like edges while
    # keeping the exact binary mask available for training annotations.
    halo = (
        np.asarray(
            Image.fromarray((organic * 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(max(1, int(min(width, height) * 0.012)))
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    halo = np.clip(halo - organic * 0.65, 0.0, 1.0)
    blended += halo[..., None] * (2.0 + 5.0 * strength)
    # Keep annotation masks binary even though the internal alpha is soft.
    binary_mask = (organic >= 0.5).astype(np.uint8) * 255
    return np.clip(blended, 0, 255).astype(np.uint8), Image.fromarray(
        binary_mask, mode="L"
    )


def _match_patch_to_surface(
    patch: Image.Image, target_region: np.ndarray, strength: float
) -> Image.Image:
    """Match patch luminance statistics to its host while preserving defect contrast."""
    patch_array = np.asarray(patch.convert("RGB"), dtype=np.float32)
    target = np.asarray(target_region, dtype=np.float32)
    patch_luma = patch_array.mean(axis=2)
    target_luma = target.mean(axis=2)
    patch_mean, target_mean = patch_luma.mean(), target_luma.mean()
    patch_std = max(2.0, patch_luma.std())
    target_std = max(2.0, target_luma.std())
    contrast = np.clip(target_std / patch_std, 0.65, 1.35)
    # Match only part-way: full histogram matching can erase the defect signal.
    adjusted = (patch_array - patch_mean) * (0.35 + 0.35 * strength) * contrast
    adjusted += patch_mean * 0.35 + target_mean * 0.65
    return Image.fromarray(np.clip(adjusted, 0, 255).astype(np.uint8), mode="RGB")


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

    result, mask_image = _surface_aware_realism(
        source, result, mask_layer, strength, rng
    )
    metadata: Dict[str, object] = {
        "defect_type": defect,
        "severity": severity,
        "seed": seed,
        "image_size": [width, height],
        "mask_area_percent": round(float(np.asarray(mask_image).mean() / 255 * 100), 3),
        "synthesis_profile": "surface_aware_v2",
        "characteristics": [
            "irregular_geometry",
            "local_texture_modulation",
            "soft_boundary",
            "subtle_context_cues",
        ],
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


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must be an RGB image")
    return array


def extract_defect_mask(
    defective_image: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray | None = None,
    *,
    sensitivity: float = 0.18,
) -> Image.Image:
    """Return a defect mask, preferring an uploaded mask over heuristic extraction.

    When no mask is supplied, a lightweight local-contrast heuristic is used. For
    production-quality annotations, upload a paired binary mask because a single
    defective image cannot always distinguish a defect from normal object edges.
    """
    image = _as_rgb_array(defective_image)
    height, width = image.shape[:2]
    if not 0.0 < float(sensitivity) <= 1.0:
        raise ValueError("sensitivity must be in the range (0, 1]")
    if mask is not None:
        mask_array = np.asarray(
            mask.convert("L") if isinstance(mask, Image.Image) else mask,
            dtype=np.uint8,
        )
        if mask_array.ndim != 2:
            raise ValueError("defect mask must be a 2D grayscale image")
        if mask_array.shape != (height, width):
            mask_array = np.asarray(
                Image.fromarray(mask_array).resize(
                    (width, height), Image.Resampling.NEAREST
                )
            )
        return Image.fromarray(
            np.where(mask_array > 0, 255, 0).astype(np.uint8), mode="L"
        )

    gray = image.astype(np.float32).mean(axis=2)
    blurred = np.asarray(
        Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(max(2, min(width, height) // 24))
        ),
        dtype=np.float32,
    )
    residual = np.abs(gray - blurred)
    threshold = max(4.0, float(np.percentile(residual, 100.0 - 100.0 * sensitivity)))
    extracted = residual >= threshold
    extracted = _remove_small_components(
        extracted, min_pixels=max(4, (width * height) // 5000)
    )
    return Image.fromarray((extracted * 255).astype(np.uint8), mode="L")


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove disconnected regions smaller than ``min_pixels`` without OpenCV."""
    if not mask.any():
        return mask
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)
    for start_y, start_x in zip(*np.where(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        component = []
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(component) >= min_pixels:
            ys, xs = zip(*component)
            kept[np.asarray(ys), np.asarray(xs)] = True
    return kept


def reuse_real_defects(
    normal_image: Image.Image | np.ndarray,
    defect_images: list[Image.Image | np.ndarray],
    defect_masks: list[Image.Image | np.ndarray | None] | None = None,
    *,
    copies_per_source: int = 1,
    scale_range: tuple[float, float] = (0.8, 1.2),
    rotation_range: tuple[float, float] = (-15.0, 15.0),
    seed: int = 42,
    sensitivity: float = 0.18,
) -> Tuple[Image.Image, Image.Image, Dict[str, object]]:
    """Place real defect appearances at reproducible random locations.

    Each reference defect is cropped to its mask, transformed, and alpha-composited
    onto the normal target. The returned mask is the union of all placed defects.
    """
    if not defect_images:
        raise ValueError("at least one defect image is required")
    if copies_per_source < 1 or copies_per_source > 20:
        raise ValueError("copies_per_source must be between 1 and 20")
    low_scale, high_scale = sorted((float(scale_range[0]), float(scale_range[1])))
    if low_scale <= 0 or high_scale > 4:
        raise ValueError("scale_range must be positive and no greater than 4.0")
    low_angle, high_angle = sorted((float(rotation_range[0]), float(rotation_range[1])))
    if high_angle - low_angle > 180:
        raise ValueError("rotation_range must span at most 180 degrees")

    target = Image.fromarray(_as_rgb_array(normal_image), mode="RGB")
    width, height = target.size
    result = target.copy()
    combined_mask = Image.new("L", target.size, 0)
    rng = np.random.default_rng(_seed_value(seed))
    masks = defect_masks or []
    placements = []

    for source_index, defect_image in enumerate(defect_images):
        defect = Image.fromarray(_as_rgb_array(defect_image), mode="RGB")
        provided_mask = masks[source_index] if source_index < len(masks) else None
        extracted_mask = extract_defect_mask(
            defect, provided_mask, sensitivity=sensitivity
        )
        bbox = extracted_mask.getbbox()
        if bbox is None:
            raise ValueError(
                f"no defect pixels found in reference image {source_index}"
            )
        patch = defect.crop(bbox)
        alpha = extracted_mask.crop(bbox)

        for copy_index in range(copies_per_source):
            scale = float(rng.uniform(low_scale, high_scale))
            angle = float(rng.uniform(low_angle, high_angle))
            new_size = (
                max(4, int(patch.width * scale)),
                max(4, int(patch.height * scale)),
            )
            patch_scaled = patch.resize(new_size, Image.Resampling.BICUBIC)
            alpha_scaled = alpha.resize(new_size, Image.Resampling.BILINEAR)
            alpha_mask = alpha.resize(new_size, Image.Resampling.NEAREST)
            patch_scaled = patch_scaled.rotate(
                angle, expand=True, resample=Image.Resampling.BICUBIC
            )
            alpha_scaled = alpha_scaled.rotate(
                angle, expand=True, resample=Image.Resampling.BILINEAR
            )
            alpha_mask = alpha_mask.rotate(
                angle, expand=True, resample=Image.Resampling.NEAREST
            )
            if patch_scaled.width > width or patch_scaled.height > height:
                ratio = (
                    min(width / patch_scaled.width, height / patch_scaled.height) * 0.8
                )
                size = (
                    max(4, int(patch_scaled.width * ratio)),
                    max(4, int(patch_scaled.height * ratio)),
                )
                patch_scaled = patch_scaled.resize(size, Image.Resampling.BICUBIC)
                alpha_scaled = alpha_scaled.resize(size, Image.Resampling.BILINEAR)
                alpha_mask = alpha_mask.resize(size, Image.Resampling.NEAREST)
            x = int(rng.integers(0, max(1, width - patch_scaled.width + 1)))
            y = int(rng.integers(0, max(1, height - patch_scaled.height + 1)))
            target_region = np.asarray(
                result.crop((x, y, x + patch_scaled.width, y + patch_scaled.height))
            )
            patch_scaled = _match_patch_to_surface(
                patch_scaled, target_region, strength=0.55
            )
            # Two-stage feathering prevents a visible cut edge while retaining a
            # sufficiently strong center for the defect signal.
            alpha_scaled = alpha_scaled.filter(
                ImageFilter.GaussianBlur(max(1, int(min(patch_scaled.size) * 0.012)))
            )
            result.paste(patch_scaled, (x, y), alpha_scaled)
            placed_mask = Image.new("L", target.size, 0)
            placed_mask.paste(alpha_mask, (x, y))
            combined_mask = Image.fromarray(
                np.maximum(np.asarray(combined_mask), np.asarray(placed_mask)).astype(
                    np.uint8
                ),
                mode="L",
            )
            placements.append(
                {
                    "source_index": source_index,
                    "copy_index": copy_index,
                    "x": x,
                    "y": y,
                    "width": patch_scaled.width,
                    "height": patch_scaled.height,
                    "scale": round(scale, 5),
                    "rotation_degrees": round(angle, 5),
                }
            )

    metadata: Dict[str, object] = {
        "mode": "real_defect_reuse",
        "seed": seed,
        "source_count": len(defect_images),
        "copies_per_source": copies_per_source,
        "placement_count": len(placements),
        "scale_range": [low_scale, high_scale],
        "rotation_range": [low_angle, high_angle],
        "synthesis_profile": "surface_aware_reuse_v2",
        "characteristics": [
            "illumination_matched_patch",
            "contrast_matched_patch",
            "feathered_alpha_boundary",
            "binary_union_mask",
        ],
        "mask_area_percent": round(
            float(np.asarray(combined_mask).mean() / 255 * 100), 3
        ),
        "placements": placements,
    }
    return result, combined_mask, metadata
