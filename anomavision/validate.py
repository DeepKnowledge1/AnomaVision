"""Single-image validation for AnomaVision models.

Runs one image through a PyTorch, ONNX, or (when a device is available) Hailo
model and writes a simple original/heatmap result image. PT and ONNX use the
same configurable preprocessing as the normal AnomaVision inference pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from anomavision.config import _shape, load_config
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.utils import create_image_transform, merge_config, resolve_threshold

matplotlib.use("Agg")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one image with a PyTorch, ONNX, or Hailo model.",
        add_help=add_help,
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--model", required=True, help="Model path (.pt/.pth, .onnx, or .hef)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config file used for preprocessing and threshold settings.",
    )
    parser.add_argument(
        "--algorithm", default=None, help="Algorithm name, e.g. patchcore or padim."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for PT/ONNX inference. Hailo uses the connected Hailo device.",
    )
    parser.add_argument(
        "--threshold",
        "--thresh",
        dest="threshold",
        type=float,
        default=None,
        help="Optional anomaly threshold. If omitted, no anomaly/normal label is shown.",
    )
    parser.add_argument(
        "--output-dir",
        default="./validation_results",
        help="Directory for result images and JSON output.",
    )
    parser.add_argument(
        "--name", default=None, help="Output name. Defaults to the input image stem."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Heatmap overlay opacity."
    )
    parser.add_argument(
        "--compare-model",
        default=None,
        help="Optional second PT/ONNX model. Both models are run on the same image.",
    )
    return parser


def _load_settings(args: argparse.Namespace) -> Dict:
    config = load_config(args.config) if args.config else {}
    return merge_config(args, config)


def _preprocess(image: Image.Image, settings: Dict) -> torch.Tensor:
    resize = _shape(settings.get("resize", [224, 224]))
    crop_size = _shape(settings.get("crop_size", None))
    transform = create_image_transform(
        resize=resize,
        crop_size=crop_size,
        normalize=bool(settings.get("normalize", True)),
        mean=settings.get("norm_mean", [0.485, 0.456, 0.406]),
        std=settings.get("norm_std", [0.229, 0.224, 0.225]),
    )
    return transform(image).unsqueeze(0)


def _run_model(
    model_path: str, image: Image.Image, settings: Dict, device: str
) -> Tuple[float, np.ndarray, float]:
    """Run one model and return score, score map, and inference time in ms."""
    suffix = Path(model_path).suffix.lower()
    wrapper = ModelWrapper(model_path, device)
    try:
        if suffix == ".hef":
            # HailoBackend owns its device-side preprocessing and expects RGB HWC.
            model_input = np.asarray(image.convert("RGB"))
        else:
            model_input = _preprocess(image.convert("RGB"), settings)

        start = time.perf_counter()
        scores, maps = wrapper.predict(model_input)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    finally:
        wrapper.close()

    score = float(np.asarray(scores).reshape(-1)[0])
    score_map = np.asarray(maps).squeeze().astype(np.float32)
    if score_map.ndim != 2:
        raise RuntimeError(
            f"Expected a 2-D score map for visualization, got shape {score_map.shape}."
        )
    return score, score_map, elapsed_ms


def _make_heatmap(image: np.ndarray, score_map: np.ndarray, alpha: float) -> np.ndarray:
    """Create an RGB heatmap overlay with robust normalization."""
    height, width = image.shape[:2]
    score_map = cv2.resize(score_map, (width, height), interpolation=cv2.INTER_LINEAR)
    finite = score_map[np.isfinite(score_map)]
    if finite.size == 0:
        normalized = np.zeros_like(score_map, dtype=np.float32)
    else:
        low = float(finite.min())
        high = float(finite.max())
        if high <= low:
            normalized = np.zeros_like(score_map, dtype=np.float32)
        else:
            normalized = np.clip((score_map - low) / (high - low), 0.0, 1.0)

    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1.0 - alpha, heat, alpha, 0.0)


def _save_result(
    image: np.ndarray,
    score_map: np.ndarray,
    score: float,
    output_path: Path,
    title: str,
    threshold: Optional[float],
    alpha: float,
) -> None:
    heatmap = _make_heatmap(image, score_map, alpha)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(heatmap)
    label = ""
    if threshold is not None:
        label = " | ANOMALY" if score >= threshold else " | NORMAL"
    axes[1].set_title(f"Heatmap | score={score:.6f}{label}")
    axes[1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _run(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    model_path = Path(args.model)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    settings = _load_settings(args)
    if args.algorithm:
        settings["algorithm"] = args.algorithm
    threshold = args.threshold
    if threshold is None:
        threshold = resolve_threshold(settings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or image_path.stem
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)

    models = [str(model_path)]
    if args.compare_model:
        compare_path = Path(args.compare_model)
        if not compare_path.is_file():
            raise FileNotFoundError(f"Comparison model not found: {compare_path}")
        models.append(str(compare_path))

    results = []
    for path in models:
        score, score_map, elapsed_ms = _run_model(path, image, settings, args.device)
        model_name = Path(path).stem
        result_path = output_dir / f"{stem}_{model_name}_result.png"
        _save_result(
            image_np,
            score_map,
            score,
            result_path,
            f"{model_name} ({Path(path).suffix.lower()})",
            threshold,
            args.alpha,
        )
        result = {
            "model": path,
            "score": score,
            "inference_ms": elapsed_ms,
            "threshold": threshold,
            "classification": (
                "anomaly"
                if threshold is not None and score >= threshold
                else "normal"
                if threshold is not None
                else None
            ),
            "result_image": str(result_path),
            "score_map_shape": list(score_map.shape),
        }
        results.append(result)
        print(f"{model_name}: score={score:.6f}, inference={elapsed_ms:.2f} ms")
        if threshold is not None:
            print(
                f"  classification={'ANOMALY' if score >= threshold else 'NORMAL'} "
                f"(threshold={threshold})"
            )
        print(f"  result={result_path}")

    if len(results) == 2:
        a, b = results
        diff = abs(a["score"] - b["score"])
        relative = diff / max(abs(a["score"]), abs(b["score"]), 1e-12)
        comparison = {
            "model_a": a["model"],
            "model_b": b["model"],
            "score_absolute_difference": diff,
            "score_relative_difference": relative,
        }
        results.append({"comparison": comparison})
        print(f"score difference={diff:.6f} ({relative * 100:.3f}% relative)")

    json_path = output_dir / f"{stem}_results.json"
    json_path.write_text(
        json.dumps({"image": str(image_path), "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"  metrics={json_path}")
    return 0


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = create_parser().parse_args()
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
