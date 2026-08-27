"""Cross-backend validation for anomaly detection models.

The validator runs the same preprocessed images through multiple AnomaVision
backends and compares anomaly scores and pixel-level anomaly maps.  It is
backend agnostic, so the same command can validate PyTorch, ONNX Runtime and
Hailo HEF models.  PatchCore and PaDiM are supported through the existing
ModelWrapper backend interface.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import anomavision
from anomavision.config import _shape, load_config
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.utils import get_logger, merge_config, setup_logging

matplotlib.use("Agg")

logger = get_logger("anomavision.validate")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Create the command-line parser for cross-backend validation.

    Args:
        add_help: Whether argparse should register the ``--help`` option.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Compare anomaly scores and heatmaps across model backends.",
        add_help=add_help,
    )
    parser.add_argument("--config", required=True, help="Path to config.yml/.json")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Two or more model files to compare, e.g. model.pt model.onnx model.hef",
    )
    parser.add_argument("--img_path", required=True, help="Directory containing test images")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        default="./validation_results",
        help="Directory for JSON report and visual comparison images",
    )
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images")
    parser.add_argument("--score_abs_tol", type=float, default=1e-3)
    parser.add_argument("--score_rel_tol", type=float, default=1e-2)
    parser.add_argument("--map_mae_tol", type=float, default=1e-3)
    parser.add_argument("--map_rel_tol", type=float, default=1e-2)
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save original image and side-by-side heatmaps for every compared pair",
    )
    return parser


def _resolve_model_path(model: str, config: edict) -> str:
    """Resolve a model path using the same layout rules as ``detect``."""
    direct = Path(model).expanduser()
    if direct.is_file():
        return str(direct.resolve())
    return str(
        Path(config.model_data_path)
        / str(config.algorithm)
        / str(config.class_name)
        / str(config.run_name)
        / model
    )


def _as_batch_map(maps: Any) -> np.ndarray:
    """Convert backend map output to a ``(B, H, W)`` float32 array."""
    arr = np.asarray(maps, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected anomaly maps with 2 or 3 dimensions, got {arr.shape}")
    return arr


def _as_scores(scores: Any) -> np.ndarray:
    """Convert backend score output to a one-dimensional float32 array."""
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    return arr


def _relative_error(value: float, reference: float) -> float:
    """Return absolute relative error with a stable zero denominator."""
    return float(abs(value - reference) / max(abs(reference), 1e-12))


def _compare_pair(
    left_name: str,
    right_name: str,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    left_maps: np.ndarray,
    right_maps: np.ndarray,
    score_abs_tol: float,
    score_rel_tol: float,
    map_mae_tol: float,
    map_rel_tol: float,
) -> dict[str, Any]:
    """Compare scores and anomaly maps from two model backends."""
    if left_scores.shape != right_scores.shape:
        raise ValueError(f"Score shapes differ: {left_scores.shape} vs {right_scores.shape}")
    if left_maps.shape != right_maps.shape:
        raise ValueError(f"Map shapes differ: {left_maps.shape} vs {right_maps.shape}")

    score_abs = np.abs(left_scores - right_scores)
    score_rel = score_abs / np.maximum(np.abs(left_scores), 1e-12)
    map_diff = np.abs(left_maps - right_maps)
    map_mae = float(np.mean(map_diff))
    map_rmse = float(np.sqrt(np.mean(np.square(left_maps - right_maps))))
    map_max_abs = float(np.max(map_diff))
    map_scale = float(max(np.max(np.abs(left_maps)), 1e-12))
    map_rel = map_mae / map_scale

    score_pass = bool(
        np.all((score_abs <= score_abs_tol) | (score_rel <= score_rel_tol))
    )
    map_pass = bool(map_mae <= map_mae_tol or map_rel <= map_rel_tol)

    return {
        "left": left_name,
        "right": right_name,
        "score": {
            "max_absolute_error": float(np.max(score_abs)),
            "mean_absolute_error": float(np.mean(score_abs)),
            "max_relative_error": float(np.max(score_rel)),
            "mean_relative_error": float(np.mean(score_rel)),
            "pass": score_pass,
        },
        "heatmap": {
            "mae": map_mae,
            "rmse": map_rmse,
            "max_absolute_error": map_max_abs,
            "relative_mae": float(map_rel),
            "pass": map_pass,
        },
        "pass": score_pass and map_pass,
    }


def _save_pair_visualization(
    output_dir: Path,
    index: int,
    image: Any,
    names: tuple[str, str],
    maps: tuple[np.ndarray, np.ndarray],
    scores: tuple[float, float],
) -> None:
    """Save an original image and the two backend heatmaps side by side."""
    image_arr = np.asarray(image)
    if image_arr.ndim == 3 and image_arr.shape[0] in (1, 3) and image_arr.shape[-1] not in (1, 3):
        image_arr = np.transpose(image_arr, (1, 2, 0))
    image_arr = np.squeeze(image_arr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_arr)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, name, heatmap, score in zip(axes[1:], names, maps, scores):
        ax.imshow(image_arr)
        ax.imshow(heatmap, alpha=0.5, cmap="jet")
        ax.set_title(f"{name}\nscore={score:.6f}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"image_{index:05d}_{names[0]}_vs_{names[1]}.png", dpi=120)
    plt.close(fig)


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Run all requested models on identical batches and create a report.

    Args:
        args: Parsed validation command arguments.

    Returns:
        Validation report containing per-pair score and heatmap metrics.
    """
    config_file = load_config(str(args.config))
    config = edict(merge_config(args, config_file))
    config.img_path = args.img_path
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.device = args.device

    algorithm = str(config.get("algorithm", "unknown")).lower()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    if args.save_visualizations:
        viz_dir.mkdir(parents=True, exist_ok=True)

    resize = _shape(config.resize)
    crop_size = _shape(config.crop_size)
    normalize = config.get("normalize", True)
    dataset = anomavision.AnodetDataset(
        str(Path(args.img_path).expanduser().resolve()),
        resize=resize,
        crop_size=crop_size,
        normalize=normalize,
        mean=config.norm_mean,
        std=config.norm_std,
    )
    if args.max_images is not None:
        class _LimitedDataset:
            def __init__(self, source, limit):
                self.source = source
                self.limit = min(len(source), limit)

            def __len__(self):
                return self.limit

            def __getitem__(self, index):
                return self.source[index]

        dataset = _LimitedDataset(dataset, args.max_images)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model_paths = [_resolve_model_path(model, config) for model in args.models]
    names = [Path(path).stem for path in model_paths]
    if len(set(names)) != len(names):
        names = [f"{name}_{i}" for i, name in enumerate(names)]

    wrappers: list[ModelWrapper] = []
    try:
        for path in model_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Model file not found: {path}")
            wrappers.append(ModelWrapper(path, config.device))

        all_scores = [[] for _ in wrappers]
        all_maps = [[] for _ in wrappers]
        visualization_images = []

        for batch_index, (batch, images, _, _) in enumerate(loader):
            if args.save_visualizations:
                visualization_images.extend(images)
            for model_index, wrapper in enumerate(wrappers):
                scores, maps = wrapper.predict(batch)
                all_scores[model_index].append(_as_scores(scores))
                all_maps[model_index].append(_as_batch_map(maps))

        scores = [np.concatenate(items, axis=0) for items in all_scores]
        maps = [np.concatenate(items, axis=0) for items in all_maps]
        pair_results = []

        for left_index, right_index in combinations(range(len(wrappers)), 2):
            result = _compare_pair(
                names[left_index],
                names[right_index],
                scores[left_index],
                scores[right_index],
                maps[left_index],
                maps[right_index],
                args.score_abs_tol,
                args.score_rel_tol,
                args.map_mae_tol,
                args.map_rel_tol,
            )
            pair_results.append(result)

            if args.save_visualizations:
                for image_index in range(len(scores[left_index])):
                    _save_pair_visualization(
                        viz_dir,
                        image_index,
                        visualization_images[image_index],
                        (names[left_index], names[right_index]),
                        (maps[left_index][image_index], maps[right_index][image_index]),
                        (float(scores[left_index][image_index]), float(scores[right_index][image_index])),
                    )

        report = {
            "algorithm": algorithm,
            "models": dict(zip(names, model_paths)),
            "images": int(len(scores[0])) if scores else 0,
            "tolerances": {
                "score_abs": args.score_abs_tol,
                "score_relative": args.score_rel_tol,
                "heatmap_mae": args.map_mae_tol,
                "heatmap_relative_mae": args.map_rel_tol,
            },
            "comparisons": pair_results,
            "pass": bool(pair_results) and all(item["pass"] for item in pair_results),
        }

        report_path = output_dir / "validation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Validation report saved to %s", report_path)
        for item in pair_results:
            logger.info(
                "%s vs %s: %s (score max abs=%.6g, heatmap MAE=%.6g)",
                item["left"], item["right"],
                "PASS" if item["pass"] else "FAIL",
                item["score"]["max_absolute_error"],
                item["heatmap"]["mae"],
            )
        return report
    finally:
        for wrapper in wrappers:
            wrapper.close()


def main(args: argparse.Namespace | None = None) -> dict[str, Any]:
    """CLI entry point for backend validation."""
    if args is None:
        args = create_parser().parse_args()
    setup_logging(enabled=True, log_level="INFO", log_to_file=False)
    report = run_validation(args)
    if not report["pass"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
