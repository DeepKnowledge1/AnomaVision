"""Run AnomaVision anomaly detection inference on images and streams."""

import argparse
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import anomavision
from anomavision.config import _shape, load_config
from anomavision.datasets.StreamDataset import StreamDataset
from anomavision.datasets.StreamSourceFactory import StreamSourceFactory
from anomavision.general import Profiler, determine_device, increment_path
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType
from anomavision.utils import (
    adaptive_gaussian_blur,
    get_logger,
    make_localization_mask,
    merge_config,
    resolve_threshold,
    setup_logging,
)

matplotlib.use("Agg")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run anomaly detection inference using trained models.",
        add_help=add_help,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yml/.json")
    parser.add_argument("--img_path", type=str, default=None, help="Path to test images")
    parser.add_argument("--model_data_path", type=str, default="./distributions", help="Directory containing model files")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm name")
    parser.add_argument("--model", type=str, default=None, help="Model file or path")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=None, help="Inference batch size; Hailo HEF requires 1")
    parser.add_argument("--thresh", type=float, default=None, help="Anomaly classification threshold")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--enable_visualization", action="store_true", default=None)
    parser.add_argument("--save_visualizations", action="store_true", default=None)
    parser.add_argument("--viz_output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--viz_alpha", type=float, default=None)
    parser.add_argument("--viz_padding", type=int, default=None)
    parser.add_argument("--viz_color", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--detailed_timing", action="store_true")
    return parser


def _load_efficientad_threshold(model_path: Path):
    """Load the calibrated EfficientAD threshold saved by training."""
    candidates = [model_path.with_suffix(".pth"), model_path.parent / "model.pth"]
    for sidecar in candidates:
        if not sidecar.exists():
            continue
        try:
            artifact = torch.load(sidecar, map_location="cpu", weights_only=False)
            if isinstance(artifact, dict) and artifact.get("algorithm") == "efficientad":
                threshold = artifact.get("threshold")
                if threshold is None and isinstance(artifact.get("model_state"), dict):
                    threshold = artifact["model_state"].get("threshold")
                if isinstance(threshold, torch.Tensor):
                    threshold = threshold.item()
                if threshold is not None:
                    return float(threshold), sidecar
        except Exception:
            continue
    return None, None


def _resolve_model_path(config) -> str:
    """Resolve a direct model path or the legacy distribution layout."""
    model = str(config.model)
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


def run_inference(args):
    total_start_time = time.time()

    if args.config is not None:
        cfg = load_config(str(args.config))
    else:
        cfg = {}
        base = Path(args.model_data_path) if args.model_data_path else None
        if base is not None and (base / "config.yml").exists():
            cfg = load_config(str(base / "config.yml"))

    config = edict(merge_config(args, cfg))
    algorithm_name = str(config.get("algorithm", "")).lower()
    config.thresh = resolve_threshold(config)

    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger("anomavision.detect")
    stream_mode = bool(config.get("stream_mode", False))

    resize = _shape(config.resize)
    crop_size = _shape(config.crop_size)
    normalize = config.get("normalize", True)
    if not config.get("img_path") and not stream_mode:
        raise ValueError("img_path is required (via --img_path or config) when stream_mode is False")
    if not config.get("model"):
        raise ValueError("model is required (via --model or config)")

    device_str = determine_device(config.device)
    model_path = Path(_resolve_model_path(config)).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if algorithm_name == "efficientad" and config.thresh is None:
        threshold, sidecar = _load_efficientad_threshold(model_path)
        if threshold is not None:
            config.thresh = threshold
            logger.info("EfficientAD calibrated threshold: %.6f (source=%s)", threshold, sidecar)
        else:
            raise RuntimeError(
                "EfficientAD threshold is not configured and no calibrated .pth sidecar was found "
                f"next to {model_path}. Train EfficientAD first so its calibration artifact is saved."
            )

    logger.info("algorithm=%s model=%s device=%s threshold=%s", algorithm_name, model_path, device_str, config.thresh)

    profilers = {name: Profiler() for name in ["setup", "model_loading", "data_loading", "inference", "postprocessing", "visualization"]}
    results = {"scores": [], "classifications": [], "images": [] if not stream_mode else None}

    with profilers["setup"]:
        if not stream_mode:
            dataset_path = os.path.realpath(config.img_path)
            logger.info("Dataset path: %s", dataset_path)
        else:
            dataset_path = None
            logger.info("Streaming source type: %s", config.get("stream_source", {}).get("type", "unknown"))
        if device_str == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    with profilers["model_loading"]:
        logger.info("Loading model: %s", model_path)
        model_type = ModelType.from_extension(str(model_path))
        if model_type == ModelType.HEF and int(config.batch_size) != 1:
            raise ValueError("Hailo HEF inference currently requires batch_size=1")
        model = ModelWrapper(str(model_path), device_str)
        logger.info("Model loaded: %s", model_type.value.upper())

    results_path = None
    if config.get("save_visualizations", False):
        results_path = increment_path(
            Path(config.get("viz_output_dir", "./visualizations"))
            / config.algorithm / config.class_name / model_type.value.upper() / config.run_name,
            exist_ok=config.get("overwrite", False), mkdir=True,
        )

    try:
        with profilers["data_loading"]:
            if stream_mode:
                source = StreamSourceFactory.create(config.stream_source)
                source.connect()
                dataset = StreamDataset(
                    source=source, resize=resize, crop_size=crop_size,
                    normalize=normalize, mean=config.norm_mean, std=config.norm_std,
                    max_frames=config.get("stream_max_frames"),
                )
                workers, pin_memory = 0, False
            else:
                dataset = anomavision.AnodetDataset(
                    dataset_path, resize=resize, crop_size=crop_size,
                    normalize=normalize, mean=config.norm_mean, std=config.norm_std,
                )
                workers = int(config.get("num_workers", 0))
                pin_memory = bool(config.get("pin_memory", False))
            dataloader = DataLoader(dataset, batch_size=int(config.batch_size), num_workers=workers, pin_memory=pin_memory)
            try:
                total_images = len(dataset)
            except TypeError:
                total_images = None

        try:
            first = next(iter(dataloader))[0]
            if device_str == "cuda":
                first = first.half()
            model.warmup(first.to(device_str), runs=2)
        except Exception as exc:
            logger.warning("Warm-up skipped: %s", exc)

        batch_count = 0
        image_counter = 0
        for batch_idx, (batch, images, _, _) in enumerate(dataloader):
            batch_count += 1
            image_counter += batch.shape[0]
            if device_str == "cuda":
                batch = batch.half()
            batch = batch.to(device_str)

            with profilers["inference"]:
                image_scores, score_maps = model.predict(batch)

            with profilers["postprocessing"]:
                score_maps = adaptive_gaussian_blur(score_maps, kernel_size=33, sigma=4)
                is_anomaly = anomavision.classification(image_scores, config.thresh)
                if algorithm_name == "patchcore":
                    masks = make_localization_mask(score_maps, is_anomaly, quantile=0.90)
                else:
                    masks = anomavision.classification(score_maps, config.thresh)

                if not stream_mode:
                    results["scores"].extend(np.asarray(image_scores).reshape(-1).tolist())
                    results["classifications"].extend(np.asarray(is_anomaly).reshape(-1).tolist())
                    results["images"].extend(images)

            if config.get("enable_visualization", False):
                with profilers["visualization"]:
                    boundaries = anomavision.visualization.framed_boundary_images(
                        images, masks, is_anomaly, padding=config.get("viz_padding", 40)
                    )
                    heatmaps = anomavision.visualization.heatmap_images(
                        images, score_maps, masks=masks, alpha=config.get("viz_alpha", 0.5)
                    )
                    highlighted = anomavision.visualization.highlighted_images(
                        [images[i] for i in range(len(images))], masks,
                        color=tuple(map(int, str(config.get("viz_color", "128,0,128")).split(","))),
                    )
                    if config.get("save_visualizations", False) and results_path:
                        for i in range(len(images)):
                            fig, axs = plt.subplots(1, 4, figsize=(16, 8))
                            fig.suptitle(
                                f"Result - Batch {batch_idx} Img {i} - score={float(np.asarray(image_scores).reshape(-1)[i]):.6f}",
                                fontsize=14,
                            )
                            for ax, image, title in zip(
                                axs,
                                [images[i], boundaries[i], heatmaps[i], highlighted[i]],
                                ["Original", "Boundary", "Heatmap", "Highlighted"],
                            ):
                                ax.imshow(image); ax.set_title(title); ax.axis("off")
                            fig.savefig(Path(results_path) / f"batch_{batch_idx}_img_{i}.png", dpi=100, bbox_inches="tight")
                            plt.close(fig)
    finally:
        model.close()
        if stream_mode:
            try:
                dataset.close()
            except Exception:
                pass

    total_pipeline_time = time.time() - total_start_time
    final_count = total_images if (not stream_mode and total_images is not None) else image_counter
    inference_seconds = profilers["inference"].accumulated_time
    fps = final_count / inference_seconds if inference_seconds > 0 else 0.0
    avg_ms = inference_seconds / batch_count * 1000.0 if batch_count else 0.0

    logger.info("ANOMAVISION PERFORMANCE: inference=%.2f ms, throughput=%.2f images/sec, total=%.2f ms", inference_seconds * 1000, fps, total_pipeline_time * 1000)
    return {
        "fps": fps,
        "avg_inference_ms": avg_ms,
        "total_time_s": total_pipeline_time,
        "total_images": final_count,
    }, results


def main(args=None):
    try:
        if args is None:
            args = create_parser().parse_args()
        return run_inference(args)
    except KeyboardInterrupt:
        get_logger("anomavision.detect").info("Process interrupted")
        raise SystemExit(1)
    except Exception as exc:
        get_logger("anomavision.detect").error("Process failed: %s", exc, exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
