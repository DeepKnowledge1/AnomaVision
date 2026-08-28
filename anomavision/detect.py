"""
Run AnomaVision anomaly detection inference on images using various model formats.

Usage - formats:
    $ python detect.py --model model.pt                     # PyTorch
    $ python detect.py --model model.torchscript             # TorchScript
    $ python detect.py --model model.onnx                    # ONNX Runtime
    $ python detect.py --model model_openvino                # OpenVINO
    $ python detect.py --model model.engine                  # TensorRT
    $ python detect.py --model model.hef                    # Hailo HEF
"""

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

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yml/.json",
    )
    parser.add_argument(
        "--img_path",
        default=None,
        type=str,
        help="Path to the dataset folder containing test images.",
    )
    parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions",
        help="Directory containing model files.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm name: padim | patchcore | efficientad.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model file (.pt, .onnx, .hef, etc.). "
        "A path outside model_data_path is also accepted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference. Hailo currently requires 1.",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Threshold for anomaly classification.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Use pinned memory for faster GPU transfers.",
    )
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        default=None,
        help="Enable visualization of results.",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        default=None,
        help="Save visualization images to disk.",
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default=None,
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Experiment name for this inference run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run directory without auto-incrementing.",
    )
    parser.add_argument(
        "--viz_alpha",
        type=float,
        default=None,
        help="Alpha value for heatmap overlay.",
    )
    parser.add_argument(
        "--viz_padding",
        type=int,
        default=None,
        help="Padding for boundary visualization.",
    )
    parser.add_argument(
        "--viz_color",
        type=str,
        default=None,
        help='RGB color for highlighting, e.g. "128,0,128".',
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--detailed_timing",
        action="store_true",
        help="Enable detailed timing measurements.",
    )

    return parser


def _load_efficientad_threshold(model_path: Path):
    """Load the calibrated EfficientAD threshold saved by training."""
    candidates = [model_path.with_suffix(".pth"), model_path.parent / "model.pth"]
    for sidecar in candidates:
        if not sidecar.exists():
            continue
        try:
            artifact = torch.load(sidecar, map_location="cpu", weights_only=False)
            if (
                isinstance(artifact, dict)
                and artifact.get("algorithm") == "efficientad"
            ):
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
    """
    Resolve a model argument without changing the legacy distribution layout.

    Absolute paths and existing relative paths are used directly.
    Bare model names continue to use:

        model_data_path / algorithm / class_name / run_name / model
    """
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

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------
    if args.config is not None:
        cfg = load_config(str(args.config))
    else:
        cfg = {}

        if args.model_data_path:
            config_path = Path(args.model_data_path) / "config.yml"

            if config_path.exists():
                cfg = load_config(str(config_path))

    config = edict(merge_config(args, cfg))

    algorithm_name = str(config.get("algorithm", "")).lower()

    # Resolve normal/algorithm-specific threshold first.
    config.thresh = resolve_threshold(config)

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    setup_logging(
        enabled=True,
        log_level=config.log_level,
        log_to_file=True,
    )

    logger = get_logger("anomavision.detect")

    stream_mode = bool(config.get("stream_mode", False))

    logger.info("Streaming mode: %s", stream_mode)

    # ---------------------------------------------------------
    # Image preprocessing
    # ---------------------------------------------------------
    resize = _shape(config.resize)
    crop_size = _shape(config.crop_size)
    normalize = config.get("normalize", True)

    logger.info(
        "Image processing: resize=%s, crop=%s, norm=%s",
        resize,
        crop_size,
        normalize,
    )

    # ---------------------------------------------------------
    # Validate required configuration
    # ---------------------------------------------------------
    if not config.get("img_path") and not stream_mode:
        raise ValueError(
            "img_path is required (via --img_path or config) "
            "when stream_mode is False"
        )

    if not config.get("model"):
        raise ValueError("model is required (via --model or config)")

    # ---------------------------------------------------------
    # Resolve device
    # ---------------------------------------------------------
    device_str = determine_device(config.device)

    logger.info("Device: %s", device_str)

    if device_str == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------
    # EfficientAD threshold calibration
    # ---------------------------------------------------------
    # IMPORTANT:
    # EfficientAD's threshold is calibrated during training.
    # It should NOT be hard-coded in config.yml.
    if algorithm_name == "efficientad" and config.thresh is None:
        model_path_for_threshold = Path(_resolve_model_path(config)).resolve()

        threshold, sidecar = _load_efficientad_threshold(model_path_for_threshold)

        if threshold is not None:
            config.thresh = threshold

            logger.info(
                "EfficientAD calibrated threshold: %.6f (source=%s)",
                threshold,
                sidecar,
            )
        else:
            raise RuntimeError(
                "EfficientAD threshold is not configured and no "
                "calibrated .pth sidecar was found next to "
                f"{model_path_for_threshold}. "
                "Train EfficientAD first so its calibration artifact "
                "is saved."
            )

    # ---------------------------------------------------------
    # Visualization color
    # ---------------------------------------------------------
    try:
        if config.get("viz_color"):
            values = tuple(map(int, str(config.viz_color).split(",")))

            if len(values) != 3:
                raise ValueError

            viz_color = values

        else:
            viz_color = (128, 0, 128)

    except (ValueError, TypeError):
        logger.warning(
            "Invalid color format '%s'. " "Using default (128,0,128)",
            getattr(config, "viz_color", None),
        )
        viz_color = (128, 0, 128)

    logger.info(
        "algorithm=%s threshold=%s",
        algorithm_name,
        config.thresh,
    )

    # ---------------------------------------------------------
    # Profilers
    # ---------------------------------------------------------
    profilers = {
        name: Profiler()
        for name in [
            "setup",
            "model_loading",
            "data_loading",
            "inference",
            "postprocessing",
            "visualization",
        ]
    }

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    results_accumulator = {
        "scores": [],
        "classifications": [],
        "images": [] if not stream_mode else None,
    }

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------
    with profilers["setup"]:
        if not stream_mode:
            dataset_path = os.path.realpath(config.img_path)
            logger.info("Dataset path: %s", dataset_path)
        else:
            dataset_path = None

            src = config.get("stream_source", {})
            logger.info(
                "Streaming source type: %s",
                src.get("type", "unknown"),
            )

        model_data_path = os.path.realpath(config.model_data_path)

        logger.info(
            "Model data path: %s",
            model_data_path,
        )

    # ---------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------
    with profilers["model_loading"]:
        model_path = _resolve_model_path(config)

        logger.info("Loading model: %s", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            model_type = ModelType.from_extension(model_path)

            # Hailo HEF currently requires batch size 1.
            if model_type == ModelType.HEF and int(config.batch_size) != 1:
                raise ValueError(
                    "Hailo HEF inference currently requires "
                    "--batch_size 1 for deterministic backend comparison"
                )

            model = ModelWrapper(
                model_path,
                device_str,
            )

            logger.info(
                "Model loaded: %s",
                model_type.value.upper(),
            )

        except Exception as exc:
            logger.error(
                "Failed to load model: %s",
                exc,
            )
            raise

    # ---------------------------------------------------------
    # Visualization output
    # ---------------------------------------------------------
    results_path = None

    if config.get("save_visualizations", False):
        viz_output_dir = config.get(
            "viz_output_dir",
            "./visualizations/",
        )

        results_path = increment_path(
            Path(viz_output_dir)
            / config.algorithm
            / config.class_name
            / model_type.value.upper()
            / config.run_name,
            exist_ok=config.get("overwrite", False),
            mkdir=True,
        )

        logger.info(
            "Visualization output: %s",
            results_path,
        )

    # ---------------------------------------------------------
    # Dataset / dataloader
    # ---------------------------------------------------------
    with profilers["data_loading"]:
        try:
            if not stream_mode:
                test_dataset = anomavision.AnodetDataset(
                    dataset_path,
                    resize=resize,
                    crop_size=crop_size,
                    normalize=normalize,
                    mean=config.norm_mean,
                    std=config.norm_std,
                )

                num_workers = int(config.get("num_workers", 0))

                pin_memory = bool(config.get("pin_memory", False))

            else:
                source = StreamSourceFactory.create(config.stream_source)

                source.connect()

                test_dataset = StreamDataset(
                    source=source,
                    resize=resize,
                    crop_size=crop_size,
                    normalize=normalize,
                    mean=config.norm_mean,
                    std=config.norm_std,
                    max_frames=config.get("stream_max_frames"),
                )

                num_workers = 0
                pin_memory = False

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=int(config.batch_size),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            try:
                total_images = len(test_dataset)
                logger.info(
                    "Total images: %s",
                    total_images,
                )
            except TypeError:
                total_images = None
                logger.info("Streaming mode (infinite/unknown length)")

        except Exception as exc:
            logger.error(
                "Failed to create dataloader: %s",
                exc,
            )
            raise

    # ---------------------------------------------------------
    # Model warm-up
    # ---------------------------------------------------------
    try:
        first = next(iter(test_dataloader))
        first_batch = first[0]

        if device_str == "cuda":
            first_batch = first_batch.half()

        first_batch = first_batch.to(device_str)

        model.warmup(
            batch=first_batch,
            runs=2,
        )

        logger.info("Warm-up complete.")

    except StopIteration:
        logger.warning("Dataset empty; skipping warm-up.")

    except Exception as exc:
        logger.warning(
            "Warm-up skipped: %s",
            exc,
        )

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    batch_count = 0
    image_counter = 0

    try:
        for batch_idx, (
            batch,
            images,
            _,
            _,
        ) in enumerate(test_dataloader):

            batch_count += 1
            image_counter += batch.shape[0]

            if device_str == "cuda":
                batch = batch.half()

            batch = batch.to(device_str)

            # -------------------------
            # Model inference
            # -------------------------
            with profilers["inference"]:
                try:
                    image_scores, score_maps = model.predict(batch)

                except Exception as exc:
                    logger.error(
                        "Inference failed batch %d: %s",
                        batch_idx,
                        exc,
                    )
                    continue

            # -------------------------
            # Post-processing
            # -------------------------
            with profilers["postprocessing"]:
                try:
                    score_maps = adaptive_gaussian_blur(
                        score_maps,
                        kernel_size=33,
                        sigma=4,
                    )

                    if config.thresh is not None:
                        is_anomaly = anomavision.classification(
                            image_scores,
                            config.thresh,
                        )
                    else:
                        is_anomaly = np.zeros_like(image_scores)

                    if algorithm_name == "patchcore":
                        localization_masks = make_localization_mask(
                            score_maps,
                            is_anomaly,
                            quantile=0.90,
                        )
                    else:
                        if config.thresh is not None:
                            localization_masks = anomavision.classification(
                                score_maps,
                                config.thresh,
                            )
                        else:
                            localization_masks = np.zeros_like(score_maps)

                    if not stream_mode:
                        results_accumulator["scores"].extend(
                            np.asarray(image_scores).reshape(-1).tolist()
                        )

                        results_accumulator["classifications"].extend(
                            np.asarray(is_anomaly).reshape(-1).tolist()
                        )

                        results_accumulator["images"].extend(images)

                except Exception as exc:
                    logger.error(
                        "Postprocessing failed batch %d: %s",
                        batch_idx,
                        exc,
                    )
                    continue

            # -------------------------
            # Visualization
            # -------------------------
            if config.get(
                "enable_visualization",
                False,
            ):
                with profilers["visualization"]:
                    try:
                        boundary_images = (
                            anomavision.visualization.framed_boundary_images(
                                images,
                                localization_masks,
                                is_anomaly,
                                padding=config.get(
                                    "viz_padding",
                                    40,
                                ),
                            )
                        )

                        heatmap_images = anomavision.visualization.heatmap_images(
                            images,
                            score_maps,
                            masks=localization_masks,
                            alpha=config.get(
                                "viz_alpha",
                                0.5,
                            ),
                        )

                        highlighted_images = (
                            anomavision.visualization.highlighted_images(
                                [images[i] for i in range(len(images))],
                                localization_masks,
                                color=viz_color,
                            )
                        )

                        for img_id in range(len(images)):
                            if (
                                config.get(
                                    "save_visualizations",
                                    False,
                                )
                                and results_path
                            ):
                                try:
                                    scores = np.asarray(image_scores).reshape(-1)

                                    score = float(scores[img_id])

                                    fig, axs = plt.subplots(
                                        1,
                                        4,
                                        figsize=(16, 8),
                                    )

                                    fig.suptitle(
                                        f"Result - Batch "
                                        f"{batch_idx} Img "
                                        f"{img_id} - "
                                        f"score={score:.6f}",
                                        fontsize=14,
                                    )

                                    axs[0].imshow(images[img_id])
                                    axs[0].set_title("Original")
                                    axs[0].axis("off")

                                    axs[1].imshow(boundary_images[img_id])
                                    axs[1].set_title("Boundary")
                                    axs[1].axis("off")

                                    axs[2].imshow(heatmap_images[img_id])
                                    axs[2].set_title("Heatmap")
                                    axs[2].axis("off")

                                    axs[3].imshow(highlighted_images[img_id])
                                    axs[3].set_title("Highlighted")
                                    axs[3].axis("off")

                                    fig.savefig(
                                        Path(results_path)
                                        / (f"batch_{batch_idx}" f"_img_{img_id}.png"),
                                        dpi=100,
                                        bbox_inches="tight",
                                    )

                                    plt.close(fig)

                                except Exception as exc:
                                    logger.warning(
                                        "Viz save failed: %s",
                                        exc,
                                    )

                    except Exception as exc:
                        logger.error(
                            "Visualization failed batch %d: %s",
                            batch_idx,
                            exc,
                        )

    finally:
        logger.info("Closing model...")

        model.close()

        if stream_mode:
            try:
                test_dataset.close()
            except Exception:
                pass

    # ---------------------------------------------------------
    # Performance summary
    # ---------------------------------------------------------
    total_pipeline_time = time.time() - total_start_time

    final_count = (
        total_images
        if (not stream_mode and total_images is not None)
        else image_counter
    )

    inference_seconds = profilers["inference"].accumulated_time

    fps = final_count / inference_seconds if inference_seconds > 0 else 0.0

    avg_ms = inference_seconds / batch_count * 1000.0 if batch_count > 0 else 0.0

    throughput = final_count / inference_seconds if inference_seconds > 0 else 0.0

    logger.info("=" * 60)
    logger.info("ANOMAVISION PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    logger.info(
        "Setup time:                %.2f ms",
        profilers["setup"].accumulated_time * 1000,
    )

    logger.info(
        "Model loading time:        %.2f ms",
        profilers["model_loading"].accumulated_time * 1000,
    )

    logger.info(
        "Data loading time:         %.2f ms",
        profilers["data_loading"].accumulated_time * 1000,
    )

    logger.info(
        "Inference time:            %.2f ms",
        profilers["inference"].accumulated_time * 1000,
    )

    logger.info(
        "Postprocessing time:       %.2f ms",
        profilers["postprocessing"].accumulated_time * 1000,
    )

    logger.info(
        "Visualization time:        %.2f ms",
        profilers["visualization"].accumulated_time * 1000,
    )

    logger.info(
        "Total pipeline time:       %.2f ms",
        total_pipeline_time * 1000,
    )

    logger.info("=" * 60)

    logger.info("=" * 60)
    logger.info("ANOMAVISION INFERENCE PERFORMANCE")
    logger.info("=" * 60)

    if fps > 0:
        logger.info(
            "Pure inference FPS:        %.2f images/sec",
            fps,
        )

    if avg_ms > 0:
        logger.info(
            "Average inference time:    %.2f ms/batch",
            avg_ms,
        )

    if batch_count > 0:
        logger.info(
            "Throughput:                %.1f images/sec " "(batch size: %s)",
            throughput,
            config.get("batch_size", 1) or 1,
        )

    logger.info("=" * 60)

    logger.info(
        "Detection complete: %d images, %.2fs, %.2f FPS",
        final_count,
        total_pipeline_time,
        fps,
    )

    # logger.info(
    #     "Scores: %s",
    #     results_accumulator["scores"],
    # )

    metrics = {
        "total_images": final_count,
        "total_time_s": total_pipeline_time,
        "fps": fps,
        "avg_inference_ms": avg_ms,
        "model": model_path,
        "model_type": model_type.value,
    }

    return metrics, results_accumulator


def main(args=None):
    try:
        if args is None:
            parser = create_parser()
            args = parser.parse_args()

        return run_inference(args)

    except KeyboardInterrupt:
        get_logger("anomavision.detect").info("Process interrupted")
        raise SystemExit(1)

    except Exception as exc:
        get_logger("anomavision.detect").error(
            "Process failed: %s",
            exc,
            exc_info=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
