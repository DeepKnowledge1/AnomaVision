"""
Run Anomaly detection inference on images using various model formats.
Usage - formats:
    $ python detect.py --model padim_model.pt                  # PyTorch
                                   padim_model.torchscript        # TorchScript
                                   padim_model.onnx               # ONNX Runtime
                                   padim_model_openvino           # OpenVINO
                                   padim_model.engine             # TensorRT
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
from anomavision.datasets.StreamDataset import StreamDataset
from anomavision.datasets.StreamSourceFactory import StreamSourceFactory

from anomavision.config import _shape, load_config
from anomavision.general import Profiler, determine_device, increment_path
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType
from anomavision.utils import (
    adaptive_gaussian_blur,
    get_logger,
    merge_config,
    setup_logging,
)

matplotlib.use("Agg")  # non-interactive, faster PNG writing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection inference using trained models."
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yml/.json"
    )

    # Dataset parameters
    parser.add_argument(
        "--img_path",
        default=None,
        type=str,
        help="Path to the dataset folder containing test images.",
    )

    # Model parameters
    parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions/anomav_exp",
        help="Directory containing model files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="padim_model.pt",
        help="Model file (.pt for PyTorch, .onnx for ONNX, .engine for TensorRT)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on (auto will choose cuda if available)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for inference"
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Threshold for anomaly classification",
    )

    # Data loading parameters
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

    # Visualization parameters
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
        default="detect_exp",
        help="experiment name for this inference run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing run directory without auto-incrementing",
    )
    parser.add_argument(
        "--viz_alpha", type=float, default=None, help="Alpha value for heatmap overlay."
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
        help='RGB color for highlighting (comma-separated, e.g., "128,0,128").',
    )

    # Logging parameters
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

    # Return parser and args to support external extension
    args, unknown = parser.parse_known_args() if __name__ == "__main__" else (None, None)
    return parser, args


def run_inference(args):
    """
    Executes the inference pipeline.

    Args:
        args: Namespace object containing configuration.

    Returns:
        metrics (dict): Performance and timing metrics.
        results (dict): Dictionary containing keys ['scores', 'classifications', 'images']
                        (Only populated for offline mode to prevent OOM in streaming).
    """
    if args.config is not None:
        cfg = load_config(str(args.config))
    else:
        # Fallback to model directory config
        cfg = load_config(str(Path(args.model_data_path) / "config.yml"))

    # Merge config with CLI args
    config = edict(merge_config(args, cfg))

    # Setup logging
    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger("anomavision.detect")

    stream_mode = config.get("stream_mode", False)
    logger.info(f"Streaming mode: {stream_mode}")

    # Parse visualization color
    try:
        viz_color = tuple(map(int, config.viz_color.split(","))) if config.viz_color else (128, 0, 128)
        if len(viz_color) != 3:
            raise ValueError
    except (ValueError, AttributeError):
        logger.warning(
            f"Invalid color format '{getattr(config, 'viz_color', 'None')}'. Using default (128,0,128)"
        )
        viz_color = (128, 0, 128)

    # Parse image processing arguments
    resize = _shape(config.resize)
    crop_size = _shape(config.crop_size)
    normalize = config.get("normalize", True)

    logger.info("Image processing: resize=%s, crop=%s, norm=%s", resize, crop_size, normalize)

    # Validation
    if not config.get("img_path") and not stream_mode:
        raise ValueError("img_path is required (via --img_path or config) when stream_mode is False")

    if not config.get("model"):
        raise ValueError("model is required (via --model or config)")

    # Profilers
    profilers = {
        "setup": Profiler(),
        "model_loading": Profiler(),
        "data_loading": Profiler(),
        "inference": Profiler(),
        "postprocessing": Profiler(),
        "visualization": Profiler(),
    }

    results_accumulator = {
        "scores": [],
        "classifications": [],
        # We only store images/maps if needed for downstream tasks to avoid memory issues
        "images": [] if not stream_mode else None
    }

    total_start_time = time.time()

    # --- Setup Phase ---
    with profilers["setup"]:
        if not stream_mode:
            DATASET_PATH = os.path.realpath(config.img_path)
            logger.info(f"Dataset path: {DATASET_PATH}")
        else:
            DATASET_PATH = None
            src = config.get("stream_source", {})
            logger.info(f"Streaming source type: {src.get('type', 'unknown')}")

        MODEL_DATA_PATH = os.path.realpath(config.model_data_path)
        device_str = determine_device(config.device)
        logger.info(f"Device: {device_str}")

        if device_str == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    # --- Model Loading Phase ---
    with profilers["model_loading"]:
        model_path = os.path.join(MODEL_DATA_PATH, config.model)
        logger.info(f"Loading model: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            model = ModelWrapper(model_path, device_str)
            model_type = ModelType.from_extension(model_path)
            logger.info(f"Model loaded: {model_type.value.upper()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # --- Viz Directory Setup ---
    RESULTS_PATH = None
    if config.get("save_visualizations", False):
        run_name = config.run_name
        viz_output_dir = config.get("viz_output_dir", "./visualizations/")
        RESULTS_PATH = increment_path(
            Path(viz_output_dir) / model_type.value.upper() / run_name,
            exist_ok=config.get("overwrite", False),
            mkdir=True,
        )
        logger.info(f"Visualization output: {RESULTS_PATH}")

    # --- Data Loading Phase ---
    with profilers["data_loading"]:
        try:
            if not stream_mode:
                test_dataset = anomavision.AnodetDataset(
                    DATASET_PATH,
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
                batch_size=config.batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            # Log dataset stats
            try:
                total_images = len(test_dataset)
                logger.info(f"Total images: {total_images}")
            except TypeError:
                total_images = None
                logger.info("Streaming mode (infinite/unknown length)")

        except Exception as e:
            logger.error(f"Failed to create dataloader: {e}")
            raise

    # --- Warm-up ---
    try:
        first = next(iter(test_dataloader))
        first_batch = first[0]
        if device_str == "cuda":
            first_batch = first_batch.half()
        first_batch = first_batch.to(device_str)
        model.warmup(batch=first_batch, runs=2)
        logger.info("Warm-up complete.")
    except StopIteration:
        logger.warning("Dataset empty; skipping warm-up.")
    except Exception as e:
        logger.warning(f"Warm-up skipped: {e}")

    # --- Inference Loop ---
    batch_count = 0
    image_counter = 0

    try:
        for batch_idx, (batch, images, _, _) in enumerate(test_dataloader):
            batch_count += 1
            image_counter += batch.shape[0]

            if device_str == "cuda":
                batch = batch.half()
            batch = batch.to(device_str)

            # 1. Inference
            with profilers["inference"]:
                try:
                    image_scores, score_maps = model.predict(batch)
                except Exception as e:
                    logger.error(f"Inference failed batch {batch_idx}: {e}")
                    continue

            # 2. Post-processing
            with profilers["postprocessing"]:
                try:
                    score_maps = adaptive_gaussian_blur(score_maps, kernel_size=33, sigma=4)

                    # Classify
                    if config.thresh is not None:
                        is_anomaly = anomavision.classification(image_scores, config.thresh)
                    else:
                        is_anomaly = np.zeros_like(image_scores)

                    # Accumulate Results (Offline only)
                    if not stream_mode:
                        results_accumulator["scores"].extend(image_scores.tolist())
                        results_accumulator["classifications"].extend(is_anomaly.tolist())
                        results_accumulator["images"].extend(images)

                except Exception as e:
                    logger.error(f"Postprocessing failed batch {batch_idx}: {e}")
                    continue

            # 3. Visualization
            if config.enable_visualization:
                with profilers["visualization"]:
                    try:
                        test_images = np.array(images)

                        boundary_images = (
                            anomavision.visualization.framed_boundary_images(
                                test_images,
                                anomavision.classification(score_maps, config.thresh)if config.thresh else np.zeros_like(score_maps),
                                is_anomaly,
                                padding=config.get("viz_padding", 40),
                            )
                        )

                        heatmap_images = anomavision.visualization.heatmap_images(
                            test_images,
                            score_maps,
                            alpha=config.get("viz_alpha", 0.5),
                        )
                        highlighted_images = anomavision.visualization.highlighted_images(
                            [images[i] for i in range(len(images))],
                            # Dummy mask if threshold not set
                            anomavision.classification(score_maps, config.thresh) if config.thresh else np.zeros_like(score_maps),
                            color=viz_color,
                        )

                        # Save/Show
                        for img_id in range(len(images)):
                            # Only save if explicitly requested
                            if config.save_visualizations and RESULTS_PATH:
                                try:
                                    fig, axs = plt.subplots(1, 4, figsize=(16, 8))
                                    fig.suptitle(f"Result - Batch {batch_idx} Img {img_id}", fontsize=14)

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

                                    save_path = os.path.join(RESULTS_PATH, f"batch_{batch_idx}_img_{img_id}.png")
                                    plt.savefig(save_path, dpi=100, bbox_inches="tight")
                                    plt.close(fig)
                                except Exception as e:
                                    logger.warning(f"Viz save failed: {e}")

                    except Exception as e:
                        logger.error(f"Visualization failed batch {batch_idx}: {e}")

    finally:
        logger.info("Closing model...")
        model.close()
        if stream_mode:
             # Clean up stream source
             try:
                 test_dataset.close()
             except: pass

    # --- Metrics & Summary ---
    total_pipeline_time = time.time() - total_start_time

    # Calculate FPS
    final_count = total_images if (not stream_mode and total_images) else image_counter
    fps = profilers["inference"].get_fps(final_count)
    avg_ms = profilers["inference"].get_avg_time_ms(batch_count)

    # 1. TIMING SUMMARY
    logger.info("=" * 60)
    logger.info("ANOMAVISION PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Setup time:                {profilers['setup'].accumulated_time * 1000:.2f} ms")
    logger.info(f"Model loading time:        {profilers['model_loading'].accumulated_time * 1000:.2f} ms")
    logger.info(f"Data loading time:         {profilers['data_loading'].accumulated_time * 1000:.2f} ms")
    logger.info(f"Inference time:            {profilers['inference'].accumulated_time * 1000:.2f} ms")
    logger.info(f"Postprocessing time:       {profilers['postprocessing'].accumulated_time * 1000:.2f} ms")
    logger.info(f"Visualization time:        {profilers['visualization'].accumulated_time * 1000:.2f} ms")
    logger.info(f"Total pipeline time:       {total_pipeline_time * 1000:.2f} ms")
    logger.info("=" * 60)

    # 2. INFERENCE PERFORMANCE
    logger.info("=" * 60)
    logger.info("ANOMAVISION INFERENCE PERFORMANCE")
    logger.info("=" * 60)
    if fps > 0:
        logger.info(f"Pure inference FPS:        {fps:.2f} images/sec")
    if avg_ms > 0:
        logger.info(f"Average inference time:    {avg_ms:.2f} ms/batch")

    if batch_count > 0:
        batch_size = config.get('batch_size', 1) or 1
        throughput = fps * (final_count / batch_count) if batch_count else 0
        logger.info(f"Throughput:                {throughput:.1f} images/sec (batch size: {batch_size})")
    logger.info("=" * 60)

    metrics = {
        "fps": fps,
        "avg_inference_ms": avg_ms,
        "total_time_s": total_pipeline_time,
        "total_images": final_count
    }

    return metrics, results_accumulator


def main():
    try:
        parser, _ = parse_args()
        args = parser.parse_args()

        metrics, results = run_inference(args)

        # If running as script, maybe we want to print summary or save results to file?
        # For now, logging handles the output.

        exit(0)
    except KeyboardInterrupt:
        logger = get_logger("anomavision.detect")
        logger.info("Process interrupted by user")
        exit(1)
    except Exception as e:
        logger = get_logger("anomavision.detect")
        logger.error(f"Process failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
