"""
Run Anomaly detection inference on images using various model formats.
Usage - formats:
    $ python detect.py --model model.pt                     # PyTorch
                                   model.torchscript        # TorchScript
                                   model.onnx               # ONNX Runtime
                                   model_openvino           # OpenVINO
                                   model.engine             # TensorRT
                                   model.hef                # Hailo HEF
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
    parser.add_argument("--config", type=str, default=None, help="Path to config.yml/.json")
    parser.add_argument("--img_path", default=None, type=str, help="Path to the dataset folder containing test images.")
    parser.add_argument("--model_data_path", type=str, default="./distributions", help="Directory containing model files.")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm name (e.g., padim, patchcore).")
    parser.add_argument("--model", type=str, default=None, help="Model file (.pt, .onnx, .hef, etc.). A path outside model_data_path is also accepted.")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"], help="Device to run inference on.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference. Hailo currently requires 1.")
    parser.add_argument("--thresh", type=float, default=None, help="Threshold for anomaly classification")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes for data loading.")
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for faster GPU transfers.")
    parser.add_argument("--enable_visualization", action="store_true", default=None, help="Enable visualization of results.")
    parser.add_argument("--save_visualizations", action="store_true", default=None, help="Save visualization images to disk.")
    parser.add_argument("--viz_output_dir", type=str, default=None, help="Directory to save visualization images.")
    parser.add_argument("--run_name", default=None, help="experiment name for this inference run")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing run directory without auto-incrementing")
    parser.add_argument("--viz_alpha", type=float, default=None, help="Alpha value for heatmap overlay.")
    parser.add_argument("--viz_padding", type=int, default=None, help="Padding for boundary visualization.")
    parser.add_argument("--viz_color", type=str, default=None, help='RGB color for highlighting (comma-separated, e.g., "128,0,128").')
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    parser.add_argument("--detailed_timing", action="store_true", help="Enable detailed timing measurements.")
    return parser


def _resolve_model_path(config) -> str:
    """Resolve a model argument without changing the legacy distribution layout."""
    model = str(config.model)
    direct = Path(model).expanduser()
    if direct.is_file():
        return str(direct.resolve())

    # Keep the existing distribution layout for bare model names.
    return str(
        Path(config.model_data_path)
        / str(config.algorithm)
        / str(config.class_name)
        / str(config.run_name)
        / model
    )


def run_inference(args):
    if args.config is not None:
        cfg = load_config(str(args.config))
    else:
        potential_paths = []
        if args.model_data_path:
            potential_paths.append(Path(args.model_data_path) / "config.yml")
        cfg = {}
        for path in potential_paths:
            if path.exists():
                cfg = load_config(str(path))
                break

    config = edict(merge_config(args, cfg))
    config.thresh = resolve_threshold(config)
    algorithm_name = str(config.get("algorithm", "")).lower()

    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger("anomavision.detect")
    stream_mode = config.get("stream_mode", False)
    logger.info(f"Streaming mode: {stream_mode}")

    try:
        viz_color = tuple(map(int, config.viz_color.split(","))) if config.viz_color else (128, 0, 128)
        if len(viz_color) != 3:
            raise ValueError
    except (ValueError, AttributeError):
        logger.warning(f"Invalid color format '{getattr(config, 'viz_color', 'None')}'. Using default (128,0,128)")
        viz_color = (128, 0, 128)

    resize = _shape(config.resize)
    crop_size = _shape(config.crop_size)
    normalize = config.get("normalize", True)
    logger.info("Image processing: resize=%s, crop=%s, norm=%s", resize, crop_size, normalize)

    if not config.get("img_path") and not stream_mode:
        raise ValueError("img_path is required (via --img_path or config) when stream_mode is False")
    if not config.get("model"):
        raise ValueError("model is required (via --model or config)")

    profilers = {name: Profiler() for name in ["setup", "model_loading", "data_loading", "inference", "postprocessing", "visualization"]}
    results_accumulator = {"scores": [], "classifications": [], "images": [] if not stream_mode else None}
    total_start_time = time.time()

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

    with profilers["model_loading"]:
        model_path = _resolve_model_path(config)
        logger.info(f"Loading model: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            model_type = ModelType.from_extension(model_path)
            if model_type == ModelType.HEF and int(config.batch_size) != 1:
                raise ValueError("Hailo HEF inference currently requires --batch_size 1 for deterministic backend comparison")
            model = ModelWrapper(model_path, device_str)
            logger.info(f"Model loaded: {model_type.value.upper()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    RESULTS_PATH = None
    if config.get("save_visualizations", False):
        viz_output_dir = config.get("viz_output_dir", "./visualizations/")
        RESULTS_PATH = increment_path(
            Path(viz_output_dir) / config.algorithm / config.class_name / model_type.value.upper() / config.run_name,
            exist_ok=config.get("overwrite", False), mkdir=True,
        )
        logger.info(f"Visualization output: {RESULTS_PATH}")

    with profilers["data_loading"]:
        try:
            if not stream_mode:
                test_dataset = anomavision.AnodetDataset(
                    DATASET_PATH, resize=resize, crop_size=crop_size, normalize=normalize,
                    mean=config.norm_mean, std=config.norm_std,
                )
                num_workers = int(config.get("num_workers", 0))
                pin_memory = bool(config.get("pin_memory", False))
            else:
                source = StreamSourceFactory.create(config.stream_source)
                source.connect()
                test_dataset = StreamDataset(
                    source=source, resize=resize, crop_size=crop_size, normalize=normalize,
                    mean=config.norm_mean, std=config.norm_std, max_frames=config.get("stream_max_frames"),
                )
                num_workers, pin_memory = 0, False
            test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=pin_memory)
            try:
                total_images = len(test_dataset)
                logger.info(f"Total images: {total_images}")
            except TypeError:
                total_images = None
                logger.info("Streaming mode (infinite/unknown length)")
        except Exception as e:
            logger.error(f"Failed to create dataloader: {e}")
            raise

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

    batch_count = 0
    image_counter = 0
    try:
        for batch_idx, (batch, images, _, _) in enumerate(test_dataloader):
            batch_count += 1
            image_counter += batch.shape[0]
            if device_str == "cuda":
                batch = batch.half()
            batch = batch.to(device_str)
            with profilers["inference"]:
                try:
                    image_scores, score_maps = model.predict(batch)
                except Exception as e:
                    logger.error(f"Inference failed batch {batch_idx}: {e}")
                    continue
            with profilers["postprocessing"]:
                try:
                    score_maps = adaptive_gaussian_blur(score_maps, kernel_size=33, sigma=4)
                    if config.thresh is not None:
                        is_anomaly = anomavision.classification(image_scores, config.thresh)
                    else:
                        is_anomaly = np.zeros_like(image_scores)
                    if algorithm_name == "patchcore":
                        localization_masks = make_localization_mask(score_maps, is_anomaly, quantile=0.90)
                    else:
                        localization_masks = anomavision.classification(score_maps, config.thresh) if config.thresh is not None else np.zeros_like(score_maps)
                    if not stream_mode:
                        results_accumulator["scores"].extend(np.asarray(image_scores).reshape(-1).tolist())
                        results_accumulator["classifications"].extend(np.asarray(is_anomaly).reshape(-1).tolist())
                        results_accumulator["images"].extend(images)
                except Exception as e:
                    logger.error(f"Postprocessing failed batch {batch_idx}: {e}")
                    continue

            if config.enable_visualization:
                with profilers["visualization"]:
                    try:
                        boundary_images = anomavision.visualization.framed_boundary_images(images, localization_masks, is_anomaly, padding=config.get("viz_padding", 40))
                        heatmap_images = anomavision.visualization.heatmap_images(images, score_maps, masks=localization_masks, alpha=config.get("viz_alpha", 0.5))
                        highlighted_images = anomavision.visualization.highlighted_images([images[i] for i in range(len(images))], localization_masks, color=viz_color)
                        for img_id in range(len(images)):
                            if config.save_visualizations and RESULTS_PATH:
                                try:
                                    fig, axs = plt.subplots(1, 4, figsize=(16, 8))
                                    fig.suptitle(f"Result - Batch {batch_idx} Img {img_id} - score={float(np.asarray(image_scores).reshape(-1)[img_id]):.6f}", fontsize=14)
                                    axs[0].imshow(images[img_id]); axs[0].set_title("Original"); axs[0].axis("off")
                                    axs[1].imshow(boundary_images[img_id]); axs[1].set_title("Boundary"); axs[1].axis("off")
                                    axs[2].imshow(heatmap_images[img_id]); axs[2].set_title("Heatmap"); axs[2].axis("off")
                                    axs[3].imshow(highlighted_images[img_id]); axs[3].set_title("Highlighted"); axs[3].axis("off")
                                    plt.savefig(os.path.join(RESULTS_PATH, f"batch_{batch_idx}_img_{img_id}.png"), dpi=100, bbox_inches="tight")
                                    plt.close(fig)
                                except Exception as e:
                                    logger.warning(f"Viz save failed: {e}")
                    except Exception as e:
                        logger.error(f"Visualization failed batch {batch_idx}: {e}")
    finally:
        logger.info("Closing model...")
        model.close()
        if stream_mode:
            try:
                test_dataset.close()
            except Exception:
                pass

    total_pipeline_time = time.time() - total_start_time
    final_count = image_counter
    fps = final_count / total_pipeline_time if total_pipeline_time > 0 else 0.0
    metrics = {
        "total_images": final_count,
        "total_time_s": total_pipeline_time,
        "fps": fps,
        "avg_inference_ms": profilers["inference"].total_time * 1000 / max(batch_count, 1),
        "model": model_path,
        "model_type": model_type.value,
    }
    logger.info("Detection complete: %d images, %.2fs, %.2f FPS", final_count, total_pipeline_time, fps)
    logger.info("Scores: %s", results_accumulator["scores"])
    return metrics, results_accumulator


def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    return run_inference(args)


if __name__ == "__main__":
    main()
