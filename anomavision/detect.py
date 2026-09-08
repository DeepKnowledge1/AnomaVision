"""
Run Anomaly detection inference on images using various model formats.
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
from anomavision.actions.integration import create_action_dispatcher, dispatch_inference_results
from anomavision.config import _shape, load_config
from anomavision.datasets.StreamDataset import StreamDataset
from anomavision.datasets.StreamSourceFactory import StreamSourceFactory
from anomavision.general import Profiler, determine_device, increment_path
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType
from anomavision.utils import adaptive_gaussian_blur, get_logger, make_localization_mask, merge_config, resolve_threshold, setup_logging

matplotlib.use("Agg")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run anomaly detection inference using trained models.", add_help=add_help)
    parser.add_argument("--config", type=str, default=None, help="Path to config.yml/.json")
    parser.add_argument("--img_path", default=None, type=str, help="Path to the dataset folder containing test images.")
    parser.add_argument("--model_data_path", type=str, default="./distributions", help="Directory containing model files.")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm name (e.g., padim, patchcore).")
    parser.add_argument("--model", type=str, default=None, help="Model file (.pt, .onnx, .engine, etc.)")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--thresh", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--enable_visualization", action="store_true", default=None)
    parser.add_argument("--save_visualizations", action="store_true", default=None)
    parser.add_argument("--viz_output_dir", type=str, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--viz_alpha", type=float, default=None)
    parser.add_argument("--viz_padding", type=int, default=None)
    parser.add_argument("--viz_color", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--detailed_timing", action="store_true")
    return parser


def run_inference(args):
    """Execute the inference pipeline and optional industrial actions."""
    if args.config is not None:
        cfg = load_config(str(args.config))
    else:
        cfg = {}
        if args.model_data_path:
            path = Path(args.model_data_path) / "config.yml"
            if path.exists():
                cfg = load_config(str(path))

    config = edict(merge_config(args, cfg))
    config.thresh = resolve_threshold(config)
    algorithm_name = str(config.get("algorithm", "")).lower()

    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger("anomavision.detect")

    stream_mode = config.get("stream_mode", False)
    logger.info("Streaming mode: %s", stream_mode)

    try:
        viz_color = tuple(map(int, config.viz_color.split(","))) if config.viz_color else (128, 0, 128)
        if len(viz_color) != 3:
            raise ValueError
    except (ValueError, AttributeError):
        logger.warning("Invalid visualization color. Using default (128,0,128)")
        viz_color = (128, 0, 128)

    resize = _shape(config.resize)
    crop_size = _shape(config.crop_size)
    normalize = config.get("normalize", True)
    logger.info("Image processing: resize=%s, crop=%s, norm=%s", resize, crop_size, normalize)

    if not config.get("img_path") and not stream_mode:
        raise ValueError("img_path is required when stream_mode is False")
    if not config.get("model"):
        raise ValueError("model is required")

    profilers = {name: Profiler() for name in ["setup", "model_loading", "data_loading", "inference", "postprocessing", "visualization"]}
    results_accumulator = {"scores": [], "classifications": [], "images": [] if not stream_mode else None}
    total_start_time = time.time()

    # Industrial actions are explicitly disabled by default.
    action_dispatcher = None
    actions_enabled = bool(config.get("actions_enabled", False))
    actions_config = config.get("actions")
    if actions_enabled:
        if not actions_config:
            logger.warning("Industrial actions are enabled, but no actions are configured")
        else:
            logger.info("Industrial actions enabled: %d action(s)", len(actions_config))
            action_dispatcher = create_action_dispatcher(
                actions_config,
                logger=logger,
                fail_fast=bool(config.get("actions_fail_fast", False)),
            )
    else:
        logger.info("Industrial actions: disabled")

    try:
        with profilers["setup"]:
            if not stream_mode:
                DATASET_PATH = os.path.realpath(config.img_path)
                logger.info("Dataset path: %s", DATASET_PATH)
            else:
                DATASET_PATH = None
                src = config.get("stream_source", {})
                logger.info("Streaming source type: %s", src.get("type", "unknown"))

            MODEL_DATA_PATH = os.path.realpath(config.model_data_path)
            device_str = determine_device(config.device)
            logger.info("Device: %s", device_str)
            if device_str == "cuda" and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True

        with profilers["model_loading"]:
            model_path = os.path.join(MODEL_DATA_PATH, config.algorithm, config.class_name, config.run_name, config.model)
            logger.info("Loading model: %s", model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = ModelWrapper(model_path, device_str)
            model_type = ModelType.from_extension(model_path)
            logger.info("Model loaded: %s", model_type.value.upper())

        RESULTS_PATH = None
        if config.get("save_visualizations", False):
            RESULTS_PATH = increment_path(
                Path(config.get("viz_output_dir", "./visualizations/")) / config.algorithm / config.class_name / model_type.value.upper() / config.run_name,
                exist_ok=config.get("overwrite", False),
                mkdir=True,
            )
            logger.info("Visualization output: %s", RESULTS_PATH)

        with profilers["data_loading"]:
            if not stream_mode:
                test_dataset = anomavision.AnodetDataset(DATASET_PATH, resize=resize, crop_size=crop_size, normalize=normalize, mean=config.norm_mean, std=config.norm_std)
                num_workers = int(config.get("num_workers", 0))
                pin_memory = bool(config.get("pin_memory", False))
            else:
                source = StreamSourceFactory.create(config.stream_source)
                source.connect()
                test_dataset = StreamDataset(source=source, resize=resize, crop_size=crop_size, normalize=normalize, mean=config.norm_mean, std=config.norm_std, max_frames=config.get("stream_max_frames"))
                num_workers = 0
                pin_memory = False

            test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=pin_memory)
            try:
                total_images = len(test_dataset)
                logger.info("Total images: %s", total_images)
            except TypeError:
                total_images = None
                logger.info("Streaming mode (infinite/unknown length)")

        try:
            first = next(iter(test_dataloader))
            first_batch = first[0]
            if device_str == "cuda":
                first_batch = first_batch.half()
            model.warmup(batch=first_batch.to(device_str), runs=2)
            logger.info("Warm-up complete.")
        except StopIteration:
            logger.warning("Dataset empty; skipping warm-up.")
        except Exception as e:
            logger.warning("Warm-up skipped: %s", e)

        batch_count = 0
        image_counter = 0
        for batch_idx, (batch, images, _, _) in enumerate(test_dataloader):
            batch_count += 1
            batch_start_frame = image_counter
            image_counter += batch.shape[0]
            if device_str == "cuda":
                batch = batch.half()
            batch = batch.to(device_str)

            with profilers["inference"]:
                try:
                    image_scores, score_maps = model.predict(batch)
                except Exception as e:
                    logger.error("Inference failed batch %s: %s", batch_idx, e)
                    continue

            with profilers["postprocessing"]:
                try:
                    score_maps = adaptive_gaussian_blur(score_maps, kernel_size=33, sigma=4)
                    is_anomaly = anomavision.classification(image_scores, config.thresh) if config.thresh is not None else np.zeros_like(image_scores)
                    if algorithm_name == "patchcore":
                        localization_masks = make_localization_mask(score_maps, is_anomaly, quantile=0.90)
                    else:
                        localization_masks = anomavision.classification(score_maps, config.thresh) if config.thresh is not None else np.zeros_like(score_maps)

                    if not stream_mode:
                        results_accumulator["scores"].extend(image_scores.tolist())
                        results_accumulator["classifications"].extend(is_anomaly.tolist())
                        results_accumulator["images"].extend(images)

                    if action_dispatcher is not None:
                        dispatch_inference_results(
                            action_dispatcher, image_scores, is_anomaly,
                            source_id=str(config.get("source_id", "anomavision")),
                            model_name=str(config.get("algorithm", "unknown")),
                            model_version=config.get("model_version"),
                            frame_start=batch_start_frame,
                            images=images,
                        )
                except Exception as e:
                    logger.error("Postprocessing failed batch %s: %s", batch_idx, e)
                    continue

            if config.enable_visualization:
                with profilers["visualization"]:
                    try:
                        boundary_images = anomavision.visualization.framed_boundary_images(images, localization_masks, is_anomaly, padding=config.get("viz_padding", 40))
                        heatmap_images = anomavision.visualization.heatmap_images(images, score_maps, masks=localization_masks, alpha=config.get("viz_alpha", 0.5))
                        highlighted_images = anomavision.visualization.highlighted_images([images[i] for i in range(len(images))], localization_masks, color=viz_color)
                        for img_id in range(len(images)):
                            if config.save_visualizations and RESULTS_PATH:
                                fig, axs = plt.subplots(1, 4, figsize=(16, 8))
                                fig.suptitle(f"Result - Batch {batch_idx} Img {img_id}", fontsize=14)
                                for ax, image, title in zip(axs, [images[img_id], boundary_images[img_id], heatmap_images[img_id], highlighted_images[img_id]], ["Original", "Boundary", "Heatmap", "Highlighted"]):
                                    ax.imshow(image); ax.set_title(title); ax.axis("off")
                                plt.savefig(os.path.join(RESULTS_PATH, f"batch_{batch_idx}_img_{img_id}.png"), dpi=100, bbox_inches="tight")
                                plt.close(fig)
                    except Exception as e:
                        logger.error("Visualization failed batch %s: %s", batch_idx, e)
    finally:
        if action_dispatcher is not None:
            logger.info("Disconnecting industrial actions...")
            action_dispatcher.disconnect_all()
        if "model" in locals():
            logger.info("Closing model...")
            model.close()
        if stream_mode and "test_dataset" in locals():
            try:
                test_dataset.close()
            except Exception:
                pass

    total_pipeline_time = time.time() - total_start_time
    final_count = total_images if (not stream_mode and total_images) else image_counter
    fps = profilers["inference"].get_fps(final_count)
    avg_ms = profilers["inference"].get_avg_time_ms(batch_count)
    logger.info("ANOMAVISION PERFORMANCE SUMMARY")
    logger.info("Total pipeline time: %.2f ms", total_pipeline_time * 1000)
    if fps > 0:
        logger.info("Pure inference FPS: %.2f images/sec", fps)
    if avg_ms > 0:
        logger.info("Average inference time: %.2f ms/batch", avg_ms)
    return {"fps": fps, "avg_inference_ms": avg_ms, "total_time_s": total_pipeline_time, "total_images": final_count}, results_accumulator


def main(args=None):
    try:
        if args is None:
            args = create_parser().parse_args()
        run_inference(args)
        exit(0)
    except KeyboardInterrupt:
        get_logger("anomavision.detect").info("Process interrupted by user")
        exit(1)
    except Exception as e:
        get_logger("anomavision.detect").error(f"Process failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
