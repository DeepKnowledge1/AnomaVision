import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import anomavision
from anomavision.config import load_config
from anomavision.general import Profiler, determine_device
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType
from anomavision.utils import (
    adaptive_gaussian_blur,
    compute_metrics,
    find_optimal_threshold,
    get_logger,
    merge_config,
    setup_logging,
)
from anomavision_downloader import (
    MODELS_DIR,
    _warn_missing_path,
    ensure_images,
    ensure_models,
    get_mvtec_dir,
)


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AnomaVision anomaly detection model performance.",
        add_help=add_help,
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yml/.json"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        help="Path to the dataset folder containing test images.",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="bottle",
        help="Class name for MVTec dataset evaluation.",
    )

    # Model parameters
    parser.add_argument(
        "--model_data_path",
        type=str,
        default=None,
        help="Directory containing AnomaVision model files.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm name (e.g., padim, patchcore).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model filename (.pt, .onnx, .engine)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device to run evaluation on.",
    )

    # Evaluation parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation",
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
        "--thresh",
        type=float,
        default=None,
        help="Threshold for accuracy calculation (optional).",
    )

    # Visualization parameters
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        help="Enable visualization of evaluation results.",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save evaluation visualization images to disk.",
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default=None,
        help="Directory to save visualization images.",
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

    return parser


def evaluate_model_with_wrapper(
    model_wrapper, test_dataloader, logger, evaluation_profiler, detailed_timing=False
):
    """
    Evaluate model using the ModelWrapper inference interface.
    Returns raw predictions for post-processing/metric calculation.
    """
    all_images = []
    all_image_classifications_target = []
    all_masks_target = []
    all_image_scores = []
    all_score_maps = []

    device_str = determine_device("cpu")
    logger.info(f"Starting evaluation on {len(test_dataloader.dataset)} images")

    try:
        for batch_idx, (batch, images, image_targets, mask_targets) in enumerate(
            test_dataloader
        ):
            batch = batch.to(device_str)

            with evaluation_profiler:
                try:
                    # ModelWrapper returns (scores, maps)
                    image_scores, score_maps = model_wrapper.predict(batch)
                except Exception as e:
                    logger.error(f"Inference failed for batch {batch_idx}: {e}")
                    continue

            # Collect results
            all_images.extend(images)
            all_image_classifications_target.extend(
                image_targets.numpy()
                if hasattr(image_targets, "numpy")
                else image_targets
            )
            all_masks_target.extend(
                mask_targets.numpy() if hasattr(mask_targets, "numpy") else mask_targets
            )

            # Handle Tensor vs Numpy returns
            if isinstance(image_scores, np.ndarray):
                all_image_scores.extend(image_scores.tolist())
                all_score_maps.extend(score_maps)
            else:
                all_image_scores.extend(image_scores.cpu().numpy().tolist())
                all_score_maps.extend(score_maps.cpu().numpy())

    except Exception as e:
        logger.error(f"Evaluation loop failed: {e}")
        raise

    # Convert to numpy arrays
    return (
        np.array(all_images),
        np.array(all_image_classifications_target),
        (
            np.squeeze(np.array(all_masks_target), axis=1)
            if len(all_masks_target) > 0
            else np.array([])
        ),
        np.array(all_image_scores),
        np.array(all_score_maps),
    )


def run_evaluation(args):
    """
    Executes the full evaluation pipeline and prints detailed report.
    """
    # Load and merge config
    if args.config:
        cfg = load_config(str(args.config))
    else:
        config_path = Path(args.model_data_path) / "config.yml"
        cfg = load_config(str(config_path)) if config_path.exists() else {}

    config = edict(merge_config(args, cfg))

    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger("anomavision.eval")

    # Profilers
    profilers = {
        "setup": Profiler(),
        "model_loading": Profiler(),
        "data_loading": Profiler(),
        "evaluation": Profiler(),
        "visualization": Profiler(),
    }

    model_type = None  # To capture for reporting later

    # Setup Phase
    with profilers["setup"]:
        # ── Download dataset if path not set or missing ───────────────────────
        if not config.dataset_path or not os.path.isdir(str(config.dataset_path)):
            _warn_missing_path(
                "Dataset", str(config.dataset_path or ""), str(get_mvtec_dir())
            )
            logger.warning(
                "dataset_path not set or missing — ensuring cached images ..."
            )
            ensure_images()
            config.dataset_path = str(get_mvtec_dir())  # ~/.anomavision/mvtec/
            logger.info("dataset_path set to cache: %s", config.dataset_path)

        # ── Download models if path not set or model file missing ─────────────
        model_path = (
            os.path.join(
                config.model_data_path,
                config.algorithm,
                config.class_name,
                config.run_name,
                config.model,
            )
            if config.model_data_path
            else None
        )

        if not model_path or not os.path.exists(model_path):
            _warn_missing_path(
                "Model", str(config.model_data_path or ""), str(MODELS_DIR)
            )
            logger.warning(
                "Model not found at '%s' — ensuring cached models ...", model_path
            )
            ensure_models()
            config.model_data_path = str(MODELS_DIR)  # ~/.anomavision/models/
            logger.info("model_data_path set to cache: %s", config.model_data_path)
        # ─────────────────────────────────────────────────────────────────────

        DATASET_PATH = (
            os.path.realpath(config.dataset_path) if config.dataset_path else None
        )
        MODEL_DATA_PATH = os.path.realpath(config.model_data_path)
        device_str = determine_device(config.device)

        if not DATASET_PATH:
            raise ValueError("Dataset path is required for evaluation.")

        if device_str == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    # Load Model Phase
    with profilers["model_loading"]:
        model_path = os.path.join(
            MODEL_DATA_PATH,
            config.algorithm,
            config.class_name,
            config.run_name,
            config.model,
        )
        logger.info(f"Loading model: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            model_wrapper = ModelWrapper(model_path, device_str)
            model_type = ModelType.from_extension(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # Data Loading Phase
    with profilers["data_loading"]:
        try:
            test_dataset = anomavision.MVTecDataset(
                DATASET_PATH,
                config.class_name,
                is_train=False,
                resize=config.resize,
                crop_size=config.crop_size,
                normalize=config.normalize,
                mean=config.norm_mean,
                std=config.norm_std,
            )

            batch_size = int(config.batch_size) if config.batch_size else 1

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=config.num_workers if config.num_workers else 0,
                pin_memory=config.pin_memory and device_str == "cuda",
                shuffle=False,
            )
        except Exception as e:
            logger.error(f"Failed to create dataloader: {e}")
            model_wrapper.close()
            raise

    # Inference Phase
    try:
        images, labels, masks, scores, maps = evaluate_model_with_wrapper(
            model_wrapper,
            test_dataloader,
            logger,
            profilers["evaluation"],
            config.detailed_timing,
        )

        # Post-process maps
        maps = adaptive_gaussian_blur(maps, kernel_size=33, sigma=4)

    except Exception:
        raise
    finally:
        model_wrapper.close()

    # Compute Metrics
    if config.thresh is None:
        best_thresh, _ = find_optimal_threshold(labels, scores)
    else:
        best_thresh = config.thresh

    metrics = compute_metrics(labels, scores, thresh=best_thresh)

    # Add timing metrics
    total_images = len(test_dataset)
    metrics["inference_fps"] = profilers["evaluation"].get_fps(total_images)
    metrics["inference_time_total_s"] = profilers["evaluation"].accumulated_time

    # Visualization Phase
    if config.enable_visualization:
        with profilers["visualization"]:
            try:
                anomavision.visualize_eval_data(
                    labels,
                    masks.astype(np.uint8).flatten(),
                    scores,
                    maps.flatten(),
                )
                if config.save_visualizations:
                    os.makedirs(config.viz_output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    viz_filepath = os.path.join(
                        config.viz_output_dir,
                        f"eval_{config.class_name}_{timestamp}.png",
                    )
                    plt.savefig(viz_filepath, dpi=300, bbox_inches="tight")
                    logger.info(f"Visualization saved: {viz_filepath}")
            except Exception as e:
                logger.error(f"Visualization failed: {e}")

    # ========================================================================
    # GENERATE DETAILED REPORT
    # ========================================================================

    evaluation_fps = profilers["evaluation"].get_fps(total_images)
    avg_evaluation_time = profilers["evaluation"].get_avg_time_ms(len(test_dataloader))

    # 1. TIMING SUMMARY
    logger.info("=" * 60)
    logger.info("ANOMAVISION EVALUATION PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Setup time:                {profilers['setup'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Model loading time:        {profilers['model_loading'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Data loading time:         {profilers['data_loading'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Evaluation time:           {profilers['evaluation'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Visualization time:        {profilers['visualization'].accumulated_time * 1000:.2f} ms"
    )

    # 2. PERFORMANCE METRICS
    logger.info("=" * 60)
    logger.info("ANOMAVISION EVALUATION PERFORMANCE")
    logger.info("=" * 60)
    if evaluation_fps > 0:
        logger.info(f"Pure evaluation FPS:       {evaluation_fps:.2f} images/sec")
    if avg_evaluation_time > 0:
        logger.info(f"Average evaluation time:   {avg_evaluation_time:.2f} ms/batch")

    if len(test_dataloader) > 0:
        images_per_batch = total_images / len(test_dataloader)
        logger.info(
            f"Evaluation throughput:     {evaluation_fps * images_per_batch:.1f} images/sec (batch size: {batch_size})"
        )

    # 3. EVALUATION SUMMARY
    logger.info("=" * 60)
    logger.info("ANOMAVISION EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.class_name}")
    logger.info(f"Total images evaluated: {total_images}")
    logger.info(f"Model type: {model_type.value.upper() if model_type else 'UNKNOWN'}")
    logger.info(f"Device: {device_str}")
    logger.info(
        f"Image processing: resize={config.resize}, crop_size={config.crop_size}, normalize={config.normalize}"
    )

    logger.info("=" * 60)
    logger.info("ANOMAVISION DETECTION METRICS")
    logger.info("=" * 60)

    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k.replace('_',' ').title():<28} {v:.6f}")
        else:
            logger.info(f"{k.replace('_',' ').title():<28} {v}")

    logger.info("=" * 60)

    logger.info("AnomaVision anomaly detection model evaluation completed successfully")

    # Return results for MLOps usage
    raw_results = {
        "images": images,
        "labels": labels,
        "masks": masks,
        "scores": scores,
        "maps": maps,
    }

    return metrics, raw_results


def main(args=None):
    try:
        if args is None:
            args = create_parser().parse_args()

        run_evaluation(args)

    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        get_logger("anomavision.eval").error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
