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
from anomavision.config import _shape, load_config

import anomavision
from anomavision.utils import (
    adaptive_gaussian_blur,
    get_logger,
    merge_config,
    setup_logging,
)

from anomavision.general import Profiler, determine_device, increment_path
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType

matplotlib.use("Agg")  # non-interactive, faster PNG writing


# Updated imports to use the inference modules


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
        default="padim_model.onnx",
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

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config is not None:
        cfg = load_config(str(args.config))
    else:
        cfg = load_config(str(Path(args.model_data_path) / "config.yml"))

    # Merge config with CLI args
    config = edict(merge_config(args, cfg))

    # Setup logging first
    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger("anomavision.detect")  # Force it into anomavision hierarchy

    # Parse visualization color
    try:
        viz_color = tuple(map(int, config.viz_color.split(",")))
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

    # Log image processing configuration
    logger.info(
        "Image processing config: resize=%s, crop_size=%s, normalize=%s",
        resize,
        crop_size,
        normalize,
    )
    if normalize:
        logger.info(
            "Normalization: mean=%s, std=%s",
            config.get("norm_mean"),
            config.get("norm_std"),
        )

    # Validation
    if not config.get("img_path"):
        logger.error("img_path is required (via --img_path or config)")
        return 1

    if not config.get("model"):
        logger.error("model is required (via --model or config)")
        return 1

    # Initialize AnomaVision profilers for different pipeline stages
    anomavision_profilers = {
        "setup": Profiler(),
        "model_loading": Profiler(),
        "data_loading": Profiler(),
        "inference": Profiler(),  # Core anomaly detection timing
        "postprocessing": Profiler(),  # Classification and scoring timing
        "visualization": Profiler(),  # Anomaly visualization timing
    }
    logger.info("Starting AnomaVision anomaly detection inference process")
    logger.info(f"Final config: {dict(config)}")

    total_start_time = time.time()

    # AnomaVision setup phase
    with anomavision_profilers["setup"]:
        DATASET_PATH = os.path.realpath(config.img_path)
        MODEL_DATA_PATH = os.path.realpath(config.model_data_path)
        device_str = determine_device(config.device)
        logger.info(f"AnomaVision selected device: {device_str}")
        logger.info(f"AnomaVision dataset path: {DATASET_PATH}")
        logger.info(f"AnomaVision model data path: {MODEL_DATA_PATH}")

        if device_str == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("AnomaVision CUDA available, enabled cuDNN benchmark")
            logger.info(f"AnomaVision CUDA device: {torch.cuda.get_device_name()}")
            logger.info(
                f"AnomaVision CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

    # AnomaVision model loading phase
    with anomavision_profilers["model_loading"]:
        model_path = os.path.join(MODEL_DATA_PATH, config.model)
        logger.info(f"Loading AnomaVision model from: {model_path}")

        if not os.path.exists(model_path):
            logger.error(f"AnomaVision model file not found: {model_path}")
            raise FileNotFoundError(f"AnomaVision model file not found: {model_path}")

        try:
            model = ModelWrapper(model_path, device_str)
            model_type = ModelType.from_extension(model_path)
            logger.info(
                f"AnomaVision model loaded successfully. Type: {model_type.value.upper()}"
            )
        except Exception as e:
            logger.error(f"Failed to load AnomaVision model: {e}")
            raise

    # Create output directory for AnomaVision visualizations if needed
    RESULTS_PATH = None
    if config.get("save_visualizations", False):
        run_name = config.run_name
        viz_output_dir = config.get("viz_output_dir", "./visualizations/")
        RESULTS_PATH = increment_path(
            Path(viz_output_dir) / model_type.value.upper() / run_name,
            exist_ok=config.get("overwrite", False),
            mkdir=True,
        )
        logger.info(f"AnomaVision visualization output directory: {RESULTS_PATH}")

    # AnomaVision data loading phase
    with anomavision_profilers["data_loading"]:
        logger.info("Creating AnomaVision dataset and dataloader")
        try:
            # Create dataset with configurable image processing parameters
            test_dataset = anomavision.AnodetDataset(
                DATASET_PATH,
                resize=resize,
                crop_size=crop_size,
                normalize=normalize,
                mean=config.norm_mean,
                std=config.norm_std,
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.num_workers > 0,
            )
            logger.info(
                f"AnomaVision dataset created successfully. Total images: {len(test_dataset)}"
            )
            logger.info(
                f"AnomaVision batch size: {config.get('batch_size', 1)}, Number of batches: {len(test_dataloader)}"
            )
        except Exception as e:
            logger.error(f"Failed to create AnomaVision dataset/dataloader: {e}")
            raise

    logger.info(
        f"Processing {len(test_dataset)} images using AnomaVision {model_type.value.upper()}"
    )

    # ---- Warm-up
    try:
        first = next(iter(test_dataloader))  # (batch, images, _, _)
        first_batch = first[0].to(device_str)
        model.warmup(batch=first_batch, runs=2)
        logger.info(
            "AnomaVision warm-up done with first batch %s.", tuple(first_batch.shape)
        )
    except StopIteration:
        logger.warning("Dataset empty; skipping warm-up.")
    except Exception as e:
        logger.warning(f"Warm-up skipped due to error: {e}")

    # ---- End warm-up

    # AnomaVision batch processing pipeline
    batch_count = 0
    try:
        for batch_idx, (batch, images, _, _) in enumerate(test_dataloader):
            batch = batch.to(device_str)
            batch_count += 1
            logger.debug(
                f"Processing AnomaVision batch {batch_idx + 1}/{len(test_dataloader)}"
            )

            # AnomaVision core inference phase
            with anomavision_profilers["inference"] as inference_prof:
                try:
                    image_scores, score_maps = model.predict(batch)

                    logger.debug(
                        f"AnomaVision image scores shape: {image_scores.shape}, Score maps shape: {score_maps.shape}"
                    )
                except Exception as e:
                    logger.error(
                        f"AnomaVision inference failed for batch {batch_idx}: {e}"
                    )
                    continue

            logger.info(
                f"AnomaVision batch shape: {batch.shape}, Inference completed in {inference_prof.elapsed_time * 1000:.2f} ms"
            )

            # AnomaVision postprocessing phase - anomaly classification
            with anomavision_profilers["postprocessing"]:
                try:
                    score_maps = adaptive_gaussian_blur(
                        score_maps, kernel_size=33, sigma=4
                    )

                    score_map_classifications = anomavision.classification(
                        score_maps, config.thresh
                    )
                    image_classifications = anomavision.classification(
                        image_scores, config.thresh
                    )

                    # Convert for AnomaVision logging
                    if isinstance(image_scores, np.ndarray):
                        image_scores_list = image_scores.tolist()
                        image_classifications_list = (
                            image_classifications.numpy().tolist()
                            if hasattr(image_classifications, "numpy")
                            else image_classifications.tolist()
                        )
                    else:
                        image_scores_list = image_scores.tolist()
                        image_classifications_list = image_classifications.tolist()

                    logger.debug(
                        f"AnomaVision batch {batch_idx + 1}: Scores: {image_scores_list}, Classifications: {image_classifications_list}"
                    )
                except Exception as e:
                    logger.error(
                        f"AnomaVision postprocessing failed for batch {batch_idx}: {e}"
                    )
                    continue

            # AnomaVision visualization phase
            if config.enable_visualization:
                with anomavision_profilers["visualization"]:
                    try:
                        test_images = np.array(images)

                        # Convert classifications to numpy for AnomaVision visualization
                        score_map_classifications_np = (
                            score_map_classifications.numpy()
                            if hasattr(score_map_classifications, "numpy")
                            else score_map_classifications
                        )
                        image_classifications_np = (
                            image_classifications.numpy()
                            if hasattr(image_classifications, "numpy")
                            else image_classifications
                        )
                        score_maps_np = (
                            score_maps
                            if isinstance(score_maps, np.ndarray)
                            else score_maps.numpy()
                        )

                        # Generate AnomaVision visualization outputs
                        boundary_images = (
                            anomavision.visualization.framed_boundary_images(
                                test_images,
                                score_map_classifications_np,
                                image_classifications_np,
                                padding=config.get("viz_padding", 40),
                            )
                        )
                        heatmap_images = anomavision.visualization.heatmap_images(
                            test_images,
                            score_maps_np,
                            alpha=config.get("viz_alpha", 0.5),
                        )
                        highlighted_images = (
                            anomavision.visualization.highlighted_images(
                                [images[i] for i in range(len(images))],
                                score_map_classifications_np,
                                color=viz_color,
                            )
                        )

                        # Display AnomaVision results
                        for img_id in range(len(images)):
                            try:
                                fig, axs = plt.subplots(1, 4, figsize=(16, 8))
                                fig.suptitle(
                                    f"AnomaVision Detection Results - Batch {img_id + 1}",
                                    fontsize=14,
                                )

                                axs[0].imshow(images[img_id])
                                axs[0].set_title("Original Image")
                                axs[0].axis("off")

                                axs[1].imshow(boundary_images[img_id])
                                axs[1].set_title("AnomaVision Boundary Detection")
                                axs[1].axis("off")

                                axs[2].imshow(heatmap_images[img_id])
                                axs[2].set_title("AnomaVision Anomaly Heatmap")
                                axs[2].axis("off")

                                axs[3].imshow(highlighted_images[img_id])
                                axs[3].set_title("AnomaVision Highlighted Anomalies")
                                axs[3].axis("off")

                                if config.save_visualizations and RESULTS_PATH:
                                    timestamp = img_id  # datetime.now().strftime("%Y%m%d_%H%M%S")
                                    combined_filepath = os.path.join(
                                        RESULTS_PATH,
                                        f"anomavision_batch_{batch_idx}_{timestamp}.png",
                                    )
                                    plt.savefig(
                                        combined_filepath, dpi=100, bbox_inches="tight"
                                    )
                                    logger.info(
                                        f"AnomaVision visualization saved: {combined_filepath}"
                                    )

                                plt.close(fig)

                            except Exception as e:
                                logger.warning(
                                    f"Failed to display AnomaVision visualization for batch {batch_idx}: {e}"
                                )

                    except Exception as e:
                        logger.error(
                            f"AnomaVision visualization failed for batch {batch_idx}: {e}"
                        )

    finally:
        logger.info("Closing AnomaVision model and freeing resources")
        model.close()

    # Calculate total AnomaVision pipeline time
    total_pipeline_time = time.time() - total_start_time

    # Log AnomaVision timing summary
    logger.info("=" * 60)
    logger.info("ANOMAVISION PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Setup time:                {anomavision_profilers['setup'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Model loading time:        {anomavision_profilers['model_loading'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Data loading time:         {anomavision_profilers['data_loading'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Inference time:            {anomavision_profilers['inference'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Postprocessing time:       {anomavision_profilers['postprocessing'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Visualization time:        {anomavision_profilers['visualization'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(f"Total pipeline time:       {total_pipeline_time * 1000:.2f} ms")
    logger.info("=" * 60)

    # AnomaVision performance metrics (focusing on meaningful metrics)
    total_images = len(test_dataset)
    inference_fps = anomavision_profilers["inference"].get_fps(total_images)
    avg_inference_time = anomavision_profilers["inference"].get_avg_time_ms(batch_count)

    logger.info("=" * 60)
    logger.info("ANOMAVISION INFERENCE PERFORMANCE")
    logger.info("=" * 60)
    if inference_fps > 0:
        logger.info(f"Pure inference FPS:        {inference_fps:.2f} images/sec")

    if avg_inference_time > 0:
        logger.info(f"Average inference time:    {avg_inference_time:.2f} ms/batch")

    # Additional useful metrics
    if batch_count > 0:
        images_per_batch = total_images / batch_count
        logger.info(
            f"Throughput:                {inference_fps * images_per_batch:.1f} images/sec (batch size: {config.batch_size})"
        )

    logger.info("=" * 60)
    logger.info(
        "AnomaVision anomaly detection inference process completed successfully"
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        logger = get_logger(__name__)
        logger.info("Process interrupted by user")
        exit(1)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Process failed with error: {e}", exc_info=True)
        exit(1)
