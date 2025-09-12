import argparse
import contextlib
import os
import time
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

# Updated imports to use the inference modules (same as detect.py)
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType
from anomavision.utils import (
    adaptive_gaussian_blur,
    get_logger,
    merge_config,
    setup_logging,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AnomaVision anomaly detection model performance using trained models."
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yml/.json"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_path",
        default=r"D:\01-DATA",
        type=str,
        required=False,
        help="Path to the dataset folder containing test images for AnomaVision evaluation.",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="bottle",
        help="Class name for MVTec dataset evaluation with AnomaVision.",
    )

    # Model parameters
    parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions/anomav_exp",
        help="Directory containing AnomaVision model files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="padim_model.onnx",
        help="AnomaVision model file (.pt for PyTorch, .onnx for ONNX, .engine for TensorRT)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device to run AnomaVision evaluation on (auto will choose cuda if available)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for AnomaVision evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for data loading in AnomaVision evaluation.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Use pinned memory for faster GPU transfers during AnomaVision evaluation.",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        default=True,
        help="Use memory efficient evaluation for AnomaVision.",
    )

    # Visualization parameters
    parser.add_argument(
        "--enable_visualization",
        action="store_false",
        help="Enable visualization of AnomaVision evaluation results.",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save AnomaVision evaluation visualization images to disk.",
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default="./eval_visualizations/",
        help="Directory to save AnomaVision visualization images.",
    )

    # Logging parameters
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for AnomaVision evaluation.",
    )
    parser.add_argument(
        "--detailed_timing",
        action="store_true",
        help="Enable detailed timing measurements for AnomaVision evaluation.",
    )

    return parser.parse_args()


def evaluate_model_with_wrapper(
    model_wrapper, test_dataloader, logger, evaluation_profiler, detailed_timing=False
):
    """
    Evaluate AnomaVision model using the ModelWrapper inference interface
    Returns: (images, image_classifications_target, masks_target, image_scores, score_maps)
    """
    all_images = []
    all_image_classifications_target = []
    all_masks_target = []
    all_image_scores = []
    all_score_maps = []

    batch_count = 0
    device_str = determine_device("cpu")
    logger.info(
        f"Starting AnomaVision evaluation on {len(test_dataloader.dataset)} images"
    )

    try:
        for batch_idx, (batch, images, image_targets, mask_targets) in enumerate(
            test_dataloader
        ):
            batch = batch.to(device_str)
            logger.debug(
                f"Processing AnomaVision evaluation batch {batch_idx + 1}/{len(test_dataloader)}"
            )

            # AnomaVision inference timing
            with evaluation_profiler as eval_prof:
                try:
                    # The ModelWrapper.predict() returns (scores, maps) as numpy arrays
                    image_scores, score_maps = model_wrapper.predict(batch)

                    if detailed_timing:
                        logger.debug(
                            f"AnomaVision batch {batch_idx}: Inference completed in {eval_prof.elapsed_time * 1000:.2f} ms"
                        )

                except Exception as e:
                    logger.error(
                        f"AnomaVision inference failed for batch {batch_idx}: {e}"
                    )
                    continue

            # Collect AnomaVision evaluation results
            all_images.extend(images)
            all_image_classifications_target.extend(
                image_targets.numpy()
                if hasattr(image_targets, "numpy")
                else image_targets
            )
            all_masks_target.extend(
                mask_targets.numpy() if hasattr(mask_targets, "numpy") else mask_targets
            )

            # Handle different return types from AnomaVision ModelWrapper
            if isinstance(image_scores, np.ndarray):
                all_image_scores.extend(image_scores.tolist())
                all_score_maps.extend(score_maps)
            else:
                all_image_scores.extend(
                    image_scores.cpu().numpy().tolist()
                    if hasattr(image_scores, "cpu")
                    else image_scores.tolist()
                )
                all_score_maps.extend(
                    score_maps.cpu().numpy()
                    if hasattr(score_maps, "cpu")
                    else score_maps
                )

            batch_count += 1

    except Exception as e:
        logger.error(f"AnomaVision evaluation failed: {e}")
        raise

    # Convert lists back to appropriate formats for AnomaVision analysis
    all_images = np.array(all_images)
    all_image_classifications_target = np.array(all_image_classifications_target)
    all_masks_target = np.squeeze(np.array(all_masks_target), axis=1)
    all_image_scores = np.array(all_image_scores)
    all_score_maps = np.array(all_score_maps)

    # Log AnomaVision evaluation performance statistics
    evaluation_fps = evaluation_profiler.get_fps(len(test_dataloader.dataset))
    if evaluation_fps > 0:
        logger.info(f"AnomaVision evaluation inference FPS: {evaluation_fps:.2f}")
        logger.info(
            f"AnomaVision total evaluation time: {evaluation_profiler.accumulated_time:.4f}s"
        )

    logger.info("AnomaVision evaluation completed successfully")

    return (
        all_images,
        all_image_classifications_target,
        all_masks_target,
        all_image_scores,
        all_score_maps,
    )


def main(args):
    # Setup logging first

    if args.config is not None:
        cfg = load_config(str(args.config))
    else:
        cfg = load_config(str(Path(args.model_data_path) / "config.yml"))

    # Merge config with CLI args
    config = edict(merge_config(args, cfg))

    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger(__name__)

    # Log image processing configuration
    logger.info(
        "Image processing config: resize=%s, crop_size=%s, normalize=%s",
        config.resize,
        config.crop_size,
        config.normalize,
    )
    if config.normalize:
        logger.info("Normalization: mean=%s, std=%s", config.norm_mean, config.norm_std)

    # Initialize AnomaVision profilers for different evaluation phases
    anomavision_profilers = {
        "setup": Profiler(),
        "model_loading": Profiler(),
        "data_loading": Profiler(),
        "evaluation": Profiler(),  # Core evaluation timing
        "visualization": Profiler(),  # Evaluation visualization timing
    }

    logger.info("Starting AnomaVision anomaly detection model evaluation")
    logger.info(f"Arguments: {vars(args)}")

    # AnomaVision evaluation setup phase
    with anomavision_profilers["setup"]:
        DATASET_PATH = os.path.realpath(config.dataset_path)
        MODEL_DATA_PATH = os.path.realpath(config.model_data_path)
        device_str = determine_device(config.device)

        logger.info(f"AnomaVision selected device: {device_str}")
        logger.info(f"AnomaVision dataset path: {DATASET_PATH}")
        logger.info(f"AnomaVision model data path: {MODEL_DATA_PATH}")
        logger.info(f"AnomaVision class name: {config.class_name}")

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
            # Use the inference ModelWrapper for AnomaVision
            model = ModelWrapper(model_path, device_str)
            model_type = ModelType.from_extension(model_path)
            logger.info(
                f"AnomaVision model loaded successfully. Type: {model_type.value.upper()}"
            )
        except Exception as e:
            logger.error(f"Failed to load AnomaVision model: {e}")
            raise

    # AnomaVision test dataset creation
    with anomavision_profilers["data_loading"]:
        logger.info("Creating AnomaVision test dataset and dataloader")

        try:
            # Use MVTecDataset for AnomaVision evaluation with configurable image processing
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
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory and device_str == "cuda",
                persistent_workers=config.num_workers > 0,
            )
            logger.info(
                f"AnomaVision test dataset created successfully. Total images: {len(test_dataset)}"
            )
            logger.info(
                f"AnomaVision batch size: {config.batch_size}, Number of batches: {len(test_dataloader)}"
            )
        except Exception as e:
            logger.error(f"Failed to create AnomaVision test dataset/dataloader: {e}")
            raise

    # Create output directory for AnomaVision visualizations if needed
    if config.save_visualizations:
        os.makedirs(config.viz_output_dir, exist_ok=True)
        logger.info(
            f"AnomaVision evaluation visualization output directory: {config.viz_output_dir}"
        )

    # Run AnomaVision evaluation
    logger.info(
        f"Starting evaluation of {len(test_dataset)} images using AnomaVision {model_type.value.upper()}"
    )

    try:
        # Run AnomaVision evaluation using ModelWrapper
        images, image_classifications_target, masks_target, image_scores, score_maps = (
            evaluate_model_with_wrapper(
                model,
                test_dataloader,
                logger,
                anomavision_profilers["evaluation"],
                config.detailed_timing,
            )
        )
        score_maps = adaptive_gaussian_blur(score_maps, kernel_size=33, sigma=4)

        logger.info(
            f"AnomaVision evaluation completed in {anomavision_profilers['evaluation'].accumulated_time:.4f}s"
        )

    except Exception as e:
        logger.error(f"AnomaVision evaluation failed: {e}")
        raise
    finally:
        # Always close the AnomaVision model to free resources
        logger.info("Closing AnomaVision model and freeing resources")
        model.close()

    # AnomaVision evaluation visualization
    if config.enable_visualization:
        with anomavision_profilers["visualization"]:
            logger.info("Generating AnomaVision evaluation visualizations")

            try:
                # Use the anodet visualization function for AnomaVision results
                anomavision.visualize_eval_data(
                    image_classifications_target,
                    masks_target.astype(np.uint8).flatten(),
                    image_scores,
                    score_maps.flatten(),
                )

                # Save AnomaVision visualizations if requested
                if config.save_visualizations:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Save the current AnomaVision evaluation figure
                    viz_filepath = os.path.join(
                        config.viz_output_dir,
                        f"anomavision_evaluation_{config.class_name}_{timestamp}.png",
                    )
                    plt.savefig(viz_filepath, dpi=300, bbox_inches="tight")
                    logger.info(
                        f"AnomaVision evaluation visualization saved: {viz_filepath}"
                    )

                plt.show()

            except Exception as e:
                logger.error(f"AnomaVision visualization failed: {e}")

    # Log AnomaVision evaluation timing summary
    logger.info("=" * 60)
    logger.info("ANOMAVISION EVALUATION PERFORMANCE SUMMARY")
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
        f"Evaluation time:           {anomavision_profilers['evaluation'].accumulated_time * 1000:.2f} ms"
    )
    logger.info(
        f"Visualization time:        {anomavision_profilers['visualization'].accumulated_time * 1000:.2f} ms"
    )
    logger.info("=" * 60)

    # AnomaVision evaluation performance metrics (focusing on meaningful metrics)
    total_images = len(test_dataset)
    evaluation_fps = anomavision_profilers["evaluation"].get_fps(total_images)
    avg_evaluation_time = anomavision_profilers["evaluation"].get_avg_time_ms(
        len(test_dataloader)
    )

    logger.info("=" * 60)
    logger.info("ANOMAVISION EVALUATION PERFORMANCE")
    logger.info("=" * 60)
    if evaluation_fps > 0:
        logger.info(f"Pure evaluation FPS:       {evaluation_fps:.2f} images/sec")

    if avg_evaluation_time > 0:
        logger.info(f"Average evaluation time:   {avg_evaluation_time:.2f} ms/batch")

    # Additional useful AnomaVision metrics
    if len(test_dataloader) > 0:
        images_per_batch = total_images / len(test_dataloader)
        logger.info(
            f"Evaluation throughput:     {evaluation_fps * images_per_batch:.1f} images/sec (batch size: {config.batch_size})"
        )

    logger.info("=" * 60)

    # Log AnomaVision evaluation summary
    logger.info("=" * 60)
    logger.info("ANOMAVISION EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.class_name}")
    logger.info(f"Total images evaluated: {total_images}")
    logger.info(f"Model type: {model_type.value.upper()}")
    logger.info(f"Device: {device_str}")
    logger.info(
        f"Image processing: resize={config.resize}, crop_size={config.crop_size}, normalize={config.normalize}"
    )
    logger.info("=" * 60)

    logger.info("AnomaVision anomaly detection model evaluation completed successfully")


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        logger = get_logger(__name__)
        logger.info("AnomaVision evaluation process interrupted by user")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(
            f"AnomaVision evaluation process failed with error: {e}", exc_info=True
        )
