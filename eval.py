import os
import anodet
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime

# Updated imports to use the inference modules (same as detect.py)
from anodet.inference.model.wrapper import ModelWrapper
from anodet.inference.modelType import ModelType
from anodet.utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detection model performance using trained models."
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_path",
        default=r"D:\01-DATA",
        type=str,
        required=False,
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
        default="./distributions/",
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
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run evaluation on (auto will choose cuda if available)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for evaluation"
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
        "--memory_efficient",
        action="store_true",
        default=True,
        help="Use memory efficient evaluation.",
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
        default="./eval_visualizations/",
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

    return parser.parse_args()


def determine_device(device_arg):
    """Determine the best device to use for evaluation"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device_arg


def evaluate_model_with_wrapper(
    model_wrapper, test_dataloader, logger, detailed_timing=False
):
    """
    Evaluate model using the ModelWrapper inference interface
    Returns: (images, image_classifications_target, masks_target, image_scores, score_maps)
    """
    all_images = []
    all_image_classifications_target = []
    all_masks_target = []
    all_image_scores = []
    all_score_maps = []

    total_inference_time = 0
    batch_count = 0
    device_str = determine_device("cpu")
    logger.info(f"Starting evaluation on {len(test_dataloader.dataset)} images")
    try:
        for batch_idx, (batch, images, image_targets, mask_targets) in enumerate(
            test_dataloader
        ):
            batch = batch.to(device_str)
            batch_start = time.time()
            logger.debug(
                f"Processing evaluation batch {batch_idx + 1}/{len(test_dataloader)}"
            )

            # Inference using the ModelWrapper
            inference_start = time.time()
            try:
                # The ModelWrapper.predict() returns (scores, maps) as numpy arrays
                image_scores, score_maps = model_wrapper.predict(batch)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                if detailed_timing:
                    logger.debug(
                        f"Batch {batch_idx}: Inference completed in {inference_time:.4f}s"
                    )

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

            # Handle different return types from ModelWrapper
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
            batch_time = time.time() - batch_start
            if detailed_timing:
                logger.debug(f"Batch {batch_idx + 1} completed in {batch_time:.4f}s")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    # Convert lists back to appropriate formats
    all_images = np.array(all_images)
    all_image_classifications_target = np.array(all_image_classifications_target)
    all_masks_target = np.array(all_masks_target)
    all_image_scores = np.array(all_image_scores)
    all_score_maps = np.array(all_score_maps)

    # Log performance statistics
    if total_inference_time > 0:
        fps = len(test_dataloader.dataset) / total_inference_time
        logger.info(f"Evaluation inference FPS: {fps:.2f}")
        logger.info(f"Total inference time: {total_inference_time:.4f}s")

    logger.info("Evaluation completed successfully")

    return (
        all_images,
        all_image_classifications_target,
        all_masks_target,
        all_image_scores,
        all_score_maps,
    )


def main(args):
    # Setup logging first
    setup_logging(args.log_level)
    logger = get_logger(__name__)

    # Setup timing
    total_start_time = time.time()
    timing_stats = {
        "setup": 0,
        "model_loading": 0,
        "data_loading": 0,
        "evaluation": 0,
        "visualization": 0,
        "total": 0,
    }

    logger.info("Starting anomaly detection model evaluation")
    logger.info(f"Arguments: {vars(args)}")

    # Setup
    setup_start = time.time()
    DATASET_PATH = os.path.realpath(args.dataset_path)
    MODEL_DATA_PATH = os.path.realpath(args.model_data_path)

    # Determine device
    device_str = determine_device(args.device)
    logger.info(f"Selected device: {device_str}")

    logger.info(f"Dataset path: {DATASET_PATH}")
    logger.info(f"Model data path: {MODEL_DATA_PATH}")
    logger.info(f"Class name: {args.class_name}")

    if device_str == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA available, enabled cuDNN benchmark")
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    timing_stats["setup"] = time.time() - setup_start

    # Load model using the inference wrapper
    model_load_start = time.time()
    model_path = os.path.join(MODEL_DATA_PATH, args.model)
    logger.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Use the inference ModelWrapper
        model = ModelWrapper(model_path, device_str)
        model_type = ModelType.from_extension(model_path)
        logger.info(f"Model loaded successfully. Type: {model_type.value.upper()}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    timing_stats["model_loading"] = time.time() - model_load_start

    # Create test dataset
    data_load_start = time.time()
    logger.info("Creating test dataset and dataloader")

    try:
        # Use MVTecDataset for evaluation (as in original eval.py)
        test_dataset = anodet.MVTecDataset(
            DATASET_PATH, args.class_name, is_train=False
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and device_str == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        logger.info(
            f"Test dataset created successfully. Total images: {len(test_dataset)}"
        )
        logger.info(
            f"Batch size: {args.batch_size}, Number of batches: {len(test_dataloader)}"
        )
    except Exception as e:
        logger.error(f"Failed to create test dataset/dataloader: {e}")
        raise

    timing_stats["data_loading"] = time.time() - data_load_start

    # Create output directory for visualizations if needed
    if args.save_visualizations:
        os.makedirs(args.viz_output_dir, exist_ok=True)
        logger.info(f"Evaluation visualization output directory: {args.viz_output_dir}")

    # Run evaluation
    eval_start = time.time()
    logger.info(
        f"Starting evaluation of {len(test_dataset)} images using {model_type.value.upper()}"
    )

    try:
        # Run evaluation using ModelWrapper
        images, image_classifications_target, masks_target, image_scores, score_maps = (
            evaluate_model_with_wrapper(
                model, test_dataloader, logger, args.detailed_timing
            )
        )

        eval_time = time.time() - eval_start
        timing_stats["evaluation"] = eval_time

        logger.info(f"Evaluation completed in {eval_time:.4f}s")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        # Always close the model to free resources
        logger.info("Closing model and freeing resources")
        model.close()

    # Visualization
    if args.enable_visualization:
        viz_start = time.time()
        logger.info("Generating evaluation visualizations")

        try:
            # Use the anodet visualization function (as in original eval.py)
            anodet.visualize_eval_data(
                image_classifications_target, masks_target, image_scores, score_maps
            )

            # Save visualizations if requested
            if args.save_visualizations:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save the current figure
                viz_filepath = os.path.join(
                    args.viz_output_dir,
                    f"evaluation_results_{args.class_name}_{timestamp}.png",
                )
                plt.savefig(viz_filepath, dpi=300, bbox_inches="tight")
                logger.info(f"Evaluation visualization saved: {viz_filepath}")

            plt.show()

        except Exception as e:
            logger.error(f"Visualization failed: {e}")

        timing_stats["visualization"] = time.time() - viz_start

    # Calculate final timing statistics
    total_time = time.time() - total_start_time
    timing_stats["total"] = total_time

    # Log timing summary
    logger.info("=" * 50)
    logger.info("EVALUATION TIMING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Setup time:           {timing_stats['setup']:.4f}s")
    logger.info(f"Model loading time:   {timing_stats['model_loading']:.4f}s")
    logger.info(f"Data loading time:    {timing_stats['data_loading']:.4f}s")
    logger.info(f"Evaluation time:      {timing_stats['evaluation']:.4f}s")
    logger.info(f"Visualization time:   {timing_stats['visualization']:.4f}s")
    logger.info(f"Total time:           {timing_stats['total']:.4f}s")
    logger.info("=" * 50)

    # Performance metrics
    total_images = len(test_dataset)
    if timing_stats["evaluation"] > 0:
        fps = total_images / timing_stats["evaluation"]
        logger.info(f"Average evaluation FPS: {fps:.2f}")

    if timing_stats["total"] > 0:
        total_fps = total_images / timing_stats["total"]
        logger.info(f"Overall processing FPS: {total_fps:.2f}")

    # Log evaluation summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Dataset: {args.class_name}")
    logger.info(f"Total images evaluated: {total_images}")
    logger.info(f"Model type: {model_type.value.upper()}")
    logger.info(f"Device: {device_str}")
    logger.info("=" * 50)

    logger.info("Anomaly detection model evaluation completed successfully")


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        logger = get_logger(__name__)
        logger.info("Evaluation process interrupted by user")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Evaluation process failed with error: {e}", exc_info=True)
