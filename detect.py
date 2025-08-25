"""
Run Anomaly detection inference on images using various model formats.

Usage - formats:
    $ python detect.py --model padim_model.pt                  # PyTorch
                                   padim_model.torchscript        # TorchScript
                                   padim_model.onnx               # ONNX Runtime
                                   padim_model_openvino           # OpenVINO
                                   padim_model.engine             # TensorRT
"""

import os
import anodet
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime

# Updated imports to use the inference modules
from anodet.inference.model.wrapper import ModelWrapper
from anodet.inference.modelType import ModelType
from anodet.utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection inference using trained models."
    )
    parser.add_argument(
        "--dataset_path",
        default=r"D:\01-DATA\dum\c2",
        type=str,
        required=False,
        help="Path to the dataset folder containing test images.",
    )
    parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions/",
        help="Directory containing model files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="padim_model_openvino",
        help="Model file (.pt for PyTorch, .onnx for ONNX, .engine for TensorRT)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on (auto will choose cuda if available)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=13.0,
        help="Threshold for anomaly classification",
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

    # Visualization parameters
    parser.add_argument(
        "--enable_visualization",
        action="store_false",
        help="Enable visualization of results.",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualization images to disk.",
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default="./visualizations/",
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--show_first_batch_only",
        action="store_true",
        default=True,
        help="Show visualization only for the first batch.",
    )
    parser.add_argument(
        "--viz_alpha", type=float, default=0.5, help="Alpha value for heatmap overlay."
    )
    parser.add_argument(
        "--viz_padding",
        type=int,
        default=40,
        help="Padding for boundary visualization.",
    )
    parser.add_argument(
        "--viz_color",
        type=str,
        default="128,0,128",
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


def save_visualization(images, filename, output_dir):
    """Save visualization images to disk"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    if len(images.shape) == 4:  # Batch of images
        for i, img in enumerate(images):
            individual_filepath = os.path.join(
                output_dir, f"{filename.split('.')[0]}_batch_{i}.png"
            )
            plt.imsave(individual_filepath, img)
    else:  # Single image
        plt.imsave(filepath, images)


def determine_device(device_arg):
    """Determine the best device to use for inference"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device_arg


def main(args):
    # Setup logging first
    setup_logging(args.log_level)
    logger = get_logger(__name__)

    # Parse visualization color
    try:
        viz_color = tuple(map(int, args.viz_color.split(",")))
        if len(viz_color) != 3:
            raise ValueError
    except ValueError:
        logger.warning(
            f"Invalid color format '{args.viz_color}'. Using default (128,0,128)"
        )
        viz_color = (128, 0, 128)

    # Setup timing
    total_start_time = time.time()
    timing_stats = {
        "setup": 0,
        "model_loading": 0,
        "data_loading": 0,
        "preprocessing": 0,
        "inference": 0,
        "postprocessing": 0,
        "visualization": 0,
        "total": 0,
    }

    logger.info("Starting anomaly detection inference process")
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

    # Create output directory for visualizations if needed
    if args.save_visualizations:
        os.makedirs(args.viz_output_dir, exist_ok=True)
        logger.info(f"Visualization output directory: {args.viz_output_dir}")

    # DataLoader
    data_load_start = time.time()
    logger.info("Creating dataset and dataloader")

    try:
        test_dataset = anodet.AnodetDataset(DATASET_PATH)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and device_str == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        logger.info(f"Dataset created successfully. Total images: {len(test_dataset)}")
        logger.info(
            f"Batch size: {args.batch_size}, Number of batches: {len(test_dataloader)}"
        )
    except Exception as e:
        logger.error(f"Failed to create dataset/dataloader: {e}")
        raise

    timing_stats["data_loading"] = time.time() - data_load_start

    logger.info(
        f"Processing {len(test_dataset)} images using {model_type.value.upper()}"
    )

    # Process batches
    total_inference_time = 0
    total_preprocessing_time = 0
    total_postprocessing_time = 0
    total_visualization_time = 0

    try:
        for batch_idx, (batch, images, _, _) in enumerate(test_dataloader):
            batch_start_time = time.time()
            batch = batch.to(device_str)
            logger.debug(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")

            # Log batch info
            if isinstance(batch, torch.Tensor):
                logger.debug(f"Batch shape: {batch.shape}, dtype: {batch.dtype}")
            else:
                logger.debug(f"Batch type: {type(batch)}")

            # Inference using the inference wrapper
            inference_start = time.time()
            try:
                # The ModelWrapper.predict() returns (scores, maps) as numpy arrays
                image_scores, score_maps = model.predict(batch)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                logger.debug(
                    f"Batch {batch_idx}: Inference completed in {inference_time:.4f}s"
                )
                logger.debug(
                    f"Image scores shape: {image_scores.shape}, Score maps shape: {score_maps.shape}"
                )

            except Exception as e:
                logger.error(f"Inference failed for batch {batch_idx}: {e}")
                continue

            # Postprocessing
            postprocess_start = time.time()
            try:
                # # Convert to torch tensors for classification if needed
                # if isinstance(image_scores, np.ndarray):
                #     image_scores_tensor = torch.from_numpy(image_scores)
                #     score_maps_tensor = torch.from_numpy(score_maps)
                # else:
                #     image_scores_tensor = image_scores
                #     score_maps_tensor = score_maps

                # Apply threshold classification
                score_map_classifications = anodet.classification(
                    score_maps, args.thresh
                )
                image_classifications = anodet.classification(image_scores, args.thresh)

                postprocess_time = time.time() - postprocess_start
                total_postprocessing_time += postprocess_time

                # Convert back to numpy for logging if needed
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

                logger.info(
                    f"Batch {batch_idx + 1}: Scores: {image_scores_list}, Classifications: {image_classifications_list}"
                )
                logger.debug(f"Postprocessing completed in {postprocess_time:.4f}s")

            except Exception as e:
                logger.error(f"Postprocessing failed for batch {batch_idx}: {e}")
                continue

            # Visualization
            if args.enable_visualization:
                viz_start = time.time()
                try:
                    test_images = np.array(images)

                    # Convert classifications to numpy if needed for visualization
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

                    boundary_images = anodet.visualization.framed_boundary_images(
                        test_images,
                        score_map_classifications_np,
                        image_classifications_np,
                        padding=args.viz_padding,
                    )
                    heatmap_images = anodet.visualization.heatmap_images(
                        test_images, score_maps_np, alpha=args.viz_alpha
                    )
                    highlighted_images = anodet.visualization.highlighted_images(
                        [images[i] for i in range(len(images))],
                        score_map_classifications_np,
                        color=viz_color,
                    )

                    viz_time = time.time() - viz_start
                    total_visualization_time += viz_time
                    logger.debug(
                        f"Visualization processing completed in {viz_time:.4f}s"
                    )

                    # Save visualizations if requested
                    if args.save_visualizations:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # save_visualization(boundary_images, f"boundary_batch_{batch_idx}_{timestamp}.png", args.viz_output_dir)
                        # save_visualization(heatmap_images, f"heatmap_batch_{batch_idx}_{timestamp}.png", args.viz_output_dir)
                        # save_visualization(highlighted_images, f"highlighted_batch_{batch_idx}_{timestamp}.png", args.viz_output_dir)
                        # logger.debug(f"Visualizations saved for batch {batch_idx}")

                    # Show first batch only (if requested)
                    if batch_idx == 0 and (
                        not args.show_first_batch_only or batch_idx == 0
                    ):
                        try:
                            fig, axs = plt.subplots(1, 4, figsize=(16, 8))
                            fig.suptitle(
                                f"Anomaly Detection Results - Batch {batch_idx + 1}",
                                fontsize=14,
                            )

                            axs[0].imshow(images[0])
                            axs[0].set_title("Original Image")
                            axs[0].axis("off")

                            axs[1].imshow(boundary_images[0])
                            axs[1].set_title("Boundary Detection")
                            axs[1].axis("off")

                            axs[2].imshow(heatmap_images[0])
                            axs[2].set_title("Anomaly Heatmap")
                            axs[2].axis("off")

                            axs[3].imshow(highlighted_images[0])
                            axs[3].set_title("Highlighted Anomalies")
                            axs[3].axis("off")

                            plt.tight_layout()

                            if args.save_visualizations:
                                combined_filepath = os.path.join(
                                    args.viz_output_dir,
                                    f"combined_batch_{batch_idx}_{timestamp}.png",
                                )
                                plt.savefig(
                                    combined_filepath, dpi=300, bbox_inches="tight"
                                )
                                logger.info(
                                    f"Combined visualization saved: {combined_filepath}"
                                )

                            plt.show()

                        except Exception as e:
                            logger.warning(
                                f"Failed to display visualization for batch {batch_idx}: {e}"
                            )

                except Exception as e:
                    logger.error(f"Visualization failed for batch {batch_idx}: {e}")

            batch_time = time.time() - batch_start_time
            logger.debug(f"Batch {batch_idx + 1} completed in {batch_time:.4f}s")

    finally:
        # Always close the model to free resources
        logger.info("Closing model and freeing resources")
        model.close()

    # Calculate final timing statistics
    total_time = time.time() - total_start_time
    timing_stats["preprocessing"] = total_preprocessing_time
    timing_stats["inference"] = total_inference_time
    timing_stats["postprocessing"] = total_postprocessing_time
    timing_stats["visualization"] = total_visualization_time
    timing_stats["total"] = total_time

    # Log timing summary
    logger.info("=" * 50)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Setup time:           {timing_stats['setup']:.4f}s")
    logger.info(f"Model loading time:   {timing_stats['model_loading']:.4f}s")
    logger.info(f"Data loading time:    {timing_stats['data_loading']:.4f}s")
    logger.info(f"Preprocessing time:   {timing_stats['preprocessing']:.4f}s")
    logger.info(f"Inference time:       {timing_stats['inference']:.4f}s")
    logger.info(f"Postprocessing time:  {timing_stats['postprocessing']:.4f}s")
    logger.info(f"Visualization time:   {timing_stats['visualization']:.4f}s")
    logger.info(f"Total time:           {timing_stats['total']:.4f}s")
    logger.info("=" * 50)

    # Performance metrics
    total_images = len(test_dataset)
    if timing_stats["inference"] > 0:
        fps = total_images / timing_stats["inference"]
        logger.info(f"Average inference FPS: {fps:.2f}")

    if timing_stats["total"] > 0:
        total_fps = total_images / timing_stats["total"]
        logger.info(f"Overall processing FPS: {total_fps:.2f}")

    logger.info("Anomaly detection inference process completed successfully")


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        logger = get_logger(__name__)
        logger.info("Process interrupted by user")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Process failed with error: {e}", exc_info=True)
