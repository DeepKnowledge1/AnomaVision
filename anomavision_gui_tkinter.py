"""
@file anomavision_gui_tkinter.py
AnomaVision Tkinter GUI implementation for anomaly detection
"""
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import traceback
from datetime import datetime
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
# Add time import for performance analysis
import time
import anomavision
from anomavision.padim import Padim
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType
from anomavision.utils import (
    create_image_transform, get_logger, merge_config, setup_logging,
    adaptive_gaussian_blur
)
from anomavision.config import load_config
from anomavision.general import Profiler, determine_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingWorker(threading.Thread):
    """Worker thread for model training"""
    def __init__(self, config, progress_callback, finished_callback, error_callback):
        super().__init__()
        self.config = config
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.error_callback = error_callback
        # Get dedicated training logger, consistent with CLI
        self.logger = get_logger("anomavision.train")
        
    def run(self):
        try:
            self.progress_callback("Starting model training...")
            self.logger.info("Starting AnomaVision model training process")
            self.logger.info(f"Training config: {self.config}")
            # Create dataset
            self.progress_callback("Loading training data...")
            self.logger.info("Creating AnomaVision training dataset")
            root = os.path.join(
                os.path.realpath(self.config['dataset_path']), 
                self.config['class_name'], 
                "train", 
                "good"
            )
            self.logger.info(f"Training data path: {root}")
            if not os.path.isdir(root):
                raise FileNotFoundError(f'Training data directory does not exist: {root}')
            ds = anomavision.AnodetDataset(
                root,
                resize=self.config['resize'],
                crop_size=self.config['crop_size'],
                normalize=self.config['normalize'],
                mean=self.config['norm_mean'],
                std=self.config['norm_std'],
            )
            if len(ds) == 0:
                raise ValueError(f"No images found in training directory: {root}")
            dl = DataLoader(ds, batch_size=int(self.config['batch_size']), shuffle=False)
            self.progress_callback(f"Dataset loaded: {len(ds)} images")
            self.logger.info(f"dataset: {len(ds)} images | batch_size={self.config['batch_size']}")
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.progress_callback(f"Using device: {device.type}")
            self.logger.info(f"device: {device.type} (cuda_available={torch.cuda.is_available()})")
            if device.type == "cuda":
                self.logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
                self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Create and train model
            self.progress_callback("Initializing model...")
            self.logger.info("Initializing AnomaVision PaDiM model")
            # Ensure layer_indices is a list of integers
            layer_indices = self.config['layer_indices']
            if isinstance(layer_indices, str):
                layer_indices = eval(layer_indices)
            self.logger.info(f"Model config: backbone={self.config['backbone']} | layers={layer_indices} | feat_dim={self.config['feat_dim']}")
            padim = Padim(
                backbone=self.config['backbone'],
                device=device,
                layer_indices=layer_indices,
                feat_dim=int(self.config['feat_dim']),
            )
            self.progress_callback(f"Model config: backbone={self.config['backbone']}, layers={layer_indices}, feat_dim={self.config['feat_dim']}")
            self.progress_callback("Starting training...")
            self.logger.info("Starting model training")
            t_fit = time.perf_counter()
            padim.fit(dl)
            self.logger.info(f"fit: completed in {time.perf_counter() - t_fit:.2f}s")
            self.progress_callback("Training completed!")
            # Save model
            model_path = Path(self.config['model_data_path']) / self.config['output_model']
            torch.save(padim, str(model_path))
            self.progress_callback(f"Model saved to: {model_path}")
            self.logger.info(f"Model saved to: {model_path}")
            # Save statistics
            try:
                stats_path = model_path.with_suffix(".pth")
                padim.save_statistics(str(stats_path), half=True)
                self.progress_callback(f"Statistics saved to: {stats_path}")
                self.logger.info(f"Statistics saved to: {stats_path}")
            except Exception as e:
                self.progress_callback(f"Failed to save statistics: {e}")
                self.logger.warning(f"Failed to save statistics: {e}")
            self.finished_callback(padim)
        except Exception as e:
            error_msg = f"Error occurred during training: {str(e)}\n{traceback.format_exc()}"
            self.error_callback(error_msg)
            self.logger.exception("Fatal error during training")

class InferenceWorker(threading.Thread):
    """Worker thread for model inference"""
    def __init__(self, model_path, img_path, config, progress_callback, finished_callback, error_callback):
        super().__init__()
        self.model_path = model_path
        self.img_path = img_path
        self.config = config
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.error_callback = error_callback
        # Get dedicated inference logger, consistent with CLI
        self.logger = get_logger("anomavision.detect")
        
    def run(self):
        try:
            self.progress_callback("Starting inference...")
            self.logger.info("Starting AnomaVision anomaly detection inference process")
            self.logger.info(f"Inference config: {self.config}")
            # Initialize performance profiler
            profilers = {
                "model_loading": Profiler(),
                "data_loading": Profiler(),
                "warmup": Profiler(),
                "inference": Profiler(),
                "postprocessing": Profiler(),
                "visualization": Profiler(),
            }
            # Model loading phase performance analysis
            with profilers["model_loading"]:
                self.progress_callback("Loading model...")
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
                self.logger.info(f"Selected device: {device_str}")
                self.logger.info(f"Dataset path: {self.img_path}")
                self.logger.info(f"Model path: {self.model_path}")
                if device_str == "cuda" and torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
                    self.logger.info("CUDA available, enabled cuDNN benchmark")
                    self.logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
                    self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                model = ModelWrapper(self.model_path, device_str)
                model_type = ModelType.from_extension(self.model_path)
                self.logger.info(f"Model loaded successfully. Type: {model_type.value.upper()}")
                self.progress_callback(f"Model loaded: {model_type.value.upper()}")
            # Data loading phase performance analysis
            with profilers["data_loading"]:
                self.progress_callback("Loading test data...")
                self.logger.info("Creating AnomaVision dataset and dataloader")
                test_dataset = anomavision.AnodetDataset(
                    self.img_path,
                    resize=self.config['resize'],
                    crop_size=self.config['crop_size'],
                    normalize=self.config['normalize'],
                    mean=self.config['norm_mean'],
                    std=self.config['norm_std'],
                )
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    pin_memory=self.config['pin_memory'],
                    persistent_workers=self.config['num_workers'] > 0,
                )
                self.progress_callback(f"Dataset loaded: {len(test_dataset)} images")
                self.logger.info(f"Dataset created successfully. Total images: {len(test_dataset)}")
                self.logger.info(f"Batch size: {self.config['batch_size']}, Number of batches: {len(test_dataloader)}")
            # Model warmup phase performance analysis
            try:
                with profilers["warmup"]:
                    self.progress_callback("Model warming up...")
                    first = next(iter(test_dataloader))  # (batch, images, _, _)
                    first_batch = first[0]
                    if device_str == "cuda":
                        first_batch = first_batch.half()
                    first_batch = first_batch.to(device_str)
                    model.warmup(batch=first_batch, runs=2)
                    self.logger.info(f"Warm-up done with first batch {tuple(first_batch.shape)}.")
                    self.progress_callback("Model warmup completed")
            except StopIteration:
                self.progress_callback("Dataset empty, skipping warmup")
                self.logger.warning("Dataset empty; skipping warm-up.")
            except Exception as e:
                self.progress_callback(f"Error during warmup: {e}")
                self.logger.warning(f"Warm-up skipped due to error: {e}")
            # Process images
            all_images = []
            all_image_scores = []
            all_score_maps = []
            all_classifications = []
            all_visualizations = []
            self.logger.info(f"Processing {len(test_dataset)} images using {model_type.value.upper()}")
            try:
                batch_count = 0
                for batch_idx, (batch, images, _, _) in enumerate(test_dataloader):
                    batch_count += 1
                    self.progress_callback(f"Processing batch {batch_idx+1}/{len(test_dataloader)}...")
                    self.logger.debug(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")
                    # Inference phase performance analysis
                    with profilers["inference"] as inference_prof:
                        # Run inference
                        if device_str == "cuda":
                            batch = batch.half()
                        batch = batch.to(device_str)
                        image_scores, score_maps = model.predict(batch)
                        self.logger.debug(f"Image scores shape: {image_scores.shape}, Score maps shape: {score_maps.shape}")
                        self.logger.info(f"Batch shape: {batch.shape}, Inference completed in {inference_prof.elapsed_time * 1000:.2f} ms")
                    # Postprocessing phase performance analysis
                    with profilers["postprocessing"]:
                        # Post-process
                        score_maps = self.adaptive_gaussian_blur(
                            score_maps, kernel_size=33, sigma=4
                        )
                        image_classifications = anomavision.classification(
                            image_scores, self.config['thresh']
                        )
                        # Convert for logging
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
                        self.logger.debug(f"Batch {batch_idx + 1}: Scores: {image_scores_list}, Classifications: {image_classifications_list}")
                        # Store results
                        all_images.extend(images)
                        all_image_scores.extend(image_scores.tolist())
                        all_score_maps.extend(score_maps)
                        all_classifications.extend(image_classifications.tolist())
                    # Visualization generation phase performance analysis
                    with profilers["visualization"]:
                        # Generate visualizations for this batch
                        batch_visualizations = self.generate_visualizations(
                            images, score_maps, image_classifications, self.config
                        )
                        all_visualizations.extend(batch_visualizations)
                # Log performance statistics
                total_pipeline_time = sum([
                    profilers["model_loading"].accumulated_time,
                    profilers["data_loading"].accumulated_time,
                    profilers["warmup"].accumulated_time,
                    profilers["inference"].accumulated_time,
                    profilers["postprocessing"].accumulated_time,
                    profilers["visualization"].accumulated_time,
                ])
                self.logger.info("=" * 60)
                self.logger.info("ANOMAVISION PERFORMANCE SUMMARY")
                self.logger.info("=" * 60)
                self.logger.info(f"Setup time:                0.00 ms")
                self.logger.info(f"Model loading time:        {profilers['model_loading'].accumulated_time * 1000:.2f} ms")
                self.logger.info(f"Data loading time:         {profilers['data_loading'].accumulated_time * 1000:.2f} ms")
                self.logger.info(f"Warmup time:               {profilers['warmup'].accumulated_time * 1000:.2f} ms")
                self.logger.info(f"Inference time:            {profilers['inference'].accumulated_time * 1000:.2f} ms")
                self.logger.info(f"Postprocessing time:       {profilers['postprocessing'].accumulated_time * 1000:.2f} ms")
                self.logger.info(f"Visualization time:        {profilers['visualization'].accumulated_time * 1000:.2f} ms")
                self.logger.info(f"Total pipeline time:       {total_pipeline_time * 1000:.2f} ms")
                self.logger.info("=" * 60)
                # Performance metrics
                total_images = len(all_images)
                inference_fps = profilers["inference"].get_fps(total_images)
                avg_inference_time = profilers["inference"].get_avg_time_ms(batch_count)
                self.logger.info("=" * 60)
                self.logger.info("ANOMAVISION INFERENCE PERFORMANCE")
                self.logger.info("=" * 60)
                if inference_fps > 0:
                    self.logger.info(f"Pure inference FPS:        {inference_fps:.2f} images/sec")
                if avg_inference_time > 0:
                    self.logger.info(f"Average inference time:    {avg_inference_time:.2f} ms/batch")
                # Additional useful metrics
                if batch_count > 0:
                    images_per_batch = total_images / batch_count
                    self.logger.info(f"Throughput:                {inference_fps * images_per_batch:.1f} images/sec (batch size: {self.config['batch_size']})")
                self.logger.info("=" * 60)
                self.logger.info("AnomaVision anomaly detection inference process completed successfully")
                self.finished_callback(
                    all_images, all_image_scores, all_score_maps,
                    all_classifications, all_visualizations
                )
            finally:
                model.close()
        except Exception as e:
            error_msg = f"Error occurred during inference: {str(e)}\n{traceback.format_exc()}"
            self.error_callback(error_msg)
            self.logger.exception("Fatal error during inference")
            
    def adaptive_gaussian_blur(self, input_array, kernel_size=33, sigma=4):
        """Apply Gaussian blur to the input array"""
        try:
            import torch
            import torchvision.transforms as T
            if torch.is_tensor(input_array):
                if input_array.dim() == 2:
                    input_reshaped = input_array.unsqueeze(0).unsqueeze(0)
                    blurred = T.GaussianBlur(kernel_size, sigma=sigma)(input_reshaped)
                    return blurred.squeeze(0).squeeze(0)
                elif input_array.dim() == 3:
                    input_reshaped = input_array.unsqueeze(1)
                    blurred = T.GaussianBlur(kernel_size, sigma=sigma)(input_reshaped)
                    return blurred.squeeze(1)
                elif input_array.dim() == 4:
                    return T.GaussianBlur(kernel_size, sigma=sigma)(input_array)
                else:
                    raise ValueError(f"Unsupported tensor dimensions: {input_array.dim()}")
        except ImportError:
            pass
        # Fallback to numpy/scipy
        if hasattr(input_array, "detach"):
            input_array = input_array.detach().cpu().numpy()
        
        # Convert float16 to float32 if needed to avoid scipy compatibility issues
        if input_array.dtype == np.float16:
            input_array = input_array.astype(np.float32)
            
        try:
            from scipy.ndimage import gaussian_filter
            truncate = (kernel_size - 1) / (2 * sigma)
            if input_array.ndim == 2:
                return gaussian_filter(input_array, sigma=sigma, truncate=truncate)
            elif input_array.ndim == 3:
                blurred_batch = []
                for i in range(input_array.shape[0]):
                    blurred_img = gaussian_filter(
                        input_array[i], sigma=sigma, truncate=truncate
                    )
                    blurred_batch.append(blurred_img)
                return np.stack(blurred_batch, axis=0)
            elif input_array.ndim == 4:
                blurred_batch = []
                for b in range(input_array.shape[0]):
                    blurred_channels = []
                    for c in range(input_array.shape[1]):
                        blurred_channel = gaussian_filter(
                            input_array[b, c], sigma=sigma, truncate=truncate
                        )
                        blurred_channels.append(blurred_channel)
                    blurred_batch.append(np.stack(blurred_channels, axis=0))
                return np.stack(blurred_batch, axis=0)
            else:
                raise ValueError(f"Unsupported numpy array dimensions: {input_array.ndim}")
        except ImportError:
            raise ImportError("SciPy is required when PyTorch is not available")
            
    def generate_visualizations(self, images, score_maps, classifications, config):
        """Generate visualization results"""
        visualizations = []
        try:
            # Convert to numpy if needed
            if isinstance(score_maps, torch.Tensor):
                score_maps_np = score_maps.detach().cpu().numpy()
            else:
                score_maps_np = np.array(score_maps)
            # For highlighted_images, we need pixel-level classifications from score maps
            # not image-level classifications
            score_map_classifications = anomavision.classification(
                score_maps_np, config['thresh']
            )
            # Also get image-level classifications
            image_classifications = classifications
            # Generate different visualization types
            heatmap_images = anomavision.visualization.heatmap_images(
                np.asarray(images),
                score_maps_np,
                alpha=config.get('viz_alpha', 0.5),  # Keep consistent with CLI alpha value
            )
            highlighted_images = anomavision.visualization.highlighted_images(
                [images[i] for i in range(len(images))],
                score_map_classifications,  # Use pixel-level classification instead of image-level classification
                color=tuple(map(int, config.get('viz_color', "128,0,128").split(","))),
                alpha=0.5  # Keep consistent with CLI
            )
            # Generate boundary images with frames based on image classifications
            boundary_images = anomavision.visualization.framed_boundary_images(
                np.asarray(images),
                score_map_classifications,
                image_classifications,
                padding=config.get('viz_padding', 30),
            )
            # Combine visualizations
            for i in range(len(images)):
                vis_data = {
                    'original': images[i],
                    'heatmap': heatmap_images[i],
                    'highlighted': highlighted_images[i] if i < len(highlighted_images) else None,
                    'boundary': boundary_images[i] if i < len(boundary_images) else None,
                    'classification': image_classifications[i] if i < len(image_classifications) else 0
                }
                visualizations.append(vis_data)
        except Exception as e:
            logger.error(f"Error generating visualization results: {e}")
            # Even if visualization fails, return empty visualizations to avoid crashing
            for i in range(len(images)):
                vis_data = {
                    'original': images[i],
                    'heatmap': None,
                    'highlighted': None,
                    'boundary': None,
                    'classification': 0
                }
                visualizations.append(vis_data)
        return visualizations

class ExportWorker(threading.Thread):
    """Worker thread for model export"""
    def __init__(self, model_path, output_dir, export_format, config, progress_callback, finished_callback, error_callback):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.export_format = export_format
        self.config = config
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.error_callback = error_callback
        # Get dedicated export logger
        self.logger = get_logger("anomavision.export")
        
    def run(self):
        try:
            self.progress_callback("Starting model export...")
            self.logger.info("Starting AnomaVision model export process")
            self.logger.info(f"Export config: {self.config}")
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from export import ModelExporter
            # Setup logging for export
            export_logger = logging.getLogger("anomavision.export")
            export_logger.setLevel(logging.INFO)
            # Create exporter
            exporter = ModelExporter(
                model_path=Path(self.model_path),
                output_dir=Path(self.output_dir),
                logger=export_logger,
                device=self.config.get('device', 'cpu')
            )
            self.logger.info(f"Export configuration: device={self.config.get('device', 'cpu')}, layers={self.config.get('layer_indices', [0, 1])}")
            self.progress_callback(f"Export config: device={self.config.get('device', 'cpu')}, layers={self.config.get('layer_indices', [0, 1])}")
            # Get input shape
            h, w = self.config['crop_size'] if self.config['crop_size'] is not None else self.config['resize']
            input_shape = [1, 3, h, w]
            self.logger.info(f"Input shape: {tuple(input_shape)}")
            self.progress_callback(f"输入形状: {tuple(input_shape)}")
            # Export based on format
            if self.export_format == "onnx":
                self.logger.info("Exporting to ONNX format")
                output_path = exporter.export_onnx(
                    input_shape=tuple(input_shape),
                    output_name=f"padim_model.onnx",
                    opset_version=self.config.get('opset', 17),
                    dynamic_batch=self.config.get('dynamic_batch', True),
                )
                if output_path:
                    self.progress_callback(f"ONNX model exported to: {output_path}")
                    self.logger.info(f"ONNX model exported to: {output_path}")
                    self.finished_callback(str(output_path))
                else:
                    raise RuntimeError("ONNX export failed")
            elif self.export_format == "torchscript":
                self.logger.info("Exporting to TorchScript format")
                output_path = exporter.export_torchscript(
                    input_shape=tuple(input_shape),
                    output_name=f"padim_model.torchscript",
                    optimize=self.config.get('optimize', False),
                )
                if output_path:
                    self.progress_callback(f"TorchScript model exported to: {output_path}")
                    self.logger.info(f"TorchScript model exported to: {output_path}")
                    self.finished_callback(str(output_path))
                else:
                    raise RuntimeError("TorchScript export failed")
            elif self.export_format == "openvino":
                self.logger.info("Exporting to OpenVINO format")
                output_path = exporter.export_openvino(
                    input_shape=tuple(input_shape),
                    output_name=f"padim_model_openvino",
                    fp16=self.config.get('fp16', True),
                    dynamic_batch=self.config.get('dynamic_batch', True),
                )
                if output_path:
                    self.progress_callback(f"OpenVINO model exported to: {output_path}")
                    self.logger.info(f"OpenVINO model exported to: {output_path}")
                    self.finished_callback(str(output_path))
                else:
                    raise RuntimeError("OpenVINO export failed")
        except Exception as e:
            error_msg = f"Error occurred during export: {str(e)}\n{traceback.format_exc()}"
            self.error_callback(error_msg)
            self.logger.exception("Fatal error during export")

class AnomaVisionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AnomaVision - Anomaly Detection Tool")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.current_model = None
        self.inference_results = None
        self.current_result_index = 0
        
        # Load default config
        self.config = self.load_default_config()
        
        # Setup UI
        self.setup_ui()
        
        # Setup logging handler for GUI
        self.setup_logging()
        
    def parse_layer_indices(self, layer_indices_str):
        """Parse layer indices string to list of integers"""
        try:
            if isinstance(layer_indices_str, str):
                # Remove brackets and spaces, then split by comma
                cleaned = layer_indices_str.strip('[] ')
                if cleaned:
                    return [int(x.strip()) for x in cleaned.split(',')]
                else:
                    return [0, 1]  # Default value
            elif isinstance(layer_indices_str, list):
                return [int(x) for x in layer_indices_str]
            else:
                return [0, 1]  # Default value
        except Exception as e:
            logger.warning(f"Error parsing layer indices: {e}, using default value [0, 1]")
            return [0, 1]
            
    def load_default_config(self):
        """Load default configuration"""
        try:
            config = load_config("config.yml")
            # Ensure default visualization parameters are consistent with CLI
            if 'viz_alpha' not in config:
                config['viz_alpha'] = 0.5
            if 'viz_color' not in config:
                config['viz_color'] = "128,0,128"
            return config or {}
        except Exception as e:
            logger.warning(f"Failed to load configuration file: {e}")
            # Return default configuration, ensuring visualization parameters are consistent with CLI
            return {
                'viz_alpha': 0.5,
                'viz_color': "128,0,128"
            }
            
    def setup_ui(self):
        """Setup the main UI"""
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_training_tab()
        self.create_inference_tab()
        self.create_export_tab()
        self.create_log_tab()
        
    def create_training_tab(self):
        """Create the training tab"""
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Training")
        
        # Training configuration group
        config_frame = ttk.LabelFrame(self.training_frame, text="Training Configuration", padding=10)
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Dataset path
        dataset_path_frame = ttk.Frame(config_frame)
        dataset_path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dataset_path_frame, text="Dataset Path:").pack(side=tk.LEFT)
        self.dataset_path_var = tk.StringVar(value=self.config.get("dataset_path", ""))
        self.dataset_path_entry = ttk.Entry(dataset_path_frame, textvariable=self.dataset_path_var)
        self.dataset_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        dataset_path_button = ttk.Button(dataset_path_frame, text="Browse", command=self.browse_dataset_path)
        dataset_path_button.pack(side=tk.LEFT)
        
        # Class name
        class_name_frame = ttk.Frame(config_frame)
        class_name_frame.pack(fill=tk.X, pady=2)
        ttk.Label(class_name_frame, text="Class Name:").pack(side=tk.LEFT)
        self.class_name_var = tk.StringVar(value=self.config.get("class_name", "E85"))
        self.class_name_entry = ttk.Entry(class_name_frame, textvariable=self.class_name_var)
        self.class_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Backbone
        backbone_frame = ttk.Frame(config_frame)
        backbone_frame.pack(fill=tk.X, pady=2)
        ttk.Label(backbone_frame, text="Backbone:").pack(side=tk.LEFT)
        self.backbone_var = tk.StringVar(value=self.config.get("backbone", "resnet18"))
        self.backbone_combo = ttk.Combobox(backbone_frame, textvariable=self.backbone_var, 
                                          values=["resnet18", "wide_resnet50"], state="readonly")
        self.backbone_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Batch size
        batch_size_frame = ttk.Frame(config_frame)
        batch_size_frame.pack(fill=tk.X, pady=2)
        ttk.Label(batch_size_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.IntVar(value=self.config.get("batch_size", 2))
        self.batch_size_spin = ttk.Spinbox(batch_size_frame, from_=1, to=128, textvariable=self.batch_size_var)
        self.batch_size_spin.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Feature dimension
        feat_dim_frame = ttk.Frame(config_frame)
        feat_dim_frame.pack(fill=tk.X, pady=2)
        ttk.Label(feat_dim_frame, text="Feature Dimension:").pack(side=tk.LEFT)
        self.feat_dim_var = tk.IntVar(value=self.config.get("feat_dim", 50))
        self.feat_dim_spin = ttk.Spinbox(feat_dim_frame, from_=1, to=1000, textvariable=self.feat_dim_var)
        self.feat_dim_spin.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Layer indices
        layer_indices_frame = ttk.Frame(config_frame)
        layer_indices_frame.pack(fill=tk.X, pady=2)
        ttk.Label(layer_indices_frame, text="Layer Indices:").pack(side=tk.LEFT)
        layer_indices = self.config.get("layer_indices", [0, 1])
        # Ensure proper formatting for list
        if isinstance(layer_indices, list):
            layer_indices_str = str(layer_indices)
        else:
            layer_indices_str = str([0, 1])
        self.layer_indices_var = tk.StringVar(value=layer_indices_str)
        self.layer_indices_entry = ttk.Entry(layer_indices_frame, textvariable=self.layer_indices_var)
        self.layer_indices_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Resize
        resize_frame = ttk.Frame(config_frame)
        resize_frame.pack(fill=tk.X, pady=2)
        ttk.Label(resize_frame, text="Resize:").pack(side=tk.LEFT)
        self.resize_var = tk.StringVar(value=str(self.config.get("resize", [224, 224])))
        self.resize_entry = ttk.Entry(resize_frame, textvariable=self.resize_var)
        self.resize_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Crop size
        crop_size_frame = ttk.Frame(config_frame)
        crop_size_frame.pack(fill=tk.X, pady=2)
        ttk.Label(crop_size_frame, text="Crop Size:").pack(side=tk.LEFT)
        self.crop_size_var = tk.StringVar(value=str(self.config.get("crop_size", "")))
        self.crop_size_entry = ttk.Entry(crop_size_frame, textvariable=self.crop_size_var)
        self.crop_size_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Model output path
        model_output_path_frame = ttk.Frame(config_frame)
        model_output_path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(model_output_path_frame, text="Model Output Path:").pack(side=tk.LEFT)
        self.model_output_path_var = tk.StringVar(value=self.config.get("model_data_path", "./distributions"))
        self.model_output_path_entry = ttk.Entry(model_output_path_frame, textvariable=self.model_output_path_var)
        self.model_output_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        model_output_path_button = ttk.Button(model_output_path_frame, text="Browse", command=self.browse_model_output_path)
        model_output_path_button.pack(side=tk.LEFT)
        
        # Output model name
        output_model_name_frame = ttk.Frame(config_frame)
        output_model_name_frame.pack(fill=tk.X, pady=2)
        ttk.Label(output_model_name_frame, text="Output Model Name:").pack(side=tk.LEFT)
        self.output_model_name_var = tk.StringVar(value=self.config.get("output_model", "padim_model.pt"))
        self.output_model_name_entry = ttk.Entry(output_model_name_frame, textvariable=self.output_model_name_var)
        self.output_model_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Buttons
        buttons_frame = ttk.Frame(self.training_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        self.train_button = ttk.Button(buttons_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)
        self.train_stop_button = ttk.Button(buttons_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.train_stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.train_progress = ttk.Progressbar(self.training_frame, mode='indeterminate')
        self.train_progress.pack(fill=tk.X, padx=10, pady=5)
        self.train_progress.pack_forget()  # Hidden by default
        
    def create_inference_tab(self):
        """Create the inference tab"""
        self.inference_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.inference_frame, text="Inference")
        
        # Model and data group
        model_data_frame = ttk.LabelFrame(self.inference_frame, text="Model and Data", padding=10)
        model_data_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        # Model path
        model_path_frame = ttk.Frame(model_data_frame)
        model_path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(model_path_frame, text="Model Path:").pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar(value=self.config.get("model", "./distributions/padim_model.pt"))
        self.model_path_entry = ttk.Entry(model_path_frame, textvariable=self.model_path_var)
        self.model_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        model_path_button = ttk.Button(model_path_frame, text="Browse", command=self.browse_model_path)
        model_path_button.pack(side=tk.LEFT)
        
        # Image path
        image_path_frame = ttk.Frame(model_data_frame)
        image_path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(image_path_frame, text="Image Path:").pack(side=tk.LEFT)
        self.image_path_var = tk.StringVar(value=self.config.get("img_path", ""))
        self.image_path_entry = ttk.Entry(image_path_frame, textvariable=self.image_path_var)
        self.image_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        image_path_button = ttk.Button(image_path_frame, text="Browse", command=self.browse_image_path)
        image_path_button.pack(side=tk.LEFT)
        
        # Inference parameters group
        params_frame = ttk.LabelFrame(self.inference_frame, text="Inference Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        # Threshold
        threshold_frame = ttk.Frame(params_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(threshold_frame, text="Threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=self.config.get("thresh", 13.0))
        self.threshold_spin = ttk.Spinbox(threshold_frame, from_=0.0, to=100.0, increment=0.1, textvariable=self.threshold_var)
        self.threshold_spin.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Device
        device_frame = ttk.Frame(params_frame)
        device_frame.pack(fill=tk.X, pady=2)
        ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar(value=self.config.get("device", "auto"))
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                        values=["auto", "cpu", "cuda"], state="readonly")
        self.device_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Batch size
        infer_batch_size_frame = ttk.Frame(params_frame)
        infer_batch_size_frame.pack(fill=tk.X, pady=2)
        ttk.Label(infer_batch_size_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.infer_batch_size_var = tk.IntVar(value=self.config.get("batch_size", 1))
        self.infer_batch_size_spin = ttk.Spinbox(infer_batch_size_frame, from_=1, to=128, textvariable=self.infer_batch_size_var)
        self.infer_batch_size_spin.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Visualization padding
        viz_padding_frame = ttk.Frame(params_frame)
        viz_padding_frame.pack(fill=tk.X, pady=2)
        ttk.Label(viz_padding_frame, text="Boundary Padding:").pack(side=tk.LEFT)
        self.viz_padding_var = tk.IntVar(value=self.config.get("viz_padding", 30))
        self.viz_padding_spin = ttk.Spinbox(viz_padding_frame, from_=0, to=100, textvariable=self.viz_padding_var)
        self.viz_padding_spin.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(self.inference_frame, text="Inference Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Results info
        self.results_info_var = tk.StringVar(value="No inference results")
        self.results_info_label = ttk.Label(results_frame, textvariable=self.results_info_var)
        self.results_info_label.pack()
        
        # Anomaly score display (single label at the top center)
        self.anomaly_score_var = tk.StringVar(value="Anomaly Score: 0.00")
        self.anomaly_score_label = ttk.Label(results_frame, textvariable=self.anomaly_score_var, 
                                            font=("Arial", 12, "bold"))
        self.anomaly_score_label.pack(pady=5)
        
        # Visualization area - Use larger minimum size for better display effect
        vis_frame = ttk.Frame(results_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a canvas for visualization with a larger minimum size
        self.vis_canvas = tk.Canvas(vis_frame, bg="lightgray", height=400)
        self.vis_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create labels for different visualizations with fixed minimum sizes
        self.original_image_label = tk.Label(self.vis_canvas, text="Original Image", bg="white", relief="solid", 
                                           width=25, height=15)
        self.heatmap_label = tk.Label(self.vis_canvas, text="Heatmap", bg="white", relief="solid", 
                                    width=25, height=15)
        self.highlighted_label = tk.Label(self.vis_canvas, text="Highlighted Anomalies", bg="white", relief="solid", 
                                        width=25, height=15)
        self.boundary_label = tk.Label(self.vis_canvas, text="Boundary Detection", bg="white", relief="solid", 
                                     width=25, height=15)
        
        # Position labels in canvas with better spacing
        self.original_image_label.place(relx=0.125, rely=0.5, anchor="center")
        self.heatmap_label.place(relx=0.375, rely=0.5, anchor="center")
        self.highlighted_label.place(relx=0.625, rely=0.5, anchor="center")
        self.boundary_label.place(relx=0.875, rely=0.5, anchor="center")
        
        # Buttons
        buttons_frame = ttk.Frame(self.inference_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        self.infer_button = ttk.Button(buttons_frame, text="Start Inference", command=self.start_inference)
        self.infer_button.pack(side=tk.LEFT, padx=5)
        self.infer_stop_button = ttk.Button(buttons_frame, text="Stop Inference", command=self.stop_inference, state=tk.DISABLED)
        self.infer_stop_button.pack(side=tk.LEFT, padx=5)
        self.prev_result_button = ttk.Button(buttons_frame, text="Previous Result", command=self.show_prev_result, state=tk.DISABLED)
        self.prev_result_button.pack(side=tk.LEFT, padx=5)
        self.next_result_button = ttk.Button(buttons_frame, text="Next Result", command=self.show_next_result, state=tk.DISABLED)
        self.next_result_button.pack(side=tk.LEFT, padx=5)
        
    def create_export_tab(self):
        """Create the export tab"""
        self.export_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.export_frame, text="Export")
        
        # Export configuration group
        config_frame = ttk.LabelFrame(self.export_frame, text="Export Configuration", padding=10)
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Model path
        export_model_path_frame = ttk.Frame(config_frame)
        export_model_path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(export_model_path_frame, text="Model Path:").pack(side=tk.LEFT)
        self.export_model_path_var = tk.StringVar(value=self.config.get("model", "./distributions/padim_model.pt"))
        self.export_model_path_entry = ttk.Entry(export_model_path_frame, textvariable=self.export_model_path_var)
        self.export_model_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        export_model_path_button = ttk.Button(export_model_path_frame, text="Browse", command=self.browse_export_model_path)
        export_model_path_button.pack(side=tk.LEFT)
        
        # Output directory
        export_output_dir_frame = ttk.Frame(config_frame)
        export_output_dir_frame.pack(fill=tk.X, pady=2)
        ttk.Label(export_output_dir_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.export_output_dir_var = tk.StringVar(value=self.config.get("model_data_path", "./distributions/anomav_exp"))
        self.export_output_dir_entry = ttk.Entry(export_output_dir_frame, textvariable=self.export_output_dir_var)
        self.export_output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        export_output_dir_button = ttk.Button(export_output_dir_frame, text="Browse", command=self.browse_export_output_dir)
        export_output_dir_button.pack(side=tk.LEFT)
        
        # Export format
        export_format_frame = ttk.Frame(config_frame)
        export_format_frame.pack(fill=tk.X, pady=2)
        ttk.Label(export_format_frame, text="Export Format:").pack(side=tk.LEFT)
        self.export_format_var = tk.StringVar(value="onnx")
        self.export_format_combo = ttk.Combobox(export_format_frame, textvariable=self.export_format_var, 
                                               values=["onnx", "torchscript", "openvino", "all"], state="readonly")
        self.export_format_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # ONNX specific options
        onnx_opset_frame = ttk.Frame(config_frame)
        onnx_opset_frame.pack(fill=tk.X, pady=2)
        ttk.Label(onnx_opset_frame, text="ONNX Opset:").pack(side=tk.LEFT)
        self.onnx_opset_var = tk.IntVar(value=self.config.get("opset", 17))
        self.onnx_opset_spin = ttk.Spinbox(onnx_opset_frame, from_=1, to=20, textvariable=self.onnx_opset_var)
        self.onnx_opset_spin.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Dynamic batch checkbox
        self.dynamic_batch_var = tk.BooleanVar(value=self.config.get("dynamic_batch", True))
        self.dynamic_batch_check = ttk.Checkbutton(config_frame, text="Dynamic Batch", variable=self.dynamic_batch_var)
        self.dynamic_batch_check.pack(anchor=tk.W)
        
        # Device selection
        export_device_frame = ttk.Frame(config_frame)
        export_device_frame.pack(fill=tk.X, pady=2)
        ttk.Label(export_device_frame, text="Export Device:").pack(side=tk.LEFT)
        self.export_device_var = tk.StringVar(value=self.config.get("device", "cpu"))
        self.export_device_combo = ttk.Combobox(export_device_frame, textvariable=self.export_device_var, 
                                               values=["cpu", "cuda"], state="readonly")
        self.export_device_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Progress bar
        self.export_progress = ttk.Progressbar(self.export_frame, mode='indeterminate')
        self.export_progress.pack(fill=tk.X, padx=10, pady=5)
        self.export_progress.pack_forget()  # Hidden by default
        
        # Export info
        self.export_info_var = tk.StringVar(value="After export is complete, files will be saved to the specified directory")
        self.export_info_label = ttk.Label(self.export_frame, textvariable=self.export_info_var)
        self.export_info_label.pack(pady=5)
        
        # Buttons
        buttons_frame = ttk.Frame(self.export_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        self.export_button = ttk.Button(buttons_frame, text="Start Export", command=self.start_export)
        self.export_button.pack(side=tk.LEFT, padx=5)
        self.export_stop_button = ttk.Button(buttons_frame, text="Stop Export", command=self.stop_export, state=tk.DISABLED)
        self.export_stop_button.pack(side=tk.LEFT, padx=5)
        
    def create_log_tab(self):
        """Create the log tab"""
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Logs")
        
        # Log controls group
        controls_frame = ttk.LabelFrame(self.log_frame, text="Log Control", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Auto-save checkbox
        self.autosave_log_var = tk.BooleanVar(value=True)
        self.autosave_log_check = ttk.Checkbutton(controls_frame, text="Auto-save Logs", variable=self.autosave_log_var)
        self.autosave_log_check.pack(side=tk.LEFT)
        
        # Log level selection
        log_level_frame = ttk.Frame(controls_frame)
        log_level_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(log_level_frame, text="Log Level:").pack(side=tk.LEFT)
        self.log_level_var = tk.StringVar(value="INFO")
        self.log_level_combo = ttk.Combobox(log_level_frame, textvariable=self.log_level_var, 
                                           values=["DEBUG", "INFO", "WARNING", "ERROR"], state="readonly",
                                           width=10)
        self.log_level_combo.pack(side=tk.LEFT)
        self.log_level_combo.bind("<<ComboboxSelected>>", self.change_log_level)
        
        # Clear log button
        clear_log_button = ttk.Button(controls_frame, text="Clear Logs", command=self.clear_log)
        clear_log_button.pack(side=tk.LEFT, padx=5)
        
        # Save log button
        save_log_button = ttk.Button(controls_frame, text="Save Logs", command=self.save_log)
        save_log_button.pack(side=tk.LEFT, padx=5)
        
        # Log info
        info_frame = ttk.LabelFrame(self.log_frame, text="Log Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X)
        
        ttk.Label(info_grid, text="Log File:").grid(row=0, column=0, sticky=tk.W)
        self.log_file_var = tk.StringVar(value="Not set")
        self.log_file_label = ttk.Label(info_grid, textvariable=self.log_file_var)
        self.log_file_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(info_grid, text="File Size:").grid(row=1, column=0, sticky=tk.W)
        self.log_size_var = tk.StringVar(value="0 bytes")
        self.log_size_label = ttk.Label(info_grid, textvariable=self.log_size_var)
        self.log_size_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(info_grid, text="Lines:").grid(row=2, column=0, sticky=tk.W)
        self.log_lines_var = tk.StringVar(value="0 lines")
        self.log_lines_label = ttk.Label(info_grid, textvariable=self.log_lines_var)
        self.log_lines_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # Log text area
        log_text_frame = ttk.LabelFrame(self.log_frame, text="Log Output", padding=10)
        log_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_text_frame, wrap=tk.WORD, font=("Courier New", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_logging(self):
        """Setup logging to display in GUI with auto-save functionality"""
        # Create logs directory if it doesn't exist
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = self.logs_dir / f"anomavision_gui_{timestamp}.log"
        
        # Update log file label
        self.log_file_var.set(str(self.log_file_path))
        
        class GuiLogHandler(logging.Handler):
            def __init__(self, text_widget, log_file_path, autosave_var, log_info_callback):
                super().__init__()
                self.text_widget = text_widget
                self.log_file_path = log_file_path
                self.autosave_var = autosave_var
                self.log_info_callback = log_info_callback
                self.log_lines = 0
                
            def emit(self, record):
                msg = self.format(record)
                # Add to GUI
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.configure(state='disabled')
                self.text_widget.see(tk.END)
                self.log_lines += 1
                # Auto-save to file if enabled
                if self.autosave_var.get():
                    try:
                        with open(self.log_file_path, 'a', encoding='utf-8') as f:
                            f.write(msg + '\n')
                        self.log_info_callback()
                    except Exception as e:
                        # Avoid infinite recursion of logging errors
                        print(f"Failed to write to log file: {e}")
                        
        # Create GUI handler
        self.gui_handler = GuiLogHandler(
            self.log_text,
            self.log_file_path,
            self.autosave_log_var,
            self.update_log_info
        )
        self.gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Add to anomavision logger
        anomavision_logger = logging.getLogger("anomavision")
        anomavision_logger.addHandler(self.gui_handler)
        anomavision_logger.setLevel(logging.INFO)
        
        # Also add to root logger for general messages
        root_logger = logging.getLogger()
        root_logger.addHandler(self.gui_handler)
        
        # Write initial log entry
        logger.info(f"AnomaVision GUI started - Log file: {self.log_file_path}")
        
    def update_log_info(self):
        """Update log information display"""
        try:
            if self.log_file_path.exists():
                file_size = self.log_file_path.stat().st_size
                self.log_size_var.set(f"{file_size:,} bytes")
                # Count lines more efficiently
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                self.log_lines_var.set(f"{line_count:,} lines")
        except Exception as e:
            print(f"Failed to update log info: {e}")
            
    def change_log_level(self, event=None):
        """Change logging level"""
        level_mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }
        log_level = level_mapping.get(self.log_level_var.get(), logging.INFO)
        # Update anomavision logger level
        anomavision_logger = logging.getLogger("anomavision")
        anomavision_logger.setLevel(log_level)
        # Update root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        logger.info(f"Log level changed to: {self.log_level_var.get()}")
        
    def save_log(self):
        """Manually save current log content to file"""
        try:
            # Create a new log file with timestamp for manual save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            manual_save_path = self.logs_dir / f"anomavision_gui_manual_{timestamp}.log"
            with open(manual_save_path, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Save Successful", f"Log manually saved to:\n{manual_save_path}")
            logger.info(f"Log manually saved to: {manual_save_path}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Error saving log:\n{str(e)}")
            logger.error(f"Failed to manually save log: {e}")
            
    def clear_log(self):
        """Clear the log display"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        
    def browse_dataset_path(self):
        """Browse for dataset path"""
        path = filedialog.askdirectory()
        if path:
            self.dataset_path_var.set(path)
            
    def browse_model_output_path(self):
        """Browse for model output path"""
        path = filedialog.askdirectory()
        if path:
            self.model_output_path_var.set(path)
            
    def browse_model_path(self):
        """Browse for model path"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Model Files", "*.pt *.pth *.onnx *.torchscript *.xml")]
        )
        if file_path:
            self.model_path_var.set(file_path)
            
    def browse_image_path(self):
        """Browse for image path"""
        path = filedialog.askdirectory()
        if path:
            self.image_path_var.set(path)
            
    def browse_export_model_path(self):
        """Browse for export model path"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Model Files", "*.pt *.pth *.onnx *.torchscript *.xml")]
        )
        if file_path:
            self.export_model_path_var.set(file_path)
            
    def browse_export_output_dir(self):
        """Browse for export output directory"""
        path = filedialog.askdirectory()
        if path:
            self.export_output_dir_var.set(path)
            
    def start_training(self):
        """Start the training process"""
        try:
            # Get configuration from UI
            config = {
                "dataset_path": self.dataset_path_var.get(),
                "class_name": self.class_name_var.get(),
                "backbone": self.backbone_var.get(),
                "batch_size": self.batch_size_var.get(),
                "feat_dim": self.feat_dim_var.get(),
                "layer_indices": self.parse_layer_indices(self.layer_indices_var.get()),
                "resize": eval(self.resize_var.get()),
                "crop_size": eval(self.crop_size_var.get()) if self.crop_size_var.get() else None,
                "normalize": self.config.get("normalize", True),
                "norm_mean": self.config.get("norm_mean", [0.485, 0.456, 0.406]),
                "norm_std": self.config.get("norm_std", [0.229, 0.224, 0.225]),
                "model_data_path": self.model_output_path_var.get(),
                "output_model": self.output_model_name_var.get(),
                "run_name": self.config.get("run_name", "anomav_exp"),
                "log_level": self.config.get("log_level", "INFO"),
            }
            # Validate required fields
            if not config["dataset_path"]:
                messagebox.showwarning("Warning", "Please select dataset path")
                return
            if not config["class_name"]:
                messagebox.showwarning("Warning", "Please enter class name")
                return
            # Disable UI elements
            self.train_button.config(state=tk.DISABLED)
            self.train_stop_button.config(state=tk.NORMAL)
            self.train_progress.pack(fill=tk.X, padx=10, pady=5)
            self.train_progress.start()
            # Start training worker
            self.training_worker = TrainingWorker(
                config,
                self.update_log,
                self.training_finished,
                self.training_error
            )
            self.training_worker.start()
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred while starting training:\n{str(e)}")
            self.train_button.config(state=tk.NORMAL)
            self.train_stop_button.config(state=tk.DISABLED)
            self.train_progress.pack_forget()
            
    def stop_training(self):
        """Stop the training process"""
        # Note: In Python, forcefully terminating threads is not recommended
        # We'll just disable the UI and let the thread finish naturally
        self.train_button.config(state=tk.NORMAL)
        self.train_stop_button.config(state=tk.DISABLED)
        self.train_progress.stop()
        self.train_progress.pack_forget()
        self.update_log("Training stopped")
        
    def training_finished(self, model):
        """Handle training finished"""
        self.current_model = model
        self.train_button.config(state=tk.NORMAL)
        self.train_stop_button.config(state=tk.DISABLED)
        self.train_progress.stop()
        self.train_progress.pack_forget()
        self.update_log("Training completed!")
        messagebox.showinfo("Completed", "Model training completed!")
        
    def training_error(self, error_msg):
        """Handle training error"""
        self.train_button.config(state=tk.NORMAL)
        self.train_stop_button.config(state=tk.DISABLED)
        self.train_progress.stop()
        self.train_progress.pack_forget()
        self.update_log(f"Training error: {error_msg}")
        messagebox.showerror("Training Error", error_msg)
        
    def start_inference(self):
        """Start the inference process"""
        try:
            # Get configuration from UI
            config = {
                "resize": eval(self.resize_var.get()) if self.resize_var.get() else [224, 224],
                "crop_size": eval(self.crop_size_var.get()) if self.crop_size_var.get() else None,
                "normalize": self.config.get("normalize", True),
                "norm_mean": self.config.get("norm_mean", [0.485, 0.456, 0.406]),
                "norm_std": self.config.get("norm_std", [0.229, 0.224, 0.225]),
                "batch_size": self.infer_batch_size_var.get(),
                "thresh": self.threshold_var.get(),
                "device": self.device_var.get(),
                "num_workers": self.config.get("num_workers", 1),
                "pin_memory": self.config.get("pin_memory", False),
                "viz_alpha": self.config.get("viz_alpha", 0.5),  # Keep consistent with CLI
                "viz_color": self.config.get("viz_color", "128,0,128"),  # Keep consistent with CLI
                "viz_padding": self.viz_padding_var.get(),  # Boundary padding
                "layer_indices": self.parse_layer_indices(self.config.get("layer_indices", [0, 1])),
            }
            model_path = self.model_path_var.get()
            img_path = self.image_path_var.get()
            # Validate required fields
            if not model_path:
                messagebox.showwarning("Warning", "Please select model file")
                return
            if not img_path:
                messagebox.showwarning("Warning", "Please select image directory")
                return
            if not os.path.exists(model_path):
                messagebox.showwarning("Warning", "Model file does not exist")
                return
            if not os.path.exists(img_path):
                messagebox.showwarning("Warning", "Image directory does not exist")
                return
            # Disable UI elements
            self.infer_button.config(state=tk.DISABLED)
            self.infer_stop_button.config(state=tk.NORMAL)
            self.prev_result_button.config(state=tk.DISABLED)
            self.next_result_button.config(state=tk.DISABLED)
            self.results_info_var.set("Processing...")
            # Start inference worker
            self.inference_worker = InferenceWorker(
                model_path, img_path, config,
                self.update_log,
                self.inference_finished,
                self.inference_error
            )
            self.inference_worker.start()
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred while starting inference:\n{str(e)}")
            self.infer_button.config(state=tk.NORMAL)
            self.infer_stop_button.config(state=tk.DISABLED)
            
    def stop_inference(self):
        """Stop the inference process"""
        # Note: In Python, forcefully terminating threads is not recommended
        # We'll just disable the UI and let the thread finish naturally
        self.infer_button.config(state=tk.NORMAL)
        self.infer_stop_button.config(state=tk.DISABLED)
        self.results_info_var.set("Inference stopped")
        self.update_log("Inference stopped")
        
    def inference_finished(self, images, scores, maps, classifications, visualizations):
        """Handle inference finished"""
        self.inference_results = {
            'images': images,
            'scores': scores,
            'maps': maps,
            'classifications': classifications,
            'visualizations': visualizations
        }
        self.current_result_index = 0
        self.infer_button.config(state=tk.NORMAL)
        self.infer_stop_button.config(state=tk.DISABLED)
        if visualizations:
            if len(visualizations) > 1:
                self.prev_result_button.config(state=tk.NORMAL)
                self.next_result_button.config(state=tk.NORMAL)
            self.show_result(0)
            self.results_info_var.set(f"Showing result: 1/{len(visualizations)}")
        else:
            self.results_info_var.set("No visualization results")
        self.update_log(f"Inference completed! Processed {len(images)} images")
        messagebox.showinfo("Completed", f"Inference completed! Processed {len(images)} images")
        
    def inference_error(self, error_msg):
        """Handle inference error"""
        self.infer_button.config(state=tk.NORMAL)
        self.infer_stop_button.config(state=tk.DISABLED)
        self.results_info_var.set("Inference failed")
        self.update_log(f"Inference error: {error_msg}")
        messagebox.showerror("Inference Error", error_msg)
        
    def show_prev_result(self):
        """Show previous result"""
        if self.inference_results and self.inference_results['visualizations']:
            self.current_result_index = (self.current_result_index - 1) % len(self.inference_results['visualizations'])
            self.show_result(self.current_result_index)
            self.results_info_var.set(f"Showing result: {self.current_result_index+1}/{len(self.inference_results['visualizations'])}")
            
    def show_next_result(self):
        """Show next result"""
        if self.inference_results and self.inference_results['visualizations']:
            self.current_result_index = (self.current_result_index + 1) % len(self.inference_results['visualizations'])
            self.show_result(self.current_result_index)
            self.results_info_var.set(f"Showing result: {self.current_result_index+1}/{len(self.inference_results['visualizations'])}")
            
    def show_result(self, index):
        """Show a specific result"""
        if not self.inference_results or not self.inference_results['visualizations']:
            return
        if index >= len(self.inference_results['visualizations']):
            return
        vis_data = self.inference_results['visualizations'][index]
        # Get the score for this image
        score = 0.0
        if 'scores' in self.inference_results and index < len(self.inference_results['scores']):
            score = self.inference_results['scores'][index]
        # Display the anomaly score at the top
        self.anomaly_score_var.set(f"Anomaly Score: {score:.2f}")
        # Get classification result
        # Note: 0 means anomaly present, 1 means normal
        is_anomaly = vis_data.get('classification', 0) == 0
        # Set border style based on classification
        border_color = "red" if is_anomaly else "green"
        # Display original image with border
        self.display_image(self.original_image_label, vis_data['original'], border_color)
        # Display heatmap with border
        if 'heatmap' in vis_data:
            self.display_image(self.heatmap_label, vis_data['heatmap'], border_color)
        # Display highlighted image with border
        if 'highlighted' in vis_data:
            self.display_image(self.highlighted_label, vis_data['highlighted'], border_color)
        # Display boundary image with border
        if 'boundary' in vis_data:
            self.display_image(self.boundary_label, vis_data['boundary'], border_color)
            
    def display_image(self, label, image, border_color="black"):
        """Display an image in a Label"""
        try:
            # Handle PyTorch tensors by converting to numpy
            if hasattr(image, 'detach'):  # PyTorch tensor
                image = image.detach().cpu().numpy()
            # Convert image to RGB if needed
            if isinstance(image, np.ndarray):
                # Ensure the image data is in the right format (H, W, C) and uint8
                if image.ndim == 4 and image.shape[0] == 1:  # Batch size of 1
                    image = image[0]
                if image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # Channel first format
                    if image.shape[0] == 1:  # Grayscale
                        image = image[0]
                    elif image.shape[0] in [3, 4]:  # RGB or RGBA
                        image = np.transpose(image, (1, 2, 0))
                # Convert to uint8 if needed
                if image.dtype != np.uint8:
                    # Normalize to 0-255 range if it's float
                    if image.dtype in [np.float32, np.float64]:
                        image_min, image_max = image.min(), image.max()
                        if image_max > image_min:
                            image = (image - image_min) / (image_max - image_min) * 255
                        image = image.astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                # The image data is already in RGB format from the preprocessing pipeline
                if image.ndim == 3:
                    # Image is already in RGB format, no conversion needed
                    pass
                elif image.ndim == 2:
                    # Grayscale to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage for tkinter
                height, width = image.shape[:2]
                
                # 获取标签的尺寸并调整图像大小以适应标签
                label.update_idletasks()  # 确保获取到最新的尺寸
                label_width = label.winfo_width()
                label_height = label.winfo_height()
                
                # 如果标签尺寸有效，则调整图像大小
                if label_width > 1 and label_height > 1:
                    # 计算缩放比例，保持宽高比
                    scale_w = label_width / width
                    scale_h = label_height / height
                    scale = min(scale_w, scale_h)
                    
                    # 确保缩放比例不会放大图像超过原始尺寸
                    scale = min(scale, 1.0)
                    
                    # 计算新的尺寸
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    # 只有在新尺寸小于原始尺寸时才进行缩放
                    if new_width < width and new_height < height:
                        # 使用OpenCV调整图像大小
                        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                pil_image = Image.fromarray(image)
                tk_image = ImageTk.PhotoImage(pil_image)
                
                # Update label with image
                label.config(image=tk_image, text="", bg="white", relief="solid", bd=3, 
                           highlightbackground=border_color, highlightthickness=2)
                # Keep a reference to avoid garbage collection
                label.image = tk_image
            else:
                # Assume it's already a QImage or QPixmap
                label.config(text="Image", bg="white", relief="solid", bd=3, 
                           highlightbackground=border_color, highlightthickness=2)
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            label.config(text="Unable to display image", bg="white", relief="solid", bd=1)
            
    def start_export(self):
        """Start the export process"""
        try:
            model_path = self.export_model_path_var.get()
            output_dir = self.export_output_dir_var.get()
            export_format = self.export_format_var.get()
            # Validate required fields
            if not model_path:
                messagebox.showwarning("Warning", "Please select model file")
                return
            if not output_dir:
                messagebox.showwarning("Warning", "Please select output directory")
                return
            if not os.path.exists(model_path):
                messagebox.showwarning("Warning", "Model file does not exist")
                return
            if not os.path.exists(output_dir):
                messagebox.showwarning("Warning", "Output directory does not exist")
                return
            # Get configuration from UI
            config = {
                "resize": eval(self.resize_var.get()) if self.resize_var.get() else [224, 224],
                "crop_size": eval(self.crop_size_var.get()) if self.crop_size_var.get() else None,
                "normalize": self.config.get("normalize", True),
                "norm_mean": self.config.get("norm_mean", [0.485, 0.456, 0.406]),
                "norm_std": self.config.get("norm_std", [0.229, 0.224, 0.225]),
                "opset": self.onnx_opset_var.get(),
                "dynamic_batch": self.dynamic_batch_var.get(),
                "device": self.export_device_var.get(),
                "fp16": self.config.get("fp16", True),
                "optimize": self.config.get("optimize", False),
                "layer_indices": self.parse_layer_indices(self.config.get("layer_indices", [0, 1])),
            }
            # Disable UI elements
            self.export_button.config(state=tk.DISABLED)
            self.export_stop_button.config(state=tk.NORMAL)
            self.export_progress.pack(fill=tk.X, padx=10, pady=5)
            self.export_progress.start()
            # Start export worker
            self.export_worker = ExportWorker(
                model_path, output_dir, export_format, config,
                self.update_log,
                self.export_finished,
                self.export_error
            )
            self.export_worker.start()
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred while starting export:\n{str(e)}")
            self.export_button.config(state=tk.NORMAL)
            self.export_stop_button.config(state=tk.DISABLED)
            self.export_progress.pack_forget()
            
    def stop_export(self):
        """Stop the export process"""
        # Note: In Python, forcefully terminating threads is not recommended
        # We'll just disable the UI and let the thread finish naturally
        self.export_button.config(state=tk.NORMAL)
        self.export_stop_button.config(state=tk.DISABLED)
        self.export_progress.stop()
        self.export_progress.pack_forget()
        self.update_log("Export stopped")
        
    def export_finished(self, output_path):
        """Handle export finished"""
        self.export_button.config(state=tk.NORMAL)
        self.export_stop_button.config(state=tk.DISABLED)
        self.export_progress.stop()
        self.export_progress.pack_forget()
        self.export_info_var.set(f"Export completed! File saved to: {output_path}")
        self.update_log(f"Export completed! File saved to: {output_path}")
        messagebox.showinfo("Completed", f"Model export completed!\nFile saved to: {output_path}")
        
    def export_error(self, error_msg):
        """Handle export error"""
        self.export_button.config(state=tk.NORMAL)
        self.export_stop_button.config(state=tk.DISABLED)
        self.export_progress.stop()
        self.export_progress.pack_forget()
        self.export_info_var.set("Export failed")
        self.update_log(f"Export error: {error_msg}")
        messagebox.showerror("Export Error", error_msg)
        
    def update_log(self, message):
        """Update the log display"""
        # This is called from worker threads, so we need to use thread-safe methods
        self.log_text.after(0, self._append_log_message, message)
        
    def _append_log_message(self, message):
        """Thread-safe method to append log message"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)

def main():
    root = tk.Tk()
    app = AnomaVisionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
