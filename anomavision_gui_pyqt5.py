import sys
import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import traceback

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QFileDialog, QMessageBox, QGroupBox, QCheckBox, QProgressBar,
    QFormLayout, QScrollArea, QGridLayout, QSplitter, QSizePolicy, QAbstractSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

# 添加 contextlib 和 time 导入用于性能分析
import contextlib
import time

import anomavision
from anomavision.padim import Padim
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType
from anomavision.utils import create_image_transform, get_logger
from anomavision.config import load_config

# 添加 Profiler 类用于性能分析
class Profiler(contextlib.ContextDecorator):
    """
    AnomaVision Performance Profiler for accurate timing measurements.

    Designed for anomaly detection inference pipelines with CUDA synchronization
    support for precise GPU timing measurements.

    Usage:
        @AnomaVisionProfiler() decorator or 'with AnomaVisionProfiler():' context manager
    """

    def __init__(self, accumulated_time=0.0):
        """
        Initialize AnomaVision profiler.

        Args:
            accumulated_time (float): Initial accumulated time in seconds
        """
        self.accumulated_time = (
            accumulated_time  # Total time accumulated across multiple runs
        )
        self.elapsed_time = 0.0  # Time for the last measurement
        self.cuda_available = (
            torch.cuda.is_available()
        )  # Check if CUDA timing sync is needed
        self._start_time = 0.0  # Internal start time marker

    def __enter__(self):
        """Enter context manager - start timing for AnomaVision operation."""
        self._start_time = self._get_precise_time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit context manager - complete timing measurement.

        Calculates elapsed time and adds to accumulated total for AnomaVision metrics.
        """
        self.elapsed_time = (
            self._get_precise_time() - self._start_time
        )  # delta-time for this operation
        self.accumulated_time += (
            self.elapsed_time
        )  # accumulate for total AnomaVision runtime

    def _get_precise_time(self):
        """
        Get precise timestamp with CUDA synchronization if available.

        For AnomaVision GPU inference, this ensures accurate timing by
        synchronizing CUDA operations before measuring time.

        Returns:
            float: Precise timestamp in seconds
        """
        if self.cuda_available:
            torch.cuda.synchronize()  # Ensure all CUDA operations complete for accurate timing
        return time.time()

    def reset(self):
        """Reset accumulated time counter for new AnomaVision measurement session."""
        self.accumulated_time = 0.0
        self.elapsed_time = 0.0

    def get_fps(self, num_samples):
        """
        Calculate FPS (Frames Per Second) for AnomaVision inference.

        Args:
            num_samples (int): Number of images/samples processed

        Returns:
            float: FPS based on accumulated time
        """
        if self.accumulated_time > 0:
            return num_samples / self.accumulated_time
        return 0.0

    def get_avg_time_ms(self, num_operations):
        """
        Get average time per operation in milliseconds.

        Args:
            num_operations (int): Number of operations performed

        Returns:
            float: Average time per operation in milliseconds
        """
        if num_operations > 0:
            return (self.accumulated_time / num_operations) * 1000
        return 0.0

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingWorker(QThread):
    """Worker thread for model training"""
    progress_update = pyqtSignal(str)
    training_finished = pyqtSignal(object)  # Pass the trained model
    error_occurred = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        # 获取专门的训练日志记录器，与CLI保持一致
        self.logger = get_logger("anomavision.train")

    def run(self):
        try:
            self.progress_update.emit("开始训练模型...")
            self.logger.info("Starting AnomaVision model training process")
            self.logger.info(f"Training config: {self.config}")
            
            # Create dataset
            self.progress_update.emit("加载训练数据...")
            self.logger.info("Creating AnomaVision training dataset")
            root = os.path.join(
                os.path.realpath(self.config['dataset_path']), 
                self.config['class_name'], 
                "train", 
                "good"
            )
            self.logger.info(f"Training data path: {root}")
            
            if not os.path.isdir(root):
                raise FileNotFoundError(f'训练数据目录不存在: {root}')
                
            ds = anomavision.AnodetDataset(
                root,
                resize=self.config['resize'],
                crop_size=self.config['crop_size'],
                normalize=self.config['normalize'],
                mean=self.config['norm_mean'],
                std=self.config['norm_std'],
            )

            if len(ds) == 0:
                raise ValueError(f"训练目录中没有找到图像: {root}")

            dl = DataLoader(ds, batch_size=int(self.config['batch_size']), shuffle=False)
            self.progress_update.emit(f"数据集加载完成: {len(ds)} 张图像")
            self.logger.info(f"dataset: {len(ds)} images | batch_size={self.config['batch_size']}")

            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.progress_update.emit(f"使用设备: {device.type}")
            self.logger.info(f"device: {device.type} (cuda_available={torch.cuda.is_available()})")
            if device.type == "cuda":
                self.logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
                self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

            # Create and train model
            self.progress_update.emit("初始化模型...")
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
            self.progress_update.emit(f"模型配置: backbone={self.config['backbone']}, layers={layer_indices}, feat_dim={self.config['feat_dim']}")

            self.progress_update.emit("开始训练...")
            self.logger.info("Starting model training")
            t_fit = time.perf_counter()
            padim.fit(dl)
            self.logger.info(f"fit: completed in {time.perf_counter() - t_fit:.2fs}")
            self.progress_update.emit("训练完成!")

            # Save model
            model_path = Path(self.config['model_data_path']) / self.config['output_model']
            torch.save(padim, str(model_path))
            self.progress_update.emit(f"模型已保存到: {model_path}")
            self.logger.info(f"Model saved to: {model_path}")

            # Save statistics
            try:
                stats_path = model_path.with_suffix(".pth")
                padim.save_statistics(str(stats_path), half=True)
                self.progress_update.emit(f"统计信息已保存到: {stats_path}")
                self.logger.info(f"Statistics saved to: {stats_path}")
            except Exception as e:
                self.progress_update.emit(f"保存统计信息失败: {e}")
                self.logger.warning(f"Failed to save statistics: {e}")

            self.training_finished.emit(padim)

        except Exception as e:
            error_msg = f"训练过程中发生错误: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
            self.logger.exception("Fatal error during training")


class InferenceWorker(QThread):
    """Worker thread for model inference"""
    progress_update = pyqtSignal(str)
    inference_finished = pyqtSignal(object, object, object, object, object)  # images, scores, maps, classifications, visualizations
    performance_update = pyqtSignal(dict)  # 新增性能更新信号
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path, img_path, config):
        super().__init__()
        self.model_path = model_path
        self.img_path = img_path
        self.config = config
        # 获取专门的推理日志记录器，与CLI保持一致
        self.logger = get_logger("anomavision.detect")

    def run(self):
        try:
            self.progress_update.emit("开始推理...")
            self.logger.info("Starting AnomaVision anomaly detection inference process")
            self.logger.info(f"Inference config: {self.config}")
            
            # 初始化性能分析器
            profilers = {
                "model_loading": Profiler(),
                "data_loading": Profiler(),
                "warmup": Profiler(),
                "inference": Profiler(),
                "postprocessing": Profiler(),
                "visualization": Profiler(),
            }
            
            # 模型加载阶段性能分析
            with profilers["model_loading"]:
                self.progress_update.emit("加载模型...")
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
                self.progress_update.emit(f"模型加载完成: {model_type.value.upper()}")

            # 数据加载阶段性能分析
            with profilers["data_loading"]:
                self.progress_update.emit("加载测试数据...")
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
                
                self.progress_update.emit(f"数据集加载完成: {len(test_dataset)} 张图像")
                self.logger.info(f"Dataset created successfully. Total images: {len(test_dataset)}")
                self.logger.info(f"Batch size: {self.config['batch_size']}, Number of batches: {len(test_dataloader)}")

            # 模型预热阶段性能分析
            try:
                with profilers["warmup"]:
                    self.progress_update.emit("模型预热中...")
                    first = next(iter(test_dataloader))  # (batch, images, _, _)
                    first_batch = first[0]
                    if device_str == "cuda":
                        first_batch = first_batch.half()
                    first_batch = first_batch.to(device_str)
                    
                    model.warmup(batch=first_batch, runs=2)
                    self.logger.info(f"Warm-up done with first batch {tuple(first_batch.shape)}.")
                    self.progress_update.emit("模型预热完成")
            except StopIteration:
                self.progress_update.emit("数据集为空，跳过预热")
                self.logger.warning("Dataset empty; skipping warm-up.")
            except Exception as e:
                self.progress_update.emit(f"预热过程中出现错误: {e}")
                self.logger.warning(f"Warm-up skipped due to error: {e}")

            # 处理图像
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
                    self.progress_update.emit(f"处理批次 {batch_idx+1}/{len(test_dataloader)}...")
                    self.logger.debug(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")
                    
                    # 推理阶段性能分析
                    with profilers["inference"] as inference_prof:
                        # Run inference
                        if device_str == "cuda":
                            batch = batch.half()
                        batch = batch.to(device_str)
                        
                        image_scores, score_maps = model.predict(batch)
                        self.logger.debug(f"Image scores shape: {image_scores.shape}, Score maps shape: {score_maps.shape}")
                        self.logger.info(f"Batch shape: {batch.shape}, Inference completed in {inference_prof.elapsed_time * 1000:.2f} ms")
                    
                    # 后处理阶段性能分析
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
                    
                    # 可视化生成阶段性能分析
                    with profilers["visualization"]:
                        # Generate visualizations for this batch
                        batch_visualizations = self.generate_visualizations(
                            images, score_maps, image_classifications, self.config
                        )
                        all_visualizations.extend(batch_visualizations)

                # 发送性能分析结果
                performance_data = {
                    "model_loading": profilers["model_loading"].accumulated_time * 1000,
                    "data_loading": profilers["data_loading"].accumulated_time * 1000,
                    "warmup": profilers["warmup"].accumulated_time * 1000,
                    "inference": profilers["inference"].accumulated_time * 1000,
                    "postprocessing": profilers["postprocessing"].accumulated_time * 1000,
                    "visualization": profilers["visualization"].accumulated_time * 1000,
                    "total_images": len(all_images),
                    "total_batches": len(test_dataloader),
                }
                
                # 记录性能统计信息
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
                
                # 性能指标
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
                
                # 发送性能数据
                self.performance_update.emit(performance_data)
                
                self.inference_finished.emit(
                    all_images, all_image_scores, all_score_maps, 
                    all_classifications, all_visualizations
                )
                
            finally:
                model.close()
                
        except Exception as e:
            error_msg = f"推理过程中发生错误: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
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
                np.array(images),
                score_maps_np,
                alpha=config.get('viz_alpha', 0.5),  # 使用与CLI一致的alpha值
            )
            
            highlighted_images = anomavision.visualization.highlighted_images(
                [images[i] for i in range(len(images))],
                score_map_classifications,  # 使用像素级分类而不是图像级分类
                color=tuple(map(int, config.get('viz_color', "128,0,128").split(","))),
                alpha=0.5  # 与CLI保持一致
            )
            
            # Generate boundary images with frames based on image classifications
            boundary_images = anomavision.visualization.framed_boundary_images(
                np.array(images),
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
            logger.error(f"生成可视化结果时出错: {e}")
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
    

class ExportWorker(QThread):
    """Worker thread for model export"""
    progress_update = pyqtSignal(str)
    export_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path, output_dir, export_format, config):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.export_format = export_format
        self.config = config
        # 获取专门的导出日志记录器
        self.logger = get_logger("anomavision.export")

    def run(self):
        try:
            self.progress_update.emit("开始导出模型...")
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
            self.progress_update.emit(f"导出配置: device={self.config.get('device', 'cpu')}, layers={self.config.get('layer_indices', [0, 1])}")
            
            # Get input shape
            h, w = self.config['crop_size'] if self.config['crop_size'] is not None else self.config['resize']
            input_shape = [1, 3, h, w]
            self.logger.info(f"Input shape: {tuple(input_shape)}")
            self.progress_update.emit(f"输入形状: {tuple(input_shape)}")
            
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
                    self.progress_update.emit(f"ONNX模型已导出到: {output_path}")
                    self.logger.info(f"ONNX model exported to: {output_path}")
                    self.export_finished.emit(str(output_path))
                else:
                    raise RuntimeError("ONNX导出失败")
                    
            elif self.export_format == "torchscript":
                self.logger.info("Exporting to TorchScript format")
                output_path = exporter.export_torchscript(
                    input_shape=tuple(input_shape),
                    output_name=f"padim_model.torchscript",
                    optimize=self.config.get('optimize', False),
                )
                if output_path:
                    self.progress_update.emit(f"TorchScript模型已导出到: {output_path}")
                    self.logger.info(f"TorchScript model exported to: {output_path}")
                    self.export_finished.emit(str(output_path))
                else:
                    raise RuntimeError("TorchScript导出失败")
                    
            elif self.export_format == "openvino":
                self.logger.info("Exporting to OpenVINO format")
                output_path = exporter.export_openvino(
                    input_shape=tuple(input_shape),
                    output_name=f"padim_model_openvino",
                    fp16=self.config.get('fp16', True),
                    dynamic_batch=self.config.get('dynamic_batch', True),
                )
                if output_path:
                    self.progress_update.emit(f"OpenVINO模型已导出到: {output_path}")
                    self.logger.info(f"OpenVINO model exported to: {output_path}")
                    self.export_finished.emit(str(output_path))
                else:
                    raise RuntimeError("OpenVINO导出失败")
            
        except Exception as e:
            error_msg = f"导出过程中发生错误: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
            self.logger.exception("Fatal error during export")


class AnomaVisionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnomaVision - 异常检测工具 (PyQt5版本)")
        self.setGeometry(100, 100, 1200, 800)
        
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
            logger.warning(f"解析层索引时出错: {e}, 使用默认值 [0, 1]")
            return [0, 1]
        
    def load_default_config(self):
        """Load default configuration"""
        try:
            config = load_config("config.yml")
            # 确保默认的可视化参数与CLI一致
            if 'viz_alpha' not in config:
                config['viz_alpha'] = 0.5
            if 'viz_color' not in config:
                config['viz_color'] = "128,0,128"
            return config or {}
        except Exception as e:
            logger.warning(f"无法加载配置文件: {e}")
            # 返回默认配置，确保可视化参数与CLI一致
            return {
                'viz_alpha': 0.5,
                'viz_color': "128,0,128"
            }
    
    def setup_ui(self):
        """Setup the main UI"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_training_tab()
        self.create_inference_tab()
        self.create_export_tab()
        
        # Create log area
        self.create_log_area()
        
        # Set initial tab
        self.tab_widget.setCurrentIndex(0)
        
    def create_training_tab(self):
        """Create the training tab"""
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)
        
        # Training configuration group
        config_group = QGroupBox("训练配置")
        config_layout = QFormLayout(config_group)
        
        # Dataset path
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setText(self.config.get("dataset_path", ""))
        dataset_path_button = QPushButton("浏览")
        dataset_path_button.clicked.connect(self.browse_dataset_path)
        dataset_path_layout = QHBoxLayout()
        dataset_path_layout.addWidget(self.dataset_path_edit)
        dataset_path_layout.addWidget(dataset_path_button)
        config_layout.addRow("数据集路径:", dataset_path_layout)
        
        # Class name
        self.class_name_edit = QLineEdit()
        self.class_name_edit.setText(self.config.get("class_name", "E85"))
        config_layout.addRow("类别名称:", self.class_name_edit)
        
        # Backbone
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["resnet18", "wide_resnet50"])
        self.backbone_combo.setCurrentText(self.config.get("backbone", "resnet18"))
        config_layout.addRow("骨干网络:", self.backbone_combo)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(self.config.get("batch_size", 2))
        config_layout.addRow("批次大小:", self.batch_size_spin)
        
        # Feature dimension
        self.feat_dim_spin = QSpinBox()
        self.feat_dim_spin.setRange(1, 1000)
        self.feat_dim_spin.setValue(self.config.get("feat_dim", 50))
        config_layout.addRow("特征维度:", self.feat_dim_spin)
        
        # Layer indices
        self.layer_indices_edit = QLineEdit()
        layer_indices = self.config.get("layer_indices", [0, 1])
        # Ensure proper formatting for list
        if isinstance(layer_indices, list):
            layer_indices_str = str(layer_indices)
        else:
            layer_indices_str = str([0, 1])
        self.layer_indices_edit.setText(layer_indices_str)
        config_layout.addRow("层索引:", self.layer_indices_edit)
        
        # Resize
        self.resize_edit = QLineEdit()
        self.resize_edit.setText(str(self.config.get("resize", [224, 224])))
        config_layout.addRow("调整大小:", self.resize_edit)
        
        # Crop size
        self.crop_size_edit = QLineEdit()
        self.crop_size_edit.setText(str(self.config.get("crop_size", "")))
        config_layout.addRow("裁剪大小:", self.crop_size_edit)
        
        # Model output path
        self.model_output_path_edit = QLineEdit()
        self.model_output_path_edit.setText(self.config.get("model_data_path", "./distributions"))
        model_output_path_button = QPushButton("浏览")
        model_output_path_button.clicked.connect(self.browse_model_output_path)
        model_output_path_layout = QHBoxLayout()
        model_output_path_layout.addWidget(self.model_output_path_edit)
        model_output_path_layout.addWidget(model_output_path_button)
        config_layout.addRow("模型输出路径:", model_output_path_layout)
        
        # Output model name
        self.output_model_name_edit = QLineEdit()
        self.output_model_name_edit.setText(self.config.get("output_model", "padim_model.pt"))
        config_layout.addRow("输出模型名称:", self.output_model_name_edit)
        
        layout.addWidget(config_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        self.train_button = QPushButton("开始训练")
        self.train_button.clicked.connect(self.start_training)
        self.train_stop_button = QPushButton("停止训练")
        self.train_stop_button.clicked.connect(self.stop_training)
        self.train_stop_button.setEnabled(False)
        buttons_layout.addWidget(self.train_button)
        buttons_layout.addWidget(self.train_stop_button)
        layout.addLayout(buttons_layout)
        
        # Progress bar
        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setVisible(False)
        layout.addWidget(self.train_progress_bar)
        
        self.tab_widget.addTab(training_tab, "训练")
        
    def create_inference_tab(self):
        """Create the inference tab"""
        inference_tab = QWidget()
        layout = QVBoxLayout(inference_tab)
        
        # Model and data group
        model_data_group = QGroupBox("模型和数据")
        model_data_layout = QFormLayout(model_data_group)
        
        # Model path
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText(self.config.get("model", "./distributions/padim_model.pt"))
        model_path_button = QPushButton("浏览")
        model_path_button.clicked.connect(self.browse_model_path)
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_path_button)
        model_data_layout.addRow("模型路径:", model_path_layout)
        
        # Image path
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setText(self.config.get("img_path", ""))
        image_path_button = QPushButton("浏览")
        image_path_button.clicked.connect(self.browse_image_path)
        image_path_layout = QHBoxLayout()
        image_path_layout.addWidget(self.image_path_edit)
        image_path_layout.addWidget(image_path_button)
        model_data_layout.addRow("图像路径:", image_path_layout)
        
        layout.addWidget(model_data_group)
        
        # Inference parameters group
        params_group = QGroupBox("推理参数")
        params_layout = QFormLayout(params_group)
        
        # Threshold
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 100.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(self.config.get("thresh", 13.0))
        params_layout.addRow("阈值:", self.threshold_spin)
        
        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(self.config.get("device", "auto"))
        params_layout.addRow("设备:", self.device_combo)
        
        # Batch size
        self.infer_batch_size_spin = QSpinBox()
        self.infer_batch_size_spin.setRange(1, 128)
        self.infer_batch_size_spin.setValue(self.config.get("batch_size", 1))
        params_layout.addRow("批次大小:", self.infer_batch_size_spin)
        
        # Visualization padding
        self.viz_padding_spin = QSpinBox()
        self.viz_padding_spin.setRange(0, 100)
        self.viz_padding_spin.setValue(self.config.get("viz_padding", 30))
        params_layout.addRow("边界填充:", self.viz_padding_spin)
        
        layout.addWidget(params_group)
        
        # 性能统计显示区域
        self.performance_group = QGroupBox("性能统计")
        self.performance_layout = QFormLayout(self.performance_group)
        self.model_loading_time_label = QLabel("模型加载: 0.00 ms")
        self.data_loading_time_label = QLabel("数据加载: 0.00 ms")
        self.warmup_time_label = QLabel("模型预热: 0.00 ms")
        self.inference_time_label = QLabel("推理: 0.00 ms")
        self.postprocessing_time_label = QLabel("后处理: 0.00 ms")
        self.visualization_time_label = QLabel("可视化: 0.00 ms")
        
        self.performance_layout.addRow(self.model_loading_time_label)
        self.performance_layout.addRow(self.data_loading_time_label)
        self.performance_layout.addRow(self.warmup_time_label)
        self.performance_layout.addRow(self.inference_time_label)
        self.performance_layout.addRow(self.postprocessing_time_label)
        self.performance_layout.addRow(self.visualization_time_label)
        
        layout.addWidget(self.performance_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        self.infer_button = QPushButton("开始推理")
        self.infer_button.clicked.connect(self.start_inference)
        self.infer_stop_button = QPushButton("停止推理")
        self.infer_stop_button.clicked.connect(self.stop_inference)
        self.infer_stop_button.setEnabled(False)
        self.prev_result_button = QPushButton("上一个结果")
        self.prev_result_button.clicked.connect(self.show_prev_result)
        self.prev_result_button.setEnabled(False)
        self.next_result_button = QPushButton("下一个结果")
        self.next_result_button.clicked.connect(self.show_next_result)
        self.next_result_button.setEnabled(False)
        buttons_layout.addWidget(self.infer_button)
        buttons_layout.addWidget(self.infer_stop_button)
        buttons_layout.addWidget(self.prev_result_button)
        buttons_layout.addWidget(self.next_result_button)
        layout.addLayout(buttons_layout)
        
        # Results display
        results_group = QGroupBox("推理结果")
        results_layout = QVBoxLayout(results_group)
        
        # Results info
        self.results_info_label = QLabel("暂无推理结果")
        results_layout.addWidget(self.results_info_label)
        
        # Anomaly score display (single label at the top center)
        self.anomaly_score_label = QLabel("异常分数: 0.00")
        self.anomaly_score_label.setAlignment(Qt.AlignCenter)
        self.anomaly_score_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        results_layout.addWidget(self.anomaly_score_label)
        
        # Visualization area
        self.vis_splitter = QSplitter(Qt.Horizontal)
        
        # Original image
        self.original_image_label = QLabel("原始图像")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 300)
        self.original_image_label.setStyleSheet("border: 1px solid black;")
        
        # Heatmap
        self.heatmap_label = QLabel("热力图")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(300, 300)
        self.heatmap_label.setStyleSheet("border: 1px solid black;")
        
        # Highlighted
        self.highlighted_label = QLabel("高亮异常")
        self.highlighted_label.setAlignment(Qt.AlignCenter)
        self.highlighted_label.setMinimumSize(300, 300)
        self.highlighted_label.setStyleSheet("border: 1px solid black;")
        
        # Boundary
        self.boundary_label = QLabel("边界检测")
        self.boundary_label.setAlignment(Qt.AlignCenter)
        self.boundary_label.setMinimumSize(300, 300)
        self.boundary_label.setStyleSheet("border: 1px solid black;")
        
        self.vis_splitter.addWidget(self.original_image_label)
        self.vis_splitter.addWidget(self.heatmap_label)
        self.vis_splitter.addWidget(self.highlighted_label)
        self.vis_splitter.addWidget(self.boundary_label)
        
        results_layout.addWidget(self.vis_splitter)
        
        layout.addWidget(results_group)
        
        self.tab_widget.addTab(inference_tab, "推理")
        
    def create_export_tab(self):
        """Create the export tab"""
        export_tab = QWidget()
        layout = QVBoxLayout(export_tab)
        
        # Export configuration group
        config_group = QGroupBox("导出配置")
        config_layout = QFormLayout(config_group)
        
        # Model path
        self.export_model_path_edit = QLineEdit()
        self.export_model_path_edit.setText(self.config.get("model", "./distributions/padim_model.pt"))
        export_model_path_button = QPushButton("浏览")
        export_model_path_button.clicked.connect(self.browse_export_model_path)
        export_model_path_layout = QHBoxLayout()
        export_model_path_layout.addWidget(self.export_model_path_edit)
        export_model_path_layout.addWidget(export_model_path_button)
        config_layout.addRow("模型路径:", export_model_path_layout)
        
        # Output directory
        self.export_output_dir_edit = QLineEdit()
        self.export_output_dir_edit.setText(self.config.get("model_data_path", "./distributions/anomav_exp"))
        export_output_dir_button = QPushButton("浏览")
        export_output_dir_button.clicked.connect(self.browse_export_output_dir)
        export_output_dir_layout = QHBoxLayout()
        export_output_dir_layout.addWidget(self.export_output_dir_edit)
        export_output_dir_layout.addWidget(export_output_dir_button)
        config_layout.addRow("输出目录:", export_output_dir_layout)
        
        # Export format
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["onnx", "torchscript", "openvino", "all"])
        self.export_format_combo.setCurrentText("onnx")
        config_layout.addRow("导出格式:", self.export_format_combo)
        
        # ONNX specific options
        self.onnx_opset_spin = QSpinBox()
        self.onnx_opset_spin.setRange(1, 20)
        self.onnx_opset_spin.setValue(self.config.get("opset", 17))
        config_layout.addRow("ONNX Opset:", self.onnx_opset_spin)
        
        self.dynamic_batch_check = QCheckBox("动态批次")
        self.dynamic_batch_check.setChecked(self.config.get("dynamic_batch", True))
        config_layout.addRow("动态批次:", self.dynamic_batch_check)
        
        # Device selection
        self.export_device_combo = QComboBox()
        self.export_device_combo.addItems(["cpu", "cuda"])
        self.export_device_combo.setCurrentText(self.config.get("device", "cpu"))
        config_layout.addRow("导出设备:", self.export_device_combo)
        
        layout.addWidget(config_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        self.export_button = QPushButton("开始导出")
        self.export_button.clicked.connect(self.start_export)
        self.export_stop_button = QPushButton("停止导出")
        self.export_stop_button.clicked.connect(self.stop_export)
        self.export_stop_button.setEnabled(False)
        buttons_layout.addWidget(self.export_button)
        buttons_layout.addWidget(self.export_stop_button)
        layout.addLayout(buttons_layout)
        
        # Progress bar
        self.export_progress_bar = QProgressBar()
        self.export_progress_bar.setVisible(False)
        layout.addWidget(self.export_progress_bar)
        
        # Export info
        self.export_info_label = QLabel("导出完成后，文件将保存到指定目录")
        layout.addWidget(self.export_info_label)
        
        self.tab_widget.addTab(export_tab, "导出")
        
    def create_log_area(self):
        """Create the log area"""
        # Log group
        log_group = QGroupBox("日志输出")
        log_layout = QVBoxLayout(log_group)
        
        # Log text area
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setMaximumHeight(150)
        log_layout.addWidget(self.log_text_edit)
        
        # Clear log button
        clear_log_button = QPushButton("清空日志")
        clear_log_button.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_button)
        
        # Add to main layout
        self.centralWidget().layout().addWidget(log_group)
        
    def setup_logging(self):
        """Setup logging to display in GUI"""
        class GuiLogHandler(logging.Handler):
            def __init__(self, text_edit):
                super().__init__()
                self.text_edit = text_edit
                
            def emit(self, record):
                msg = self.format(record)
                self.text_edit.append(msg)
                
        # Add handler to anomavision logger
        gui_handler = GuiLogHandler(self.log_text_edit)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        anomavision_logger = logging.getLogger("anomavision")
        anomavision_logger.addHandler(gui_handler)
        anomavision_logger.setLevel(logging.INFO)
        
        # Also add to root logger for general messages
        root_logger = logging.getLogger()
        root_logger.addHandler(gui_handler)
        
    def browse_dataset_path(self):
        """Browse for dataset path"""
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.dataset_path_edit.setText(path)
            
    def browse_model_output_path(self):
        """Browse for model output path"""
        path = QFileDialog.getExistingDirectory(self, "选择模型输出目录")
        if path:
            self.model_output_path_edit.setText(path)
            
    def browse_model_path(self):
        """Browse for model path"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", 
            "模型文件 (*.pt *.pth *.onnx *.torchscript *.xml)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def browse_image_path(self):
        """Browse for image path"""
        path = QFileDialog.getExistingDirectory(self, "选择图像目录")
        if path:
            self.image_path_edit.setText(path)
            
    def browse_export_model_path(self):
        """Browse for export model path"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", 
            "模型文件 (*.pt *.pth *.onnx *.torchscript *.xml)"
        )
        if file_path:
            self.export_model_path_edit.setText(file_path)
            
    def browse_export_output_dir(self):
        """Browse for export output directory"""
        path = QFileDialog.getExistingDirectory(self, "选择导出输出目录")
        if path:
            self.export_output_dir_edit.setText(path)
            
    def start_training(self):
        """Start the training process"""
        try:
            # Get configuration from UI
            config = {
                "dataset_path": self.dataset_path_edit.text(),
                "class_name": self.class_name_edit.text(),
                "backbone": self.backbone_combo.currentText(),
                "batch_size": self.batch_size_spin.value(),
                "feat_dim": self.feat_dim_spin.value(),
                "layer_indices": self.parse_layer_indices(self.layer_indices_edit.text()),
                "resize": eval(self.resize_edit.text()),
                "crop_size": eval(self.crop_size_edit.text()) if self.crop_size_edit.text() else None,
                "normalize": self.config.get("normalize", True),
                "norm_mean": self.config.get("norm_mean", [0.485, 0.456, 0.406]),
                "norm_std": self.config.get("norm_std", [0.229, 0.224, 0.225]),
                "model_data_path": self.model_output_path_edit.text(),
                "output_model": self.output_model_name_edit.text(),
                "run_name": self.config.get("run_name", "anomav_exp"),
                "log_level": self.config.get("log_level", "INFO"),
            }
            
            # Validate required fields
            if not config["dataset_path"]:
                QMessageBox.warning(self, "警告", "请选择数据集路径")
                return
                
            if not config["class_name"]:
                QMessageBox.warning(self, "警告", "请输入类别名称")
                return
                
            # Disable UI elements
            self.train_button.setEnabled(False)
            self.train_stop_button.setEnabled(True)
            self.train_progress_bar.setVisible(True)
            self.train_progress_bar.setRange(0, 0)  # Indeterminate progress
            
            # Start training worker
            self.training_worker = TrainingWorker(config)
            self.training_worker.progress_update.connect(self.update_log)
            self.training_worker.training_finished.connect(self.training_finished)
            self.training_worker.error_occurred.connect(self.training_error)
            self.training_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动训练时发生错误:\n{str(e)}")
            self.train_button.setEnabled(True)
            self.train_stop_button.setEnabled(False)
            self.train_progress_bar.setVisible(False)
            
    def stop_training(self):
        """Stop the training process"""
        if hasattr(self, 'training_worker') and self.training_worker.isRunning():
            self.training_worker.terminate()
            self.training_worker.wait()
            self.train_button.setEnabled(True)
            self.train_stop_button.setEnabled(False)
            self.train_progress_bar.setVisible(False)
            self.update_log("训练已停止")
            
    def training_finished(self, model):
        """Handle training finished"""
        self.current_model = model
        self.train_button.setEnabled(True)
        self.train_stop_button.setEnabled(False)
        self.train_progress_bar.setVisible(False)
        self.update_log("训练完成!")
        QMessageBox.information(self, "完成", "模型训练完成!")
        
    def training_error(self, error_msg):
        """Handle training error"""
        self.train_button.setEnabled(True)
        self.train_stop_button.setEnabled(False)
        self.train_progress_bar.setVisible(False)
        self.update_log(f"训练错误: {error_msg}")
        QMessageBox.critical(self, "训练错误", error_msg)
        
    def start_inference(self):
        """Start the inference process"""
        try:
            # Get configuration from UI
            config = {
                "resize": eval(self.resize_edit.text()) if self.resize_edit.text() else [224, 224],
                "crop_size": eval(self.crop_size_edit.text()) if self.crop_size_edit.text() else None,
                "normalize": self.config.get("normalize", True),
                "norm_mean": self.config.get("norm_mean", [0.485, 0.456, 0.406]),
                "norm_std": self.config.get("norm_std", [0.229, 0.224, 0.225]),
                "batch_size": self.infer_batch_size_spin.value(),
                "thresh": self.threshold_spin.value(),
                "device": self.device_combo.currentText(),
                "num_workers": self.config.get("num_workers", 1),
                "pin_memory": self.config.get("pin_memory", False),
                "viz_alpha": self.config.get("viz_alpha", 0.5),  # 确保与CLI一致
                "viz_color": self.config.get("viz_color", "128,0,128"),  # 确保与CLI一致
                "viz_padding": self.viz_padding_spin.value(),  # 边界填充
                "layer_indices": self.parse_layer_indices(self.config.get("layer_indices", [0, 1])),
            }
            
            model_path = self.model_path_edit.text()
            img_path = self.image_path_edit.text()
            
            # Validate required fields
            if not model_path:
                QMessageBox.warning(self, "警告", "请选择模型文件")
                return
                
            if not img_path:
                QMessageBox.warning(self, "警告", "请选择图像目录")
                return
                
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "警告", "模型文件不存在")
                return
                
            if not os.path.exists(img_path):
                QMessageBox.warning(self, "警告", "图像目录不存在")
                return
                
            # Disable UI elements
            self.infer_button.setEnabled(False)
            self.infer_stop_button.setEnabled(True)
            self.prev_result_button.setEnabled(False)
            self.next_result_button.setEnabled(False)
            self.results_info_label.setText("正在处理...")
            
            # 重置性能显示
            self.model_loading_time_label.setText("模型加载: 0.00 ms")
            self.data_loading_time_label.setText("数据加载: 0.00 ms")
            self.warmup_time_label.setText("模型预热: 0.00 ms")
            self.inference_time_label.setText("推理: 0.00 ms")
            self.postprocessing_time_label.setText("后处理: 0.00 ms")
            self.visualization_time_label.setText("可视化: 0.00 ms")
            
            # Start inference worker
            self.inference_worker = InferenceWorker(model_path, img_path, config)
            self.inference_worker.progress_update.connect(self.update_log)
            self.inference_worker.inference_finished.connect(self.inference_finished)
            self.inference_worker.performance_update.connect(self.update_performance_display)  # 连接性能更新信号
            self.inference_worker.error_occurred.connect(self.inference_error)
            self.inference_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动推理时发生错误:\n{str(e)}")
            self.infer_button.setEnabled(True)
            self.infer_stop_button.setEnabled(False)

    def update_performance_display(self, performance_data):
        """更新性能显示"""
        self.model_loading_time_label.setText(f"模型加载: {performance_data['model_loading']:.2f} ms")
        self.data_loading_time_label.setText(f"数据加载: {performance_data['data_loading']:.2f} ms")
        self.warmup_time_label.setText(f"模型预热: {performance_data['warmup']:.2f} ms")
        self.inference_time_label.setText(f"推理: {performance_data['inference']:.2f} ms")
        self.postprocessing_time_label.setText(f"后处理: {performance_data['postprocessing']:.2f} ms")
        self.visualization_time_label.setText(f"可视化: {performance_data['visualization']:.2f} ms")

    def stop_inference(self):
        """Stop the inference process"""
        if hasattr(self, 'inference_worker') and self.inference_worker.isRunning():
            self.inference_worker.terminate()
            self.inference_worker.wait()
            self.infer_button.setEnabled(True)
            self.infer_stop_button.setEnabled(False)
            self.results_info_label.setText("推理已停止")
            self.update_log("推理已停止")
            
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
        self.infer_button.setEnabled(True)
        self.infer_stop_button.setEnabled(False)
        
        if visualizations:
            self.prev_result_button.setEnabled(len(visualizations) > 1)
            self.next_result_button.setEnabled(len(visualizations) > 1)
            self.show_result(0)
            self.results_info_label.setText(f"显示结果: 1/{len(visualizations)}")
        else:
            self.results_info_label.setText("无可视化结果")
            
        self.update_log(f"推理完成! 处理了 {len(images)} 张图像")
        QMessageBox.information(self, "完成", f"推理完成! 处理了 {len(images)} 张图像")

    def inference_error(self, error_msg):
        """Handle inference error"""
        self.infer_button.setEnabled(True)
        self.infer_stop_button.setEnabled(False)
        self.results_info_label.setText("推理失败")
        self.update_log(f"推理错误: {error_msg}")
        QMessageBox.critical(self, "推理错误", error_msg)
        
    def show_prev_result(self):
        """Show previous result"""
        if self.inference_results and self.inference_results['visualizations']:
            self.current_result_index = (self.current_result_index - 1) % len(self.inference_results['visualizations'])
            self.show_result(self.current_result_index)
            self.results_info_label.setText(f"显示结果: {self.current_result_index+1}/{len(self.inference_results['visualizations'])}")
            
    def show_next_result(self):
        """Show next result"""
        if self.inference_results and self.inference_results['visualizations']:
            self.current_result_index = (self.current_result_index + 1) % len(self.inference_results['visualizations'])
            self.show_result(self.current_result_index)
            self.results_info_label.setText(f"显示结果: {self.current_result_index+1}/{len(self.inference_results['visualizations'])}")
            
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
        self.anomaly_score_label.setText(f"异常分数: {score:.2f}")
        
        # Get classification result
        # Note: 0 means anomaly present, 1 means normal
        is_anomaly = vis_data.get('classification', 0) == 0
        
        # Set border style based on classification
        border_style = "border: 3px solid red;" if is_anomaly else "border: 3px solid green;"
        
        # Display original image with border
        self.display_image(self.original_image_label, vis_data['original'])
        self.original_image_label.setStyleSheet(border_style)
        
        # Display heatmap with border
        if 'heatmap' in vis_data:
            self.display_image(self.heatmap_label, vis_data['heatmap'])
            self.heatmap_label.setStyleSheet(border_style)
            
        # Display highlighted image with border
        if 'highlighted' in vis_data:
            self.display_image(self.highlighted_label, vis_data['highlighted'])
            self.highlighted_label.setStyleSheet(border_style)
            
        # Display boundary image with border
        if 'boundary' in vis_data:
            self.display_image(self.boundary_label, vis_data['boundary'])
            self.boundary_label.setStyleSheet(border_style)
            
    def display_image(self, label, image):
        """Display an image in a QLabel"""
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
                
                # Convert to QImage
                height, width = image.shape[:2]
                bytes_per_line = width * 3
                q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                # Assume it's already a QImage or QPixmap
                q_img = image
            
            # Scale to label size
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap.scaled(
                label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        except Exception as e:
            logger.error(f"显示图像时出错: {e}")
            label.setText("无法显示图像")
            
    def start_export(self):
        """Start the export process"""
        try:
            model_path = self.export_model_path_edit.text()
            output_dir = self.export_output_dir_edit.text()
            export_format = self.export_format_combo.currentText()
            
            # Validate required fields
            if not model_path:
                QMessageBox.warning(self, "警告", "请选择模型文件")
                return
                
            if not output_dir:
                QMessageBox.warning(self, "警告", "请选择输出目录")
                return
                
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "警告", "模型文件不存在")
                return
                
            if not os.path.exists(output_dir):
                QMessageBox.warning(self, "警告", "输出目录不存在")
                return
                
            # Get configuration from UI
            config = {
                "resize": eval(self.resize_edit.text()) if self.resize_edit.text() else [224, 224],
                "crop_size": eval(self.crop_size_edit.text()) if self.crop_size_edit.text() else None,
                "normalize": self.config.get("normalize", True),
                "norm_mean": self.config.get("norm_mean", [0.485, 0.456, 0.406]),
                "norm_std": self.config.get("norm_std", [0.229, 0.224, 0.225]),
                "opset": self.onnx_opset_spin.value(),
                "dynamic_batch": self.dynamic_batch_check.isChecked(),
                "device": self.export_device_combo.currentText(),
                "fp16": self.config.get("fp16", True),
                "optimize": self.config.get("optimize", False),
                "layer_indices": self.parse_layer_indices(self.config.get("layer_indices", [0, 1])),
            }
            
            # Disable UI elements
            self.export_button.setEnabled(False)
            self.export_stop_button.setEnabled(True)
            self.export_progress_bar.setVisible(True)
            self.export_progress_bar.setRange(0, 0)  # Indeterminate progress
            
            # Start export worker
            self.export_worker = ExportWorker(model_path, output_dir, export_format, config)
            self.export_worker.progress_update.connect(self.update_log)
            self.export_worker.export_finished.connect(self.export_finished)
            self.export_worker.error_occurred.connect(self.export_error)
            self.export_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动导出时发生错误:\n{str(e)}")
            self.export_button.setEnabled(True)
            self.export_stop_button.setEnabled(False)
            self.export_progress_bar.setVisible(False)
            
    def stop_export(self):
        """Stop the export process"""
        if hasattr(self, 'export_worker') and self.export_worker.isRunning():
            self.export_worker.terminate()
            self.export_worker.wait()
            self.export_button.setEnabled(True)
            self.export_stop_button.setEnabled(False)
            self.export_progress_bar.setVisible(False)
            self.update_log("导出已停止")
            
    def export_finished(self, output_path):
        """Handle export finished"""
        self.export_button.setEnabled(True)
        self.export_stop_button.setEnabled(False)
        self.export_progress_bar.setVisible(False)
        self.export_info_label.setText(f"导出完成! 文件保存到: {output_path}")
        self.update_log(f"导出完成! 文件保存到: {output_path}")
        QMessageBox.information(self, "完成", f"模型导出完成!\n文件保存到: {output_path}")
        
    def export_error(self, error_msg):
        """Handle export error"""
        self.export_button.setEnabled(True)
        self.export_stop_button.setEnabled(False)
        self.export_progress_bar.setVisible(False)
        self.export_info_label.setText("导出失败")
        self.update_log(f"导出错误: {error_msg}")
        QMessageBox.critical(self, "导出错误", error_msg)
        
    def update_log(self, message):
        """Update the log display"""
        self.log_text_edit.append(message)
        
    def clear_log(self):
        """Clear the log display"""
        self.log_text_edit.clear()


def main():
    app = QApplication(sys.argv)
    gui = AnomaVisionGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
