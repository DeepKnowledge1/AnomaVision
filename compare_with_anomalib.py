"""
Comprehensive comparison between your anomaly detection implementation and Anomalib.
This script benchmarks both implementations on model size, speed, performance, and memory usage.

Requirements:
    pip install anomalib torch torchvision numpy pandas matplotlib seaborn tabulate psutil

Usage:
    python compare_with_anomalib.py --dataset_path /path/to/mvtec --class_name bottle
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ===============================
# Data Classes for Results
# ===============================


@dataclass
class ModelMetrics:
    """Store metrics for a single model."""

    name: str
    # Performance metrics
    image_auroc: float = 0.0
    pixel_auroc: float = 0.0

    # Speed metrics
    training_time: float = 0.0
    inference_fps: float = 0.0
    inference_time_per_image: float = 0.0

    # Model size metrics
    model_size_mb: float = 0.0
    export_size_mb: Dict[str, float] = None

    # Memory metrics
    peak_memory_mb: float = 0.0
    training_memory_mb: float = 0.0
    inference_memory_mb: float = 0.0

    # Additional info
    backbone: str = ""
    device: str = ""

    def __post_init__(self):
        if self.export_size_mb is None:
            self.export_size_mb = {}


def evaluate_model_with_wrapper(model, test_dataloader):
    """
    Evaluate AnomaVision model using the ModelWrapper inference interface
    Returns: (images, image_classifications_target, masks_target, image_scores, score_maps)
    """
    from anomavision.general import determine_device

    all_images = []
    all_image_classifications_target = []
    all_masks_target = []
    all_image_scores = []
    all_score_maps = []

    batch_count = 0
    device_str = determine_device("cpu")
    try:
        for batch_idx, (batch, images, image_targets, mask_targets) in enumerate(
            test_dataloader
        ):
            batch = batch.to(device_str)

            image_scores, score_maps = model.predict(batch)

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
        raise RuntimeError(f"Error during AnomaVision evaluation: {e}")

    # Convert lists back to appropriate formats for AnomaVision analysis
    all_images = np.array(all_images)
    all_image_classifications_target = np.array(all_image_classifications_target)
    all_masks_target = np.squeeze(np.array(all_masks_target), axis=1)
    all_image_scores = np.array(all_image_scores)
    all_score_maps = np.array(all_score_maps)
    return (
        all_images,
        all_image_classifications_target,
        all_masks_target,
        all_image_scores,
        all_score_maps,
    )


class BenchmarkRunner:
    """Main benchmark runner class."""

    def __init__(self, dataset_path: str, class_name: str, device: str = "auto"):
        self.dataset_path = Path(dataset_path)
        self.class_name = class_name
        self.device = self._setup_device(device)
        self.results = {}

        # Setup paths
        self.output_dir = Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

        # Model save paths
        self.your_model_path = self.output_dir / "your_model"
        self.anomalib_model_path = self.output_dir / "anomalib_model"

        self.your_model_path.mkdir(exist_ok=True)
        self.anomalib_model_path.mkdir(exist_ok=True)

    def _setup_device(self, device: str) -> torch.device:
        """Setup and return the appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024

    def _reset_memory(self):
        """Reset memory tracking."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # ===============================
    # Your Implementation
    # ===============================

    def benchmark_your_implementation(self) -> ModelMetrics:
        """Benchmark your PaDiM implementation."""
        print("\n" + "=" * 60)
        print("BENCHMARKING YOUR IMPLEMENTATION")
        print("=" * 60)

        try:

            from sklearn.metrics import roc_auc_score

            # Import your modules
            import anomavision
            from anomavision import MVTecDataset, Padim
            from anomavision.general import determine_device
            from anomavision.inference.model.wrapper import ModelWrapper
            from anomavision.utils import adaptive_gaussian_blur

            metrics = ModelMetrics(name="Your PaDiM")
            metrics.backbone = "resnet18"
            metrics.device = str(self.device)

            # === 1. Setup Datasets ===
            print("\n1. Loading datasets...")
            train_dataset = MVTecDataset(
                self.dataset_path,
                self.class_name,
                is_train=True,
                resize=224,
                normalize=True,
            )

            test_dataset = MVTecDataset(
                self.dataset_path,
                self.class_name,
                is_train=False,
                resize=224,
                normalize=True,
            )

            batch_size = 8
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            print(f"   Train samples: {len(train_dataset)}")
            print(f"   Test samples: {len(test_dataset)}")
            print(f"   Batch size: {batch_size}")

            # === 2. Training ===
            print("\n2. Training model...")
            self._reset_memory()

            model = Padim(
                backbone="resnet18", device=self.device, feat_dim=50, layer_indices=[0]
            )

            # Measure training
            start_time = time.time()
            start_memory = self._get_memory_usage()

            model.fit(train_loader)

            metrics.training_time = time.time() - start_time
            metrics.training_memory_mb = self._get_memory_usage() - start_memory
            metrics.peak_memory_mb = self._get_memory_usage()

            print(f"   Training time: {metrics.training_time:.2f}s")
            print(f"   Training memory: {metrics.training_memory_mb:.2f} MB")

            # === 3. Save Model ===
            print("\n3. Saving model...")
            model_path = self.your_model_path / "padim_model.pt"
            torch.save(model, model_path)

            stats_path = model_path.with_suffix(".pth")
            try:
                model.save_statistics(str(stats_path))

            except Exception as e:
                raise RuntimeError(f"Error saving statistics: {e}")

            metrics.model_size_mb = stats_path.stat().st_size / (1024 * 1024)
            print(f"   Model size: {metrics.model_size_mb:.2f} MB")

            # Try to save statistics version
            try:
                stats_path = self.your_model_path / "padim_model.pth"
                model.save_statistics(str(stats_path))
                stats_size = stats_path.stat().st_size / (1024 * 1024)
                print(f"   Statistics file size: {stats_size:.2f} MB")
            except:
                pass

            # === 4. Export Sizes ===
            print("\n4. Testing export formats...")

            # ONNX export
            try:
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                onnx_path = self.your_model_path / "model.onnx"
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    input_names=["input"],
                    output_names=["scores", "maps"],
                    dynamic_axes={"input": {0: "batch"}},
                )
                metrics.export_size_mb["onnx"] = onnx_path.stat().st_size / (
                    1024 * 1024
                )
                print(f"   ONNX size: {metrics.export_size_mb['onnx']:.2f} MB")
            except Exception as e:
                print(f"   ONNX export failed: {e}")

            # TorchScript export
            try:
                traced = torch.jit.trace(model, dummy_input)
                ts_path = self.your_model_path / "model.torchscript"
                traced.save(str(ts_path))
                metrics.export_size_mb["torchscript"] = ts_path.stat().st_size / (
                    1024 * 1024
                )
                print(
                    f"   TorchScript size: {metrics.export_size_mb['torchscript']:.2f} MB"
                )
            except Exception as e:
                print(f"   TorchScript export failed: {e}")

            # === 5. Inference Speed ===
            print("\n5. Measuring inference speed...")
            device_str = determine_device(self.device.type)

            self._reset_memory()

            model = ModelWrapper(model_path, device_str)
            # model.eval()
            inference_times = []
            start_memory = self._get_memory_usage()

            batch = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = model.warmup(batch)

                # Actual timing
                for _ in range(100):

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    _ = model.predict(batch)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    inference_times.append(time.perf_counter() - start)

            metrics.inference_memory_mb = self._get_memory_usage() - start_memory
            metrics.inference_time_per_image = np.mean(inference_times) * 1000  # ms
            metrics.inference_fps = 1.0 / np.mean(inference_times)

            print(f"   Inference time: {metrics.inference_time_per_image:.2f} ms/image")
            print(f"   Inference FPS: {metrics.inference_fps:.2f}")
            print(f"   Inference memory: {metrics.inference_memory_mb:.2f} MB")

            # === 6. Performance Evaluation ===
            print("\n6. Evaluating performance...")

            # Run evaluation
            images, y_true, masks_true, y_score, masks_score = (
                evaluate_model_with_wrapper(model, test_loader)
            )
            # images, image_classifications_target, masks_target, image_scores, score_maps = evaluate_model_with_wrapper(model, test_loader)

            masks_score = adaptive_gaussian_blur(masks_score, kernel_size=33, sigma=4)

            # anodet.visualize_eval_data(
            #     y_true,
            #     masks_true.astype(np.uint8).flatten(),
            #     y_score,
            #     masks_score.flatten(),
            # )

            # Calculate metrics
            metrics.image_auroc = roc_auc_score(y_true, y_score)

            # For pixel-level, handle the flattened masks
            mask_labels = masks_true.flatten().astype(np.uint8)
            mask_preds = masks_score.flatten()

            # Only calculate if we have anomalous pixels
            if np.sum(mask_labels) > 0:
                metrics.pixel_auroc = roc_auc_score(mask_labels, mask_preds)

            print(f"   Image AUROC: {metrics.image_auroc:.4f}")
            print(f"   Pixel AUROC: {metrics.pixel_auroc:.4f}")

            return metrics

        except Exception as e:
            print(f"\nError in your implementation benchmark: {e}")
            import traceback

            traceback.print_exc()
            return ModelMetrics(name="Your PaDiM (Failed)")

    # ===============================
    # Anomalib Implementation
    # ===============================

    def benchmark_anomalib(self) -> ModelMetrics:
        """Benchmark Anomalib's PaDiM implementation."""
        print("\n" + "=" * 60)
        print("BENCHMARKING ANOMALIB")
        print("=" * 60)

        try:
            # Import Anomalib modules
            from anomalib.data import MVTec
            from anomalib.engine import Engine
            from anomalib.metrics import AUROC
            from anomalib.models import Padim as AnomalibPadim
            from pytorch_lightning import Trainer
            from pytorch_lightning.callbacks import ModelCheckpoint

            metrics = ModelMetrics(name="Anomalib PaDiM")
            metrics.backbone = "resnet18"
            metrics.device = str(self.device)
            batch_size = 8
            # === 1. Setup Data Module ===
            print("\n1. Loading datasets...")
            datamodule = MVTec(
                root=str(self.dataset_path),
                category=self.class_name,
                # image_size=(224, 224),
                train_batch_size=batch_size,
                eval_batch_size=batch_size,
                num_workers=0,
            )

            datamodule.setup()

            print(f"   Train samples: {len(datamodule.train_dataloader().dataset)}")
            print(f"   Test samples: {len(datamodule.test_dataloader().dataset)}")
            print(f"   Batch size: {batch_size}")

            # === 2. Training ===
            print("\n2. Training model...")
            self._reset_memory()

            model = AnomalibPadim(
                backbone="resnet18",
                layers=["layer1"],  # , "layer2", "layer3"],
                pre_trained=True,
                n_features=50,
            )

            # Setup trainer
            # trainer = Trainer(
            #     max_epochs=1,
            #     accelerator="gpu" if self.device.type == "cuda" else "cpu",
            #     devices=1,
            #     logger=False,
            #     enable_checkpointing=False,
            #     enable_progress_bar=False
            # )

            trainer = Engine(
                max_epochs=1, logger=False, enable_progress_bar=False, callbacks=[]
            )

            # Measure training
            start_time = time.time()
            start_memory = self._get_memory_usage()

            trainer.fit(model, datamodule=datamodule)

            metrics.training_time = time.time() - start_time
            metrics.training_memory_mb = self._get_memory_usage() - start_memory
            metrics.peak_memory_mb = self._get_memory_usage()

            print(f"   Training time: {metrics.training_time:.2f}s")
            print(f"   Training memory: {metrics.training_memory_mb:.2f} MB")

            # === 3. Save Model ===
            print("\n3. Saving model...")
            model_path = self.anomalib_model_path / "model.ckpt"
            # trainer.save_checkpoint(model_path)
            trainer.trainer.save_checkpoint(str(model_path))

            metrics.model_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"   Model size: {metrics.model_size_mb:.2f} MB")

            # === 4. Export Sizes ===
            print("\n4. Testing export formats...")

            from anomalib.deploy import ExportType

            # ONNX export
            try:
                onnx_path = trainer.export(
                    model,
                    ExportType.ONNX,
                    export_root=self.anomalib_model_path,
                    input_size=(224, 224),
                    # ckpt_path=ckpt_path,   # use the checkpoint from above
                )
                metrics.export_size_mb["onnx"] = os.path.getsize(onnx_path) / (
                    1024 * 1024
                )

                print(f"   ONNX size: {metrics.export_size_mb['onnx']:.2f} MB")
            except Exception as e:
                print(f"   ONNX export failed: {e}")

            # === 5. Inference Speed ===
            print("\n5. Measuring inference speed...")
            self._reset_memory()

            model.eval()
            inference_times = []
            start_memory = self._get_memory_usage()

            batch = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = model(batch)

                # Actual timing
                for _ in range(100):

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    _ = model(batch)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    inference_times.append(time.perf_counter() - start)

            metrics.inference_memory_mb = self._get_memory_usage() - start_memory
            metrics.inference_time_per_image = np.mean(inference_times) * 1000  # ms
            metrics.inference_fps = 1.0 / np.mean(inference_times)

            print(f"   Inference time: {metrics.inference_time_per_image:.2f} ms/image")
            print(f"   Inference FPS: {metrics.inference_fps:.2f}")
            print(f"   Inference memory: {metrics.inference_memory_mb:.2f} MB")

            # === 6. Performance Evaluation ===
            print("\n6. Evaluating performance...")

            # Test the model
            test_results = trainer.test(model, datamodule=datamodule)

            # Extract metrics
            if test_results and len(test_results) > 0:
                metrics.image_auroc = test_results[0].get("image_AUROC", 0.0)
                metrics.pixel_auroc = test_results[0].get("pixel_AUROC", 0.0)

            print(f"   Image AUROC: {metrics.image_auroc:.4f}")
            print(f"   Pixel AUROC: {metrics.pixel_auroc:.4f}")

            return metrics

        except Exception as e:
            print(f"\nError in Anomalib benchmark: {e}")
            import traceback

            traceback.print_exc()
            return ModelMetrics(name="Anomalib PaDiM (Failed)")

    # ===============================
    # Comparison and Visualization
    # ===============================

    def run_comparison(self):
        """Run full comparison between implementations."""
        print("\n" + "=" * 60)
        print("STARTING COMPREHENSIVE COMPARISON")
        print("=" * 60)
        print(f"Dataset: {self.dataset_path}")
        print(f"Class: {self.class_name}")
        print(f"Device: {self.device}")

        # Run benchmarks
        your_metrics = self.benchmark_your_implementation()
        anomalib_metrics = self.benchmark_anomalib()

        # Store results
        self.results = {
            "your_implementation": your_metrics,
            "anomalib": anomalib_metrics,
        }

        # Generate comparison report
        self.generate_report()
        self.generate_visualizations()

        return self.results

    def generate_report(self):
        """Generate detailed comparison report."""
        print("\n" + "=" * 60)
        print("COMPARISON REPORT")
        print("=" * 60)

        your = self.results["your_implementation"]
        anomalib = self.results["anomalib"]

        # Create comparison table
        comparison_data = [
            ["Metric", "Your Implementation", "Anomalib", "Difference (%)"],
            ["=" * 20, "=" * 20, "=" * 20, "=" * 20],
            # Performance
            ["**PERFORMANCE**", "", "", ""],
            [
                "Image AUROC",
                f"{your.image_auroc:.4f}",
                f"{anomalib.image_auroc:.4f}",
                (
                    f"{((your.image_auroc - anomalib.image_auroc) / anomalib.image_auroc * 100):.2f}%"
                    if anomalib.image_auroc > 0
                    else "N/A"
                ),
            ],
            [
                "Pixel AUROC",
                f"{your.pixel_auroc:.4f}",
                f"{anomalib.pixel_auroc:.4f}",
                (
                    f"{((your.pixel_auroc - anomalib.pixel_auroc) / anomalib.pixel_auroc * 100):.2f}%"
                    if anomalib.pixel_auroc > 0
                    else "N/A"
                ),
            ],
            # Speed
            ["**SPEED**", "", "", ""],
            [
                "Training Time (s)",
                f"{your.training_time:.2f}",
                f"{anomalib.training_time:.2f}",
                (
                    f"{((your.training_time - anomalib.training_time) / anomalib.training_time * 100):.2f}%"
                    if anomalib.training_time > 0
                    else "N/A"
                ),
            ],
            [
                "Inference FPS",
                f"{your.inference_fps:.2f}",
                f"{anomalib.inference_fps:.2f}",
                (
                    f"{((your.inference_fps - anomalib.inference_fps) / anomalib.inference_fps * 100):.2f}%"
                    if anomalib.inference_fps > 0
                    else "N/A"
                ),
            ],
            [
                "ms/image",
                f"{your.inference_time_per_image:.2f}",
                f"{anomalib.inference_time_per_image:.2f}",
                (
                    f"{((your.inference_time_per_image - anomalib.inference_time_per_image) / anomalib.inference_time_per_image * 100):.2f}%"
                    if anomalib.inference_time_per_image > 0
                    else "N/A"
                ),
            ],
            # Size
            ["**MODEL SIZE**", "", "", ""],
            [
                "PyTorch (MB)",
                f"{your.model_size_mb:.2f}",
                f"{anomalib.model_size_mb:.2f}",
                (
                    f"{((your.model_size_mb - anomalib.model_size_mb) / anomalib.model_size_mb * 100):.2f}%"
                    if anomalib.model_size_mb > 0
                    else "N/A"
                ),
            ],
        ]

        # Add export sizes if available
        if your.export_size_mb.get("onnx") and anomalib.export_size_mb.get("onnx"):
            comparison_data.append(
                [
                    "ONNX (MB)",
                    f"{your.export_size_mb['onnx']:.2f}",
                    f"{anomalib.export_size_mb['onnx']:.2f}",
                    f"{((your.export_size_mb['onnx'] - anomalib.export_size_mb['onnx']) / anomalib.export_size_mb['onnx'] * 100):.2f}%",
                ]
            )

        # Memory
        comparison_data.extend(
            [
                ["**MEMORY**", "", "", ""],
                [
                    "Training Mem (MB)",
                    f"{your.training_memory_mb:.2f}",
                    f"{anomalib.training_memory_mb:.2f}",
                    (
                        f"{((your.training_memory_mb - anomalib.training_memory_mb) / anomalib.training_memory_mb * 100):.2f}%"
                        if anomalib.training_memory_mb > 0
                        else "N/A"
                    ),
                ],
                [
                    "Inference Mem (MB)",
                    f"{your.inference_memory_mb:.2f}",
                    f"{anomalib.inference_memory_mb:.2f}",
                    (
                        f"{((your.inference_memory_mb - anomalib.inference_memory_mb) / anomalib.inference_memory_mb * 100):.2f}%"
                        if anomalib.inference_memory_mb > 0
                        else "N/A"
                    ),
                ],
                [
                    "Peak Memory (MB)",
                    f"{your.peak_memory_mb:.2f}",
                    f"{anomalib.peak_memory_mb:.2f}",
                    (
                        f"{((your.peak_memory_mb - anomalib.peak_memory_mb) / anomalib.peak_memory_mb * 100):.2f}%"
                        if anomalib.peak_memory_mb > 0
                        else "N/A"
                    ),
                ],
            ]
        )

        # Print table
        print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))

        # Save to file
        report_path = self.output_dir / f"comparison_report_{self.class_name}.txt"
        with open(report_path, "w") as f:
            f.write(f"Comparison Report - {self.class_name}\n")
            f.write(f"Device: {self.device}\n\n")
            f.write(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))

        print(f"\nReport saved to: {report_path}")

        # Also save as JSON for programmatic access
        json_path = self.output_dir / f"comparison_results_{self.class_name}.json"
        results_dict = {"your_implementation": vars(your), "anomalib": vars(anomalib)}
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"JSON results saved to: {json_path}")

    def generate_visualizations(self):
        """Generate comparison visualizations."""
        print("\n Generating visualizations...")

        your = self.results["your_implementation"]
        anomalib = self.results["anomalib"]

        # Set style
        sns.set_style("whitegrid")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Comparison: Your Implementation vs Anomalib - {self.class_name}",
            fontsize=16,
        )

        # 1. Performance Comparison
        ax = axes[0, 0]
        categories = ["Image AUROC", "Pixel AUROC"]
        your_scores = [your.image_auroc, your.pixel_auroc]
        anomalib_scores = [anomalib.image_auroc, anomalib.pixel_auroc]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(
            x - width / 2,
            your_scores,
            width,
            label="Your Implementation",
            color="steelblue",
        )
        ax.bar(x + width / 2, anomalib_scores, width, label="Anomalib", color="coral")
        ax.set_xlabel("Metric")
        ax.set_ylabel("AUROC Score")
        ax.set_title("Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim([0, 1.1])

        # Add value labels on bars
        for i, (y_val, a_val) in enumerate(zip(your_scores, anomalib_scores)):
            ax.text(
                i - width / 2, y_val + 0.01, f"{y_val:.3f}", ha="center", va="bottom"
            )
            ax.text(
                i + width / 2, a_val + 0.01, f"{a_val:.3f}", ha="center", va="bottom"
            )

        # 2. Speed Comparison
        ax = axes[0, 1]
        categories = ["Training (s)", "Inference (ms)"]
        your_times = [your.training_time, your.inference_time_per_image]
        anomalib_times = [anomalib.training_time, anomalib.inference_time_per_image]

        x = np.arange(len(categories))
        ax.bar(
            x - width / 2,
            your_times,
            width,
            label="Your Implementation",
            color="steelblue",
        )
        ax.bar(x + width / 2, anomalib_times, width, label="Anomalib", color="coral")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Time")
        ax.set_title("Speed Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        # 3. FPS Comparison
        ax = axes[0, 2]
        fps_data = [your.inference_fps, anomalib.inference_fps]
        bars = ax.bar(
            ["Your Implementation", "Anomalib"], fps_data, color=["steelblue", "coral"]
        )
        ax.set_ylabel("FPS")
        ax.set_title("Inference Speed (FPS)")

        # Add value labels
        for bar, val in zip(bars, fps_data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
            )

        # 4. Model Size Comparison
        ax = axes[1, 0]
        categories = ["PyTorch"]
        your_sizes = [your.model_size_mb]
        anomalib_sizes = [anomalib.model_size_mb]

        # Add ONNX if available
        if your.export_size_mb.get("onnx") and anomalib.export_size_mb.get("onnx"):
            categories.append("ONNX")
            your_sizes.append(your.export_size_mb["onnx"])
            anomalib_sizes.append(anomalib.export_size_mb["onnx"])

        x = np.arange(len(categories))
        ax.bar(
            x - width / 2,
            your_sizes,
            width,
            label="Your Implementation",
            color="steelblue",
        )
        ax.bar(x + width / 2, anomalib_sizes, width, label="Anomalib", color="coral")
        ax.set_xlabel("Format")
        ax.set_ylabel("Size (MB)")
        ax.set_title("Model Size Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        # 5. Memory Usage Comparison
        ax = axes[1, 1]
        categories = ["Training", "Inference", "Peak"]
        your_memory = [
            your.training_memory_mb,
            your.inference_memory_mb,
            your.peak_memory_mb,
        ]
        anomalib_memory = [
            anomalib.training_memory_mb,
            anomalib.inference_memory_mb,
            anomalib.peak_memory_mb,
        ]

        x = np.arange(len(categories))
        ax.bar(
            x - width / 2,
            your_memory,
            width,
            label="Your Implementation",
            color="steelblue",
        )
        ax.bar(x + width / 2, anomalib_memory, width, label="Anomalib", color="coral")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        # 6. Overall Summary
        ax = axes[1, 2]

        # Calculate overall scores
        your_wins = 0
        anomalib_wins = 0

        # Performance
        if your.image_auroc > anomalib.image_auroc:
            your_wins += 1
        else:
            anomalib_wins += 1

        # Speed
        if your.inference_fps > anomalib.inference_fps:
            your_wins += 1
        else:
            anomalib_wins += 1

        # Size
        if your.model_size_mb < anomalib.model_size_mb:
            your_wins += 1
        else:
            anomalib_wins += 1

        # Memory
        if your.peak_memory_mb < anomalib.peak_memory_mb:
            your_wins += 1
        else:
            anomalib_wins += 1

        # Create pie chart
        sizes = [your_wins, anomalib_wins]
        labels = ["Your Implementation", "Anomalib"]
        colors = ["steelblue", "coral"]
        explode = (0.1, 0) if your_wins > anomalib_wins else (0, 0.1)

        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.0f%%",
            shadow=True,
            startangle=90,
        )
        ax.set_title("Overall Comparison\n(Wins across metrics)")

        plt.tight_layout()

        # Save figure
        fig_path = self.output_dir / f"comparison_charts_{self.class_name}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Visualizations saved to: {fig_path}")

        plt.show()


# ===============================
# Quick Comparison Function
# ===============================


def quick_compare(dataset_path: str, class_name: str = "bottle", device: str = "auto"):
    """
    Quick one-line comparison function.

    Args:
        dataset_path: Path to MVTec dataset
        class_name: Class to test on
        device: Device to use ("auto", "cuda", or "cpu")

    Returns:
        Dictionary with comparison results
    """
    runner = BenchmarkRunner(dataset_path, class_name, device)
    results = runner.run_comparison()
    return results


# ===============================
# Main Entry Point
# ===============================


def main():
    parser = argparse.ArgumentParser(
        description="Compare your anomaly detection implementation with Anomalib"
    )

    parser.add_argument(
        "--dataset_path", type=str, default="D:/01-DATA", help="Path to MVTec dataset"
    )

    parser.add_argument(
        "--class_name",
        type=str,
        default="bottle",
        choices=[
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ],
        help="MVTec class to test on",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for testing",
    )

    parser.add_argument(
        "--all_classes", action="store_true", help="Run comparison on all MVTec classes"
    )

    args = parser.parse_args()

    if args.all_classes:
        # Run on all classes
        all_classes = [
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ]

        all_results = {}
        for class_name in all_classes:
            print(f"\n{'='*60}")
            print(f"Testing on class: {class_name}")
            print("=" * 60)

            try:
                runner = BenchmarkRunner(args.dataset_path, class_name, args.device)
                results = runner.run_comparison()
                all_results[class_name] = results
            except Exception as e:
                print(f"Failed on {class_name}: {e}")
                all_results[class_name] = None

        # Generate summary report
        generate_summary_report(all_results)
    else:
        # Run on single class
        runner = BenchmarkRunner(args.dataset_path, args.class_name, args.device)
        results = runner.run_comparison()

        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE!")
        print("=" * 60)
        print(f"Results saved in: benchmark_results/")


def generate_summary_report(all_results: Dict):
    """Generate summary report across all classes."""
    output_dir = Path("benchmark_results")

    # Collect metrics
    summary_data = []

    for class_name, results in all_results.items():
        if results is None:
            continue

        your = results["your_implementation"]
        anomalib = results["anomalib"]

        summary_data.append(
            {
                "Class": class_name,
                "Your_Image_AUROC": your.image_auroc,
                "Anomalib_Image_AUROC": anomalib.image_auroc,
                "Your_Pixel_AUROC": your.pixel_auroc,
                "Anomalib_Pixel_AUROC": anomalib.pixel_auroc,
                "Your_FPS": your.inference_fps,
                "Anomalib_FPS": anomalib.inference_fps,
                "Your_Size_MB": your.model_size_mb,
                "Anomalib_Size_MB": anomalib.model_size_mb,
                "Your_Memory_MB": your.peak_memory_mb,
                "Anomalib_Memory_MB": anomalib.peak_memory_mb,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(summary_data)

    # Calculate averages
    avg_data = {
        "Class": "AVERAGE",
        "Your_Image_AUROC": df["Your_Image_AUROC"].mean(),
        "Anomalib_Image_AUROC": df["Anomalib_Image_AUROC"].mean(),
        "Your_Pixel_AUROC": df["Your_Pixel_AUROC"].mean(),
        "Anomalib_Pixel_AUROC": df["Anomalib_Pixel_AUROC"].mean(),
        "Your_FPS": df["Your_FPS"].mean(),
        "Anomalib_FPS": df["Anomalib_FPS"].mean(),
        "Your_Size_MB": df["Your_Size_MB"].mean(),
        "Anomalib_Size_MB": df["Anomalib_Size_MB"].mean(),
        "Your_Memory_MB": df["Your_Memory_MB"].mean(),
        "Anomalib_Memory_MB": df["Anomalib_Memory_MB"].mean(),
    }

    df = pd.concat([df, pd.DataFrame([avg_data])], ignore_index=True)

    # Save to CSV
    csv_path = output_dir / "summary_all_classes.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL CLASSES")
    print("=" * 60)

    # Performance summary
    print("\n### Performance Summary ###")
    print(
        f"Average Image AUROC - Your: {avg_data['Your_Image_AUROC']:.4f}, Anomalib: {avg_data['Anomalib_Image_AUROC']:.4f}"
    )
    print(
        f"Average Pixel AUROC - Your: {avg_data['Your_Pixel_AUROC']:.4f}, Anomalib: {avg_data['Anomalib_Pixel_AUROC']:.4f}"
    )

    # Speed summary
    print("\n### Speed Summary ###")
    print(
        f"Average FPS - Your: {avg_data['Your_FPS']:.2f}, Anomalib: {avg_data['Anomalib_FPS']:.2f}"
    )

    # Size summary
    print("\n### Size Summary ###")
    print(
        f"Average Model Size - Your: {avg_data['Your_Size_MB']:.2f} MB, Anomalib: {avg_data['Anomalib_Size_MB']:.2f} MB"
    )

    # Memory summary
    print("\n### Memory Summary ###")
    print(
        f"Average Peak Memory - Your: {avg_data['Your_Memory_MB']:.2f} MB, Anomalib: {avg_data['Anomalib_Memory_MB']:.2f} MB"
    )

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Summary Comparison Across All MVTec Classes", fontsize=16)

    # Plot 1: Image AUROC comparison
    ax = axes[0, 0]
    classes = df["Class"].values[:-1]  # Exclude average
    x = np.arange(len(classes))
    width = 0.35

    ax.bar(
        x - width / 2,
        df["Your_Image_AUROC"].values[:-1],
        width,
        label="Your Implementation",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        df["Anomalib_Image_AUROC"].values[:-1],
        width,
        label="Anomalib",
        color="coral",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Image AUROC")
    ax.set_title("Image AUROC by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: FPS comparison
    ax = axes[0, 1]
    ax.bar(
        x - width / 2,
        df["Your_FPS"].values[:-1],
        width,
        label="Your Implementation",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        df["Anomalib_FPS"].values[:-1],
        width,
        label="Anomalib",
        color="coral",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("FPS")
    ax.set_title("Inference Speed by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Model size comparison
    ax = axes[1, 0]
    ax.bar(
        x - width / 2,
        df["Your_Size_MB"].values[:-1],
        width,
        label="Your Implementation",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        df["Anomalib_Size_MB"].values[:-1],
        width,
        label="Anomalib",
        color="coral",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Size (MB)")
    ax.set_title("Model Size by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Overall averages
    ax = axes[1, 1]
    categories = [
        "Image\nAUROC",
        "Pixel\nAUROC",
        "FPS\n(norm)",
        "Size\n(norm)",
        "Memory\n(norm)",
    ]

    # Normalize FPS, Size, and Memory for comparison (0-1 scale)
    your_fps_norm = avg_data["Your_FPS"] / max(
        avg_data["Your_FPS"], avg_data["Anomalib_FPS"]
    )
    anomalib_fps_norm = avg_data["Anomalib_FPS"] / max(
        avg_data["Your_FPS"], avg_data["Anomalib_FPS"]
    )

    # For size and memory, invert so smaller is better
    max_size = max(avg_data["Your_Size_MB"], avg_data["Anomalib_Size_MB"])
    your_size_norm = 1 - (avg_data["Your_Size_MB"] / max_size)
    anomalib_size_norm = 1 - (avg_data["Anomalib_Size_MB"] / max_size)

    max_memory = max(avg_data["Your_Memory_MB"], avg_data["Anomalib_Memory_MB"])
    your_memory_norm = 1 - (avg_data["Your_Memory_MB"] / max_memory)
    anomalib_memory_norm = 1 - (avg_data["Anomalib_Memory_MB"] / max_memory)

    your_scores = [
        avg_data["Your_Image_AUROC"],
        avg_data["Your_Pixel_AUROC"],
        your_fps_norm,
        your_size_norm,
        your_memory_norm,
    ]

    anomalib_scores = [
        avg_data["Anomalib_Image_AUROC"],
        avg_data["Anomalib_Pixel_AUROC"],
        anomalib_fps_norm,
        anomalib_size_norm,
        anomalib_memory_norm,
    ]

    x = np.arange(len(categories))
    ax.bar(
        x - width / 2,
        your_scores,
        width,
        label="Your Implementation",
        color="steelblue",
    )
    ax.bar(x + width / 2, anomalib_scores, width, label="Anomalib", color="coral")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (normalized)")
    ax.set_title("Overall Average Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "summary_all_classes.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSummary visualization saved to: {fig_path}")

    plt.show()


if __name__ == "__main__":
    main()
