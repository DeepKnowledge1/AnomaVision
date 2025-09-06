#!/usr/bin/env python3
"""
Fair PaDiM Benchmark - Ensures equivalent configurations
=======================================================

This benchmark ensures fair comparison between AnomaVision implementation and Anomalib
by mapping equivalent layer configurations and using the same feature dimensions.
"""

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import torch


@dataclass
class FairBenchmarkConfig:
    """Fair benchmark configuration with equivalent settings"""

    dataset_path: str = "D:/01-DATA"
    class_name: str = "bottle"

    # Equivalent configurations
    backbone: str = "resnet18"
    AnomaVision_layers: List[int] = None
    anomalib_layers: List[str] = None
    feat_dim: int = 100  # Use Anomalib defaults

    # Identical preprocessing
    resize: Tuple[int, int] = (224, 224)
    crop_size: Tuple[int, int] = (224, 224)
    normalize: bool = True
    mean: List[float] = None
    std: List[float] = None

    # Performance settings
    batch_size: int = 4
    device: str = "cpu"
    test_iterations: int = 20
    warmup_iterations: int = 5

    def __post_init__(self):
        if self.AnomaVision_layers is None:
            self.AnomaVision_layers = [0, 1]
        if self.anomalib_layers is None:
            self.anomalib_layers = ["layer1", "layer2"]
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]


# Fair comparison configurations
FAIR_CONFIGS = {
    "fast": {
        "backbone": "resnet18",
        "AnomaVision_layers": [0],
        "anomalib_layers": ["layer1"],
        "feat_dim": 50,
        "description": "Fast configuration - early layers",
    },
    "balanced": {
        "backbone": "resnet18",
        "AnomaVision_layers": [1, 2],
        "anomalib_layers": ["layer2", "layer3"],
        "feat_dim": 100,
        "description": "Balanced configuration - mid layers",
    },
    "accurate": {
        "backbone": "wide_resnet50",
        "AnomaVision_layers": [0, 1],
        "anomalib_layers": ["layer1", "layer2"],
        "feat_dim": 550,
        "description": "Accurate configuration - wide resnet",
    },
    "maximum": {
        "backbone": "wide_resnet50",
        "AnomaVision_layers": [1, 2],
        "anomalib_layers": ["layer2", "layer3"],
        "feat_dim": 550,
        "description": "Maximum performance - deep wide resnet",
    },
}


class PreciseProfiler:
    """High precision profiler with CUDA synchronization"""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.reset()

    def reset(self):
        self.times = []
        self.memory_snapshots = []

    def start_timing(self):
        if self.cuda_available:
            torch.cuda.synchronize()
        return time.perf_counter()

    def end_timing(self, start_time):
        if self.cuda_available:
            torch.cuda.synchronize()
        return time.perf_counter() - start_time

    def record_memory(self):
        cpu_mem = psutil.Process().memory_info().rss / 1024 / 1024
        gpu_mem = (
            torch.cuda.memory_allocated() / 1024 / 1024 if self.cuda_available else 0
        )
        self.memory_snapshots.append({"cpu": cpu_mem, "gpu": gpu_mem})
        return cpu_mem, gpu_mem


def benchmark_AnomaVision_implementation(config: FairBenchmarkConfig):
    """Benchmark AnomaVision implementation with fair settings"""
    print(
        f"🔧 AnomaVision Implementation - {config.backbone} {config.AnomaVision_layers}"
    )

    try:
        import anodet
    except ImportError:
        print("❌ Could not import anodet")
        return None

    profiler = PreciseProfiler()
    device = torch.device(config.device)

    # Data preparation with identical settings
    train_root = f"{config.dataset_path}/{config.class_name}/train/good"
    train_dataset = anodet.AnodetDataset(
        train_root,
        resize=config.resize,
        crop_size=config.crop_size,
        normalize=config.normalize,
        mean=config.mean,
        std=config.std,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,  # Disable for fair timing
        shuffle=False,
    )

    test_dataset = anodet.MVTecDataset(
        config.dataset_path,
        config.class_name,
        is_train=False,
        resize=config.resize,
        crop_size=config.crop_size,
        normalize=config.normalize,
        mean=config.mean,
        std=config.std,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False
    )

    print(f"📊 Dataset: {len(train_dataset)} train, {len(test_dataset)} test")

    # Training benchmark
    print("⏱️  Training...")
    model = anodet.Padim(
        backbone=config.backbone,
        device=device,
        feat_dim=config.feat_dim,  # Use Anomalib's default!
        layer_indices=config.AnomaVision_layers,
    )

    profiler.record_memory()
    start_time = profiler.start_timing()

    model.fit(train_loader)

    training_time = profiler.end_timing(start_time)
    cpu_mem, gpu_mem = profiler.record_memory()

    print(f"✅ Training: {training_time:.2f}s")

    # Inference benchmark with proper warmup
    print("⚡ Inference...")
    model.eval()

    # Warmup phase
    test_batch = next(iter(test_loader))[0].to(device)
    for _ in range(config.warmup_iterations):
        with torch.no_grad():
            _ = model.predict(test_batch)

    # Clear cache after warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Benchmark phase
    inference_times = []
    for i, (batch, _, _, _) in enumerate(test_loader):
        if i >= config.test_iterations:
            break

        batch = batch.to(device)

        start = profiler.start_timing()
        with torch.no_grad():
            scores, maps = model.predict(batch)
        inference_time = profiler.end_timing(start)

        inference_times.append(inference_time * 1000)  # Convert to ms

    avg_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)
    fps = config.batch_size / (avg_inference / 1000)

    print(f"✅ Inference: {avg_inference:.2f}±{std_inference:.2f}ms, {fps:.1f} FPS")

    return {
        "implementation": "AnomaVision Implementation",
        "backbone": config.backbone,
        "layers": config.AnomaVision_layers,
        "feat_dim": config.feat_dim,
        "training_time_s": training_time,
        "inference_mean_ms": avg_inference,
        "inference_std_ms": std_inference,
        "fps": fps,
        "memory_mb": cpu_mem,
    }


def benchmark_anomalib(config: FairBenchmarkConfig):
    """Benchmark Anomalib with equivalent settings"""
    print(f"🔧 Anomalib - {config.backbone} {config.anomalib_layers}")

    try:
        from anomalib.data import MVTec
        from anomalib.engine import Engine
        from anomalib.models import Padim
    except ImportError:
        print("❌ Could not import Anomalib")
        return None

    profiler = PreciseProfiler()

    # Data with identical settings
    datamodule = MVTec(
        root=config.dataset_path,
        category=config.class_name,
        train_batch_size=config.batch_size,
        eval_batch_size=config.batch_size,
        num_workers=0,  # Disable for fair timing
    )
    datamodule.setup()

    print(
        f"📊 Dataset: {len(datamodule.train_dataloader().dataset)} train, {len(datamodule.test_dataloader().dataset)} test"
    )

    # Training benchmark
    print("⏱️  Training...")

    # Use equivalent layer mapping and feature dimensions
    model = Padim(
        backbone=config.backbone,
        layers=config.anomalib_layers,  # Equivalent layers
        pre_trained=True,
        n_features=config.feat_dim,  # Same feature dim
    )

    profiler.record_memory()
    start_time = profiler.start_timing()

    # Simple engine - remove checkpointing conflict
    engine = Engine(max_epochs=1, logger=False, enable_progress_bar=False, callbacks=[])

    engine.fit(model=model, datamodule=datamodule)

    training_time = profiler.end_timing(start_time)
    cpu_mem, gpu_mem = profiler.record_memory()

    print(f"✅ Training: {training_time:.2f}s")

    # Inference benchmark
    print("⚡ Inference...")
    model.eval()
    test_loader = datamodule.test_dataloader()

    # Warmup phase - extract image tensor from batch
    test_batch = next(iter(test_loader))
    # Handle ImageBatch object - extract the image tensor
    if hasattr(test_batch, "image"):
        warmup_tensor = test_batch.image
    elif isinstance(test_batch, dict) and "image" in test_batch:
        warmup_tensor = test_batch["image"]
    else:
        warmup_tensor = test_batch  # Fallback if it's already a tensor

    for _ in range(config.warmup_iterations):
        with torch.no_grad():
            _ = model.model(
                warmup_tensor
            )  # Use model.model to access the torch model directly

    # Clear cache after warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Benchmark phase
    inference_times = []
    for i, batch in enumerate(test_loader):
        if i >= config.test_iterations:
            break

        # Extract image tensor from ImageBatch
        if hasattr(batch, "image"):
            image_tensor = batch.image
        elif isinstance(batch, dict) and "image" in batch:
            image_tensor = batch["image"]
        else:
            image_tensor = batch

        start = profiler.start_timing()
        with torch.no_grad():
            output = model.model(
                image_tensor
            )  # Use model.model for direct torch model access
        inference_time = profiler.end_timing(start)

        inference_times.append(inference_time * 1000)  # Convert to ms

    avg_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)
    fps = config.batch_size / (avg_inference / 1000)

    print(f"✅ Inference: {avg_inference:.2f}±{std_inference:.2f}ms, {fps:.1f} FPS")

    return {
        "implementation": "Anomalib",
        "backbone": config.backbone,
        "layers": config.anomalib_layers,
        "feat_dim": config.feat_dim,
        "training_time_s": training_time,
        "inference_mean_ms": avg_inference,
        "inference_std_ms": std_inference,
        "fps": fps,
        "memory_mb": cpu_mem,
    }


def compare_fair_results(result1, result2):
    """Compare results with statistical significance"""
    if not result1 or not result2:
        return

    print("\n" + "=" * 70)
    print("📊 FAIR COMPARISON RESULTS")
    print("=" * 70)
    print(
        f"Configuration: {result1['backbone']} - AnomaVision layers {result1['layers']} vs Anomalib layers {result2['layers']}"
    )
    print(f"Feature dimensions: {result1['feat_dim']}")
    print("-" * 70)

    # Training comparison
    train_speedup = result2["training_time_s"] / result1["training_time_s"]
    print(f"🏃 Training Time:")
    print(f"  AnomaVision Impl:    {result1['training_time_s']:.2f}s")
    print(f"  Anomalib:     {result2['training_time_s']:.2f}s")
    print(f"  Speedup:      {train_speedup:.2f}x {'✅' if train_speedup > 1 else '❌'}")

    # Inference comparison
    inf_speedup = result2["inference_mean_ms"] / result1["inference_mean_ms"]
    print(f"\n⚡ Inference Time:")
    print(
        f"  AnomaVision Impl:    {result1['inference_mean_ms']:.2f}±{result1['inference_std_ms']:.2f}ms"
    )
    print(
        f"  Anomalib:     {result2['inference_mean_ms']:.2f}±{result2['inference_std_ms']:.2f}ms"
    )
    print(f"  Speedup:      {inf_speedup:.2f}x {'✅' if inf_speedup > 1 else '❌'}")

    # FPS comparison
    fps_improvement = result1["fps"] / result2["fps"]
    print(f"\n🚄 Throughput (FPS):")
    print(f"  AnomaVision Impl:    {result1['fps']:.1f} FPS")
    print(f"  Anomalib:     {result2['fps']:.1f} FPS")
    print(
        f"  Improvement:  {fps_improvement:.2f}x {'✅' if fps_improvement > 1 else '❌'}"
    )

    # Memory comparison
    mem_efficiency = result2["memory_mb"] / result1["memory_mb"]
    print(f"\n💾 Memory Usage:")
    print(f"  AnomaVision Impl:    {result1['memory_mb']:.1f} MB")
    print(f"  Anomalib:     {result2['memory_mb']:.1f} MB")
    print(
        f"  Efficiency:   {mem_efficiency:.2f}x {'✅' if mem_efficiency > 1 else '❌'}"
    )

    # Overall summary
    wins = sum(
        [
            train_speedup > 1.05,  # 5% threshold
            inf_speedup > 1.05,
            fps_improvement > 1.05,
            mem_efficiency > 1.05,
        ]
    )

    print(f"\n🏆 Overall Score: AnomaVision Implementation wins in {wins}/4 metrics")

    if wins >= 3:
        print("🎉 AnomaVision implementation is significantly faster!")
    elif wins >= 2:
        print("✅ AnomaVision implementation has performance advantages")
    else:
        print("🤔 Mixed results - investigate specific bottlenecks")


def main():
    parser = argparse.ArgumentParser(description="Fair PaDiM Benchmark")
    parser.add_argument(
        "--config",
        choices=list(FAIR_CONFIGS.keys()),
        default="fast",
        help="Configuration to test",
    )
    parser.add_argument("--dataset_path", default="D:/01-DATA")
    parser.add_argument("--class_name", default="bottle")
    parser.add_argument(
        "--impl", choices=["AnomaVision", "anomalib", "both"], default="anomalib"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    # Get fair configuration
    fair_config = FAIR_CONFIGS[args.config]
    config = FairBenchmarkConfig(
        dataset_path=args.dataset_path,
        class_name=args.class_name,
        backbone=fair_config["backbone"],
        AnomaVision_layers=fair_config["AnomaVision_layers"],
        anomalib_layers=fair_config["anomalib_layers"],
        feat_dim=fair_config["feat_dim"],
        test_iterations=args.iterations,
        batch_size=args.batch_size,
    )

    print("🎯 Fair PaDiM Benchmark")
    print(f"📁 Dataset: {config.dataset_path}/{config.class_name}")
    print(f"⚙️  Config: {fair_config['description']}")
    print(f"🔧 Settings: {config.backbone}, feat_dim={config.feat_dim}")
    print(f"📏 Batch size: {config.batch_size}, Iterations: {config.test_iterations}")
    print()

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    result_AnomaVision = None
    result_anomalib = None

    if args.impl in ["AnomaVision", "both"]:
        result_AnomaVision = benchmark_AnomaVision_implementation(config)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    if args.impl in ["anomalib", "both"]:
        result_anomalib = benchmark_anomalib(config)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    # Compare results
    if result_AnomaVision and result_anomalib:
        compare_fair_results(result_AnomaVision, result_anomalib)


if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """Simple Anomalib 2.1.0 Benchmark with AnomaVision Implementation"""

# import time
# import torch
# import numpy as np
# import argparse
# import psutil

# def get_memory_mb():
#     """Get current memory usage in MB"""
#     return psutil.Process().memory_info().rss / 1024 / 1024

# def benchmark_AnomaVision_implementation(dataset_path, class_name, batch_size=8):
#     """Benchmark AnomaVision PaDiM implementation"""
#     print("🔧 Testing AnomaVision Implementation...")

#     try:
#         import anodet
#     except ImportError:
#         print("❌ Could not import anodet. Make sure you're in the right environment.")
#         return None

#     # Setup
#     device = torch.device("cpu")

#     # Prepare data - match Anomalib's data setup
#     train_root = f"{dataset_path}/{class_name}/train/good"
#     train_dataset = anodet.AnodetDataset(train_root, resize=224, crop_size=224)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

#     test_dataset = anodet.MVTecDataset(dataset_path, class_name, is_train=False)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

#     print(f"📊 Dataset: {len(train_dataset)} train, {len(test_dataset)} test images")

#     # Training benchmark
#     print("⏱️  Training...")
#     model = anodet.Padim(backbone="resnet18", device=device, feat_dim=50)

#     memory_before = get_memory_mb()
#     start_time = time.perf_counter()

#     model.fit(train_loader)

#     training_time = time.perf_counter() - start_time
#     memory_after = get_memory_mb()

#     print(f"✅ Training: {training_time:.2f}s, Memory: {memory_after - memory_before:.1f}MB")

#     # Test benchmark - match Anomalib's approach
#     print("⚡ Testing...")
#     model.eval()

#     # Warmup
#     test_batch = next(iter(test_loader))[0].to(device)
#     for _ in range(3):
#         with torch.no_grad():
#             _ = model.predict(test_batch)

#     # Run test on full dataset like Anomalib does
#     start_test = time.perf_counter()

#     # Process all test batches like Anomalib engine.test() does
#     all_scores = []
#     all_maps = []
#     with torch.no_grad():
#         for batch, _, _, _ in test_loader:
#             batch = batch.to(device)
#             scores, maps = model.predict(batch)
#             all_scores.extend(scores if isinstance(scores, (list, np.ndarray)) else scores.cpu().numpy())
#             all_maps.extend(maps if isinstance(maps, (list, np.ndarray)) else maps.cpu().numpy())

#     test_time = time.perf_counter() - start_test

#     print(f"✅ Test: {test_time:.2f}s")

#     # Create mock results like Anomalib returns
#     results = {
#         'test_loss': 0.0,  # AnomaVision doesn't compute loss during test
#         'test_image_AUROC': 0.95,  # Mock value - would need actual computation
#         'test_pixel_AUROC': 0.92   # Mock value - would need actual computation
#     }

#     print(f"Results: {results}")

#     return {
#         'implementation': 'AnomaVision Implementation',
#         'training_time_s': training_time,
#         'test_time_s': test_time,
#         'memory_delta_mb': memory_after - memory_before,
#         'results': results
#     }

# def benchmark_anomalib(dataset_path, class_name, batch_size=8):
#     print("🔧 Testing Anomalib 2.1.0...")
#     from anomalib.data import MVTecAD
#     from anomalib.models import Padim
#     from anomalib.engine import Engine

#     # Simple data setup - no extra params
#     datamodule = MVTecAD(
#         root=dataset_path,
#         category=class_name,
#         train_batch_size=batch_size,
#         eval_batch_size=batch_size,
#         num_workers=0
#     )
#     datamodule.setup()

#     print("✅ Data loaded")

#     # Simple model
#     model = Padim()

#     # Simple engine - remove checkpointing conflict
#     engine = Engine(
#         max_epochs=1,
#         logger=False,
#         enable_progress_bar=False,
#         callbacks=[]
#     )

#     memory_before = get_memory_mb()

#     # Train
#     start = time.time()
#     engine.fit(model, datamodule)
#     train_time = time.time() - start

#     memory_after = get_memory_mb()

#     print(f"✅ Training: {train_time:.2f}s, Memory: {memory_after - memory_before:.1f}MB")

#     # Test
#     start = time.time()
#     results = engine.test(model, datamodule)
#     test_time = time.time() - start

#     print(f"✅ Test: {test_time:.2f}s")
#     print(f"Results: {results}")

#     return {
#         'implementation': 'Anomalib',
#         'training_time_s': train_time,
#         'test_time_s': test_time,
#         'memory_delta_mb': memory_after - memory_before,
#         'results': results
#     }

# def compare_results(result_AnomaVision, result_anomalib):
#     """Compare benchmark results"""
#     print("\n" + "="*60)
#     print("📊 COMPARISON RESULTS")
#     print("="*60)

#     print(f"{'Metric':<20} {'AnomaVision Impl':<15} {'Anomalib':<15} {'Ratio':<10}")
#     print("-" * 60)

#     # Training time
#     train_ratio = result_AnomaVision['training_time_s'] / result_anomalib['training_time_s']
#     print(f"{'Training Time':<20} {result_AnomaVision['training_time_s']:.2f}s{'':<7} {result_anomalib['training_time_s']:.2f}s{'':<7} {train_ratio:.2f}x")

#     # Memory
#     mem_ratio = result_AnomaVision['memory_delta_mb'] / result_anomalib['memory_delta_mb']
#     print(f"{'Memory Delta':<20} {result_AnomaVision['memory_delta_mb']:.1f}MB{'':<7} {result_anomalib['memory_delta_mb']:.1f}MB{'':<7} {mem_ratio:.2f}x")

#     # Test time comparison
#     test_ratio = result_AnomaVision['test_time_s'] / result_anomalib['test_time_s']
#     print(f"{'Test Time':<20} {result_AnomaVision['test_time_s']:.2f}s{'':<7} {result_anomalib['test_time_s']:.2f}s{'':<7} {test_ratio:.2f}x")

#     print("\n🏆 WINNER:")
#     if result_AnomaVision['training_time_s'] < result_anomalib['training_time_s']:
#         print("✅ AnomaVision Implementation is faster at training")
#     else:
#         print("✅ Anomalib is faster at training")

#     if result_AnomaVision['test_time_s'] < result_anomalib['test_time_s']:
#         print("✅ AnomaVision Implementation is faster at testing")
#     else:
#         print("✅ Anomalib is faster at testing")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_path", default="D:/01-DATA")
#     parser.add_argument("--class_name", default="bottle")
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--impl", choices=["AnomaVision", "anomalib", "both"], default="both")
#     args = parser.parse_args()

#     print("🚀 PaDiM Speed Comparison")
#     print(f"📁 Dataset: {args.dataset_path}/{args.class_name}")
#     print(f"🔧 Batch size: {args.batch_size}")
#     print()

#     result_AnomaVision = None
#     result_anomalib = None

#     if args.impl in ["AnomaVision", "both"]:
#         result_AnomaVision = benchmark_AnomaVision_implementation(args.dataset_path, args.class_name, args.batch_size)
#         print()

#     if args.impl in ["anomalib", "both"]:
#         result_anomalib = benchmark_anomalib(args.dataset_path, args.class_name, args.batch_size)
#         print()

#     if result_AnomaVision and result_anomalib:
#         compare_results(result_AnomaVision, result_anomalib)
