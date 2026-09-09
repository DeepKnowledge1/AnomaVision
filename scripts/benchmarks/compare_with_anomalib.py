"""Fair, reproducible AnomaVision vs Anomalib PaDiM benchmark.

The benchmark intentionally compares the same algorithmic configuration:

* MVTec AD official train/test split.
* 224x224 input images.
* ImageNet normalization.
* ResNet-18 ``layer1`` features only.
* 50 selected feature channels.
* Same batch size, device, seed, warm-up and timing policy.
* Image- and pixel-level AUROC calculated from raw model outputs with the
  same sklearn implementation, rather than mixing framework metrics.
* Inference timing excludes data loading, preprocessing and Python reporting.
* Model size is the serialized ``state_dict`` for both implementations,
  avoiding an unfair comparison between a Lightning checkpoint and a plain
  PyTorch object.

Important fairness note:
    Training time is end-to-end training time. Anomalib uses its Engine,
    while AnomaVision uses its native ``fit`` method. Therefore this metric
    includes framework/training-loop overhead and must not be presented as
    pure algorithm execution time.

Anomalib source:
    The benchmark uses a local Anomalib clone located next to AnomaVision,
    for example::

        D:\\Projects\\
        ├── AnomaVision\\
        └── anomalib\\

    Both ``anomalib\\src`` and a flat ``anomalib\\anomalib`` layout are
    supported. Anomalib itself is NOT installed from PyPI by this script.
    Its runtime dependencies, such as Lightning, must still be installed in
    the Python environment used to run the benchmark.

Requirements:
    torch torchvision numpy pandas scikit-learn psutil tabulate lightning

Examples:
    python scripts/benchmarks/compare_with_anomalib.py \
        --dataset_path /path/to/mvtec --class_name bottle

    python scripts/benchmarks/compare_with_anomalib.py \
        --dataset_path /path/to/mvtec --all_classes --device cuda
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import platform
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Local Anomalib source
# -----------------------------------------------------------------------------
# Do not install Anomalib as a package just to run this benchmark. Prefer the
# sibling source checkout so the comparison is against the exact code being
# inspected/modified locally.
ANOMAVISION_ROOT = Path(__file__).resolve().parents[2]
ANOMALIB_ROOT = ANOMAVISION_ROOT.parent / "anomalib"


def add_local_anomalib_to_path() -> Path:
    """Add the sibling Anomalib source checkout to ``sys.path``.

    Supports both the modern ``anomalib/src/anomalib`` layout and a flat
    ``anomalib/anomalib`` layout. The returned path is the directory that
    should be placed on ``sys.path``.
    """
    candidates = [ANOMALIB_ROOT / "src", ANOMALIB_ROOT]

    for candidate in candidates:
        package_dir = candidate / "anomalib"
        if package_dir.is_dir():
            candidate_str = str(candidate.resolve())
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate.resolve()

    raise ModuleNotFoundError(
        "Local Anomalib clone was not found. Expected one of:\n"
        f"  {ANOMALIB_ROOT / 'src' / 'anomalib'}\n"
        f"  {ANOMALIB_ROOT / 'anomalib'}\n"
        "The benchmark intentionally does not install Anomalib from PyPI."
    )


# -----------------------------------------------------------------------------
# One benchmark contract. Do not change one side independently.
# -----------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
NORMALIZE = True
BATCH_SIZE = 8
TIMING_BATCH_SIZE = 1
WARMUP_ITERS = 10
TIMING_ITERS = 100
SEED = 42
BACKBONE = "resnet18"
LAYERS = ["layer1"]
LAYER_INDICES = [0]
N_FEATURES = 50

MVTec_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


@dataclass
class ModelMetrics:
    name: str
    image_auroc: float = float("nan")
    pixel_auroc: float = float("nan")
    training_time_s: float = float("nan")
    latency_ms: float = float("nan")
    p95_latency_ms: float = float("nan")
    throughput_fps: float = float("nan")
    state_dict_size_mb: float = float("nan")
    training_memory_mb: float = float("nan")
    inference_memory_mb: float = float("nan")
    backbone: str = BACKBONE
    layers: str = "layer1"
    n_features: int = N_FEATURES
    device: str = ""
    environment: Dict[str, str] = field(default_factory=dict)


def set_seed(seed: int = SEED) -> None:
    """Set all relevant RNGs used by the benchmark."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def extract_anomalib_outputs(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract image scores and anomaly maps across Anomalib API versions."""
    score = None
    anomaly_map = None

    for key in ("pred_score", "anomaly_score", "image_score"):
        if hasattr(output, key):
            score = getattr(output, key)
            break
        if isinstance(output, dict) and key in output:
            score = output[key]
            break

    for key in ("anomaly_map", "pred_mask", "score_map"):
        if hasattr(output, key):
            anomaly_map = getattr(output, key)
            break
        if isinstance(output, dict) and key in output:
            anomaly_map = output[key]
            break

    if score is None or anomaly_map is None:
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            score, anomaly_map = output[0], output[1]
        else:
            raise TypeError(
                "Could not extract Anomalib image score and anomaly map from "
                f"output type {type(output)!r}."
            )

    score = score if isinstance(score, torch.Tensor) else torch.as_tensor(score)
    anomaly_map = anomaly_map if isinstance(anomaly_map, torch.Tensor) else torch.as_tensor(anomaly_map)

    if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
        anomaly_map = anomaly_map[:, 0]
    if score.ndim > 1:
        score = score.reshape(score.shape[0], -1).amax(dim=1)

    return score, anomaly_map


def extract_anomavision_outputs(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize AnomaVision's ``(image_scores, score_maps)`` output."""
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError(f"Unexpected AnomaVision output: {type(output)!r}")
    score, anomaly_map = output[0], output[1]
    score = score if isinstance(score, torch.Tensor) else torch.as_tensor(score)
    anomaly_map = anomaly_map if isinstance(anomaly_map, torch.Tensor) else torch.as_tensor(anomaly_map)
    if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
        anomaly_map = anomaly_map[:, 0]
    if score.ndim > 1:
        score = score.reshape(score.shape[0], -1).amax(dim=1)
    return score, anomaly_map


def compute_auroc(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
) -> Tuple[float, float]:
    """Compute both AUROCs with exactly the same sklearn code for both models."""
    image_labels = np.asarray(image_labels).reshape(-1).astype(np.uint8)
    image_scores = np.asarray(image_scores).reshape(-1)
    image_auroc = (
        float(roc_auc_score(image_labels, image_scores))
        if np.unique(image_labels).size >= 2
        else float("nan")
    )

    masks = np.asarray(masks)
    anomaly_maps = np.asarray(anomaly_maps)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if anomaly_maps.ndim == 4 and anomaly_maps.shape[1] == 1:
        anomaly_maps = anomaly_maps[:, 0]
    if masks.shape[-2:] != anomaly_maps.shape[-2:]:
        raise ValueError(
            "Prediction/ground-truth map size mismatch: "
            f"pred={anomaly_maps.shape}, gt={masks.shape}"
        )

    pixel_labels = masks.reshape(-1).astype(np.uint8)
    pixel_scores = anomaly_maps.reshape(-1)
    pixel_auroc = (
        float(roc_auc_score(pixel_labels, pixel_scores))
        if np.unique(pixel_labels).size >= 2
        else float("nan")
    )
    return image_auroc, pixel_auroc


def environment(device: torch.device, seed: int) -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(device),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": str(seed),
        "image_size": "224x224",
        "normalization": "ImageNet",
        "backbone": BACKBONE,
        "layers": ",".join(LAYERS),
        "n_features": str(N_FEATURES),
        "batch_size": str(BATCH_SIZE),
        "timing_batch_size": str(TIMING_BATCH_SIZE),
        "warmup_iters": str(WARMUP_ITERS),
        "timing_iters": str(TIMING_ITERS),
    }


class BenchmarkRunner:
    """Run a controlled AnomaVision/Anomalib PaDiM comparison."""

    def __init__(self, dataset_path: str, class_name: str, device: str = "auto", seed: int = SEED) -> None:
        self.dataset_path = Path(dataset_path)
        self.class_name = class_name
        self.device = self._setup_device(device)
        self.seed = int(seed)
        set_seed(self.seed)
        self.output_dir = Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

    @staticmethod
    def _setup_device(device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(device)

    def _reset_memory(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _memory_now_mb(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1024**2
        return psutil.Process().memory_info().rss / 1024**2

    def _build_anomavision_datasets(self):
        from anomavision import MVTecDataset
        train = MVTecDataset(self.dataset_path, self.class_name, is_train=True, resize=IMAGE_SIZE, crop_size=IMAGE_SIZE, normalize=NORMALIZE)
        test = MVTecDataset(self.dataset_path, self.class_name, is_train=False, resize=IMAGE_SIZE, crop_size=IMAGE_SIZE, normalize=NORMALIZE)
        return train, test

    def _build_anomalib_datamodule(self):
        from anomalib.data import MVTec
        kwargs = {
            "root": str(self.dataset_path),
            "category": self.class_name,
            "image_size": IMAGE_SIZE,
            "train_batch_size": BATCH_SIZE,
            "eval_batch_size": BATCH_SIZE,
            "num_workers": 0,
            "pin_memory": False,
        }
        params = inspect.signature(MVTec).parameters
        if "normalization" in params:
            try:
                from anomalib.data import NormalizationMethod
                kwargs["normalization"] = NormalizationMethod.IMAGENET
            except (ImportError, AttributeError):
                kwargs["normalization"] = "imagenet"
        elif "normalize" in params:
            kwargs["normalize"] = True
        datamodule = MVTec(**{k: v for k, v in kwargs.items() if k in params})
        datamodule.setup()
        return datamodule

    @staticmethod
    def _state_dict_size_mb(model: torch.nn.Module, path: Path) -> float:
        """Compare serialized model parameters/buffers, not framework checkpoints."""
        torch.save(model.state_dict(), path)
        return path.stat().st_size / 1024**2

    def _benchmark_latency(self, model: torch.nn.Module, batch: torch.Tensor, call_model) -> Tuple[float, float, float, float]:
        model.eval()
        batch = batch.to(self.device)
        self._reset_memory()
        with torch.inference_mode():
            for _ in range(WARMUP_ITERS):
                call_model(batch)
            sync(self.device)
            times = []
            for _ in range(TIMING_ITERS):
                sync(self.device)
                start = time.perf_counter()
                call_model(batch)
                sync(self.device)
                times.append(time.perf_counter() - start)
        times = np.asarray(times, dtype=np.float64)
        return (
            float(times.mean() * 1000),
            float(np.percentile(times, 95) * 1000),
            float(1.0 / times.mean()),
            float(self._memory_now_mb()),
        )

    def benchmark_anomavision(self) -> ModelMetrics:
        print("\n" + "=" * 70)
        print("ANOMAVISION PaDiM")
        print("=" * 70)
        from anomavision import Padim

        set_seed(self.seed)
        train_dataset, test_dataset = self._build_anomavision_datasets()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

        metrics = ModelMetrics(name="AnomaVision PaDiM", device=str(self.device), environment=environment(self.device, self.seed))
        print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
        print("Configuration: ResNet18 / layer1 / 50 features / 224x224 / ImageNet")

        self._reset_memory()
        model = Padim(backbone=BACKBONE, device=self.device, feat_dim=N_FEATURES, layer_indices=LAYER_INDICES)
        start = time.perf_counter()
        model.fit(train_loader)
        sync(self.device)
        metrics.training_time_s = time.perf_counter() - start
        metrics.training_memory_mb = self._memory_now_mb()

        state_path = self.output_dir / f"anomavision_{self.class_name}_state_dict.pt"
        metrics.state_dict_size_mb = self._state_dict_size_mb(model, state_path)

        timing_batch = next(iter(test_loader))[0][:TIMING_BATCH_SIZE]
        metrics.latency_ms, metrics.p95_latency_ms, metrics.throughput_fps, metrics.inference_memory_mb = self._benchmark_latency(model, timing_batch, model.predict)

        image_labels, image_scores, masks, maps = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for batch, _images, labels, batch_masks in test_loader:
                scores, score_maps = extract_anomavision_outputs(model.predict(batch.to(self.device)))
                image_labels.append(tensor_to_numpy(labels))
                image_scores.append(tensor_to_numpy(scores))
                masks.append(tensor_to_numpy(batch_masks))
                maps.append(tensor_to_numpy(score_maps))

        metrics.image_auroc, metrics.pixel_auroc = compute_auroc(np.concatenate(image_labels), np.concatenate(image_scores), np.concatenate(masks), np.concatenate(maps))
        print_metrics(metrics)
        return metrics

    def benchmark_anomalib(self) -> ModelMetrics:
        print("\n" + "=" * 70)
        print("ANOMALIB PaDiM")
        print("=" * 70)

        # Use the sibling source checkout, not a pip-installed Anomalib.
        add_local_anomalib_to_path()

        try:
            import lightning  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The local Anomalib clone was found, but its Lightning runtime "
                "dependency is missing. Install the dependency in this venv "
                "with:\n\n"
                "    python -m pip install lightning\n\n"
                "Do NOT install Anomalib from PyPI for this benchmark."
            ) from exc

        from anomalib.engine import Engine
        from anomalib.models import Padim as AnomalibPadim

        set_seed(self.seed)
        datamodule = self._build_anomalib_datamodule()
        metrics = ModelMetrics(name="Anomalib PaDiM", device=str(self.device), environment=environment(self.device, self.seed))

        train_ds = datamodule.train_dataloader().dataset
        test_ds = datamodule.test_dataloader().dataset
        print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
        print("Configuration: ResNet18 / layer1 / 50 features / 224x224 / ImageNet")

        self._reset_memory()
        model = AnomalibPadim(backbone=BACKBONE, layers=LAYERS, pre_trained=True, n_features=N_FEATURES)
        accelerator = "gpu" if self.device.type == "cuda" else "cpu"
        engine = Engine(max_epochs=1, accelerator=accelerator, devices=1, logger=False, enable_progress_bar=False, enable_checkpointing=False)

        # Intentionally end-to-end: includes Anomalib Engine/training-loop overhead.
        start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        sync(self.device)
        metrics.training_time_s = time.perf_counter() - start
        metrics.training_memory_mb = self._memory_now_mb()

        state_path = self.output_dir / f"anomalib_{self.class_name}_state_dict.pt"
        metrics.state_dict_size_mb = self._state_dict_size_mb(model, state_path)

        test_loader = datamodule.test_dataloader()
        first_batch = next(iter(test_loader))
        if isinstance(first_batch, dict):
            timing_batch = first_batch["image"][:TIMING_BATCH_SIZE]
        else:
            timing_batch = first_batch[0][:TIMING_BATCH_SIZE]

        def forward(batch):
            return model(batch)

        metrics.latency_ms, metrics.p95_latency_ms, metrics.throughput_fps, metrics.inference_memory_mb = self._benchmark_latency(model, timing_batch, forward)

        image_labels, image_scores, masks, maps = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for batch in test_loader:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    labels = batch.get("label", batch.get("gt_label"))
                    batch_masks = batch.get("mask", batch.get("gt_mask"))
                else:
                    images = batch[0].to(self.device)
                    labels = batch[1]
                    batch_masks = batch[2] if len(batch) > 2 else None

                if labels is None or batch_masks is None:
                    raise RuntimeError("Anomalib test loader did not provide labels/masks.")

                scores, score_maps = extract_anomalib_outputs(model(images))
                image_labels.append(tensor_to_numpy(labels))
                image_scores.append(tensor_to_numpy(scores))
                masks.append(tensor_to_numpy(batch_masks))
                maps.append(tensor_to_numpy(score_maps))

        metrics.image_auroc, metrics.pixel_auroc = compute_auroc(np.concatenate(image_labels), np.concatenate(image_scores), np.concatenate(masks), np.concatenate(maps))
        print_metrics(metrics)
        return metrics

    def run(self) -> Dict[str, ModelMetrics]:
        print("\n" + "=" * 70)
        print("FAIR PaDiM COMPARISON")
        print("=" * 70)
        print(f"Dataset : {self.dataset_path}")
        print(f"Class   : {self.class_name}")
        print(f"Device  : {self.device}")
        print("Both models: ResNet18 / layer1 / 50 features / 224x224 / ImageNet")

        anomavision = self.benchmark_anomavision()
        self._reset_memory()
        set_seed(self.seed)
        anomalib = self.benchmark_anomalib()
        results = {"anomavision": anomavision, "anomalib": anomalib}
        self.write_report(results)
        return results

    def write_report(self, results: Dict[str, ModelMetrics]) -> None:
        av, ab = results["anomavision"], results["anomalib"]
        rows = [
            ["Image AUROC", av.image_auroc, ab.image_auroc, "higher is better"],
            ["Pixel AUROC", av.pixel_auroc, ab.pixel_auroc, "higher is better"],
            ["Training time (s)", av.training_time_s, ab.training_time_s, "lower is better; includes framework overhead"],
            ["Batch-1 latency (ms)", av.latency_ms, ab.latency_ms, "lower is better"],
            ["Batch-1 P95 latency (ms)", av.p95_latency_ms, ab.p95_latency_ms, "lower is better"],
            ["Batch-1 throughput (FPS)", av.throughput_fps, ab.throughput_fps, "higher is better"],
            ["State dict size (MB)", av.state_dict_size_mb, ab.state_dict_size_mb, "lower is better"],
            ["Measured memory (MB)", av.inference_memory_mb, ab.inference_memory_mb, "GPU peak allocated; CPU RSS"],
        ]
        table = [["Metric", "AnomaVision", "Anomalib", "Interpretation"]]
        for metric, left, right, note in rows:
            fmt = ".4f" if "AUROC" in metric else ".2f"
            table.append([metric, format(left, fmt), format(right, fmt), note])
        print("\n" + tabulate(table, headers="firstrow", tablefmt="grid"))

        report = {
            "benchmark_contract": {
                "dataset": "MVTec AD",
                "class": self.class_name,
                "input_size": IMAGE_SIZE,
                "normalization": "ImageNet",
                "backbone": BACKBONE,
                "layers": LAYERS,
                "n_features": N_FEATURES,
                "train_batch_size": BATCH_SIZE,
                "timing_batch_size": TIMING_BATCH_SIZE,
                "warmup_iters": WARMUP_ITERS,
                "timing_iters": TIMING_ITERS,
                "seed": self.seed,
                "device": str(self.device),
                "accuracy_metric": "sklearn.metrics.roc_auc_score on raw outputs",
                "training_time_definition": "end-to-end training including framework overhead",
                "model_size_definition": "serialized model.state_dict() only",
                "anomalib_source": str(ANOMALIB_ROOT.resolve()),
            },
            "results": {name: asdict(metrics) for name, metrics in results.items()},
        }
        json_path = self.output_dir / f"comparison_results_{self.class_name}.json"
        txt_path = self.output_dir / f"comparison_report_{self.class_name}.txt"
        csv_path = self.output_dir / f"comparison_{self.class_name}.csv"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        txt_path.write_text(tabulate(table, headers="firstrow", tablefmt="grid"), encoding="utf-8")
        pd.DataFrame(rows, columns=["metric", "anomavision", "anomalib", "interpretation"]).to_csv(csv_path, index=False)
        print(f"\nSaved: {json_path}")
        print(f"Saved: {txt_path}")
        print(f"Saved: {csv_path}")


def print_metrics(metrics: ModelMetrics) -> None:
    print(f"  Image AUROC        : {metrics.image_auroc:.4f}")
    print(f"  Pixel AUROC        : {metrics.pixel_auroc:.4f}")
    print(f"  Training time      : {metrics.training_time_s:.2f} s")
    print(f"  Batch-1 latency    : {metrics.latency_ms:.2f} ms")
    print(f"  Batch-1 P95        : {metrics.p95_latency_ms:.2f} ms")
    print(f"  Batch-1 throughput : {metrics.throughput_fps:.2f} FPS")
    print(f"  State dict size    : {metrics.state_dict_size_mb:.2f} MB")
    print(f"  Memory             : {metrics.inference_memory_mb:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair AnomaVision vs Anomalib PaDiM benchmark")
    parser.add_argument("--dataset_path", required=True, help="Path to MVTec AD root directory")
    parser.add_argument("--class_name", default="bottle", choices=MVTec_CLASSES)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--all_classes", action="store_true", help="Run all 15 MVTec classes")
    args = parser.parse_args()

    if args.all_classes:
        for class_name in MVTec_CLASSES:
            print(f"\n\n{'#' * 80}\n# {class_name.upper()}\n{'#' * 80}")
            BenchmarkRunner(args.dataset_path, class_name, args.device, args.seed).run()
    else:
        BenchmarkRunner(args.dataset_path, args.class_name, args.device, args.seed).run()


if __name__ == "__main__":
    main()
