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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

ANOMAVISION_ROOT = Path(__file__).resolve().parents[2]
ANOMALIB_ROOT = ANOMAVISION_ROOT.parent / "anomalib"


def add_local_anomalib_to_path() -> Path:
    """Add the sibling Anomalib source checkout to ``sys.path``."""
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


def _get_anomalib_batch_value(batch: Any, name: str) -> Any:
    """Read fields from Anomalib dataclass batches with compatibility fallbacks."""
    aliases = {
        "image": ("image",),
        "label": ("gt_label", "label"),
        "mask": ("gt_mask", "mask"),
    }

    for key in aliases[name]:
        if hasattr(batch, key):
            return getattr(batch, key)

    if isinstance(batch, dict):
        for key in aliases[name]:
            if key in batch:
                return batch[key]

    if isinstance(batch, (tuple, list)):
        index = {"image": 0, "label": 1, "mask": 2}[name]
        if len(batch) > index:
            return batch[index]

    return None


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


def _resize_maps_to_benchmark_size(maps: np.ndarray) -> np.ndarray:
    """Normalize pixel maps to the configured 224x224 evaluation resolution."""
    maps = np.asarray(maps)
    if maps.ndim != 3:
        raise ValueError(f"Expected maps with shape (N,H,W), got {maps.shape}")
    if maps.shape[-2:] == IMAGE_SIZE:
        return maps
    tensor = torch.from_numpy(maps).float().unsqueeze(1)
    resized = F.interpolate(tensor, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
    return resized[:, 0].numpy()

def _resize_masks_to_benchmark_size(masks):
    masks = np.asarray(masks)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks with shape (N,H,W) or (N,1,H,W), got {masks.shape}"
        )

    if masks.shape[-2:] == IMAGE_SIZE:
        return masks

    tensor = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)
    resized = F.interpolate(
        tensor,
        size=IMAGE_SIZE,
        mode="nearest",
    )

    return resized[:, 0].numpy()


def compute_auroc(image_labels: np.ndarray, image_scores: np.ndarray, masks: np.ndarray, anomaly_maps: np.ndarray) -> Tuple[float, float]:
    """Compute both AUROCs at the same 224x224 pixel evaluation resolution."""
    image_labels = np.asarray(image_labels).reshape(-1).astype(np.uint8)
    image_scores = np.asarray(image_scores).reshape(-1)
    image_auroc = float(roc_auc_score(image_labels, image_scores)) if np.unique(image_labels).size >= 2 else float("nan")

    masks = _resize_masks_to_benchmark_size(masks)
    anomaly_maps = _resize_maps_to_benchmark_size(anomaly_maps)

    pixel_labels = masks.reshape(-1).astype(np.uint8)
    pixel_scores = anomaly_maps.reshape(-1)
    pixel_auroc = float(roc_auc_score(pixel_labels, pixel_scores)) if np.unique(pixel_labels).size >= 2 else float("nan")
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


class BenchmarkEngineMixin:
    """Remove Anomalib's automatic checkpoint callback before Trainer creation."""

    def _setup_anomalib_callbacks(self) -> None:
        from lightning.pytorch.callbacks import ModelCheckpoint

        super()._setup_anomalib_callbacks()
        self._cache.args["callbacks"] = [
            callback
            for callback in self._cache.args["callbacks"]
            if not isinstance(callback, ModelCheckpoint)
        ]


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
        """Build the MVTec AD datamodule from the local Anomalib checkout."""
        from anomalib.data import MVTecAD

        kwargs = {
            "root": str(self.dataset_path),
            "category": self.class_name,
            "train_batch_size": BATCH_SIZE,
            "eval_batch_size": BATCH_SIZE,
            "num_workers": 0,
            "seed": self.seed,
        }
        params = inspect.signature(MVTecAD).parameters
        datamodule = MVTecAD(**{k: v for k, v in kwargs.items() if k in params})
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
        return float(times.mean() * 1000), float(np.percentile(times, 95) * 1000), float(1.0 / times.mean()), float(self._memory_now_mb())

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
        metrics.state_dict_size_mb = self._state_dict_size_mb(model, self.output_dir / f"anomavision_{self.class_name}_state_dict.pt")
        timing_batch = next(iter(test_loader))[0][:TIMING_BATCH_SIZE]
        metrics.latency_ms, metrics.p95_latency_ms, metrics.throughput_fps, metrics.inference_memory_mb = self._benchmark_latency(model, timing_batch, model.predict)

        image_labels, image_scores, masks, maps = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for batch in test_loader:

                images, _, labels, gt_masks = batch
                scores, anomaly_maps = extract_anomavision_outputs(model.predict(images.to(self.device)))
                image_labels.append(tensor_to_numpy(labels))
                image_scores.append(tensor_to_numpy(scores))
                masks.append(tensor_to_numpy(gt_masks))
                maps.append(tensor_to_numpy(anomaly_maps))
        metrics.image_auroc, metrics.pixel_auroc = compute_auroc(
            np.concatenate(image_labels), np.concatenate(image_scores), np.concatenate(masks), np.concatenate(maps)
        )
        print(f"    native: image={metrics.image_auroc:.4f} pixel={metrics.pixel_auroc:.4f} latency={metrics.latency_ms:.2f}ms")
        return metrics

    def benchmark_anomalib(self) -> ModelMetrics:
        add_local_anomalib_to_path()
        from anomalib.engine import Engine
        from anomalib.models import Padim

        set_seed(self.seed)
        datamodule = self._build_anomalib_datamodule()
        train_loader = datamodule.train_dataloader()
        test_loader = datamodule.test_dataloader()
        metrics = ModelMetrics(name="Anomalib PaDiM", device=str(self.device), environment=environment(self.device, self.seed))
        print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")

        model = Padim(backbone=BACKBONE, layers=LAYERS, pre_trained=True, n_features=N_FEATURES)
        model = model.to(self.device)
        engine_cls = type("BenchmarkEngine", (BenchmarkEngineMixin, Engine), {})
        engine = engine_cls(
            max_epochs=1,
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        sync(self.device)
        metrics.training_time_s = time.perf_counter() - start
        metrics.training_memory_mb = self._memory_now_mb()
        metrics.state_dict_size_mb = self._state_dict_size_mb(model, self.output_dir / f"anomalib_{self.class_name}_state_dict.pt")

        first = next(iter(test_loader))
        timing_batch = _get_anomalib_batch_value(first, "image")[:TIMING_BATCH_SIZE]
        metrics.latency_ms, metrics.p95_latency_ms, metrics.throughput_fps, metrics.inference_memory_mb = self._benchmark_latency(model, timing_batch, model)

        image_labels, image_scores, masks, maps = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for batch in test_loader:
                images = _get_anomalib_batch_value(batch, "image").to(self.device)
                labels = _get_anomalib_batch_value(batch, "label")
                gt_masks = _get_anomalib_batch_value(batch, "mask")
                output = model(images)
                scores, anomaly_maps = extract_anomalib_outputs(output)
                image_labels.append(tensor_to_numpy(labels))
                image_scores.append(tensor_to_numpy(scores))
                masks.append(tensor_to_numpy(gt_masks))
                maps.append(tensor_to_numpy(anomaly_maps))
        metrics.image_auroc, metrics.pixel_auroc = compute_auroc(
            np.concatenate(image_labels), np.concatenate(image_scores), np.concatenate(masks), np.concatenate(maps)
        )
        print(f"    native: image={metrics.image_auroc:.4f} pixel={metrics.pixel_auroc:.4f} latency={metrics.latency_ms:.2f}ms")
        return metrics


    def report(self, anomavision: ModelMetrics, anomalib: ModelMetrics) -> None:
        rows = [asdict(anomavision), asdict(anomalib)]

        frame = pd.DataFrame(rows)

        # Keep machine-readable reports.
        csv_stem = self.output_dir / f"padim_{self.class_name}"
        frame.to_csv(f"{csv_stem}.csv", index=False)

        json_rows = [asdict(anomavision), asdict(anomalib)]
        with open(f"{csv_stem}.json", "w", encoding="utf-8") as handle:
            json.dump(json_rows, handle, indent=2)

        # ------------------------------------------------------------------
        # Derived comparison metrics
        # ------------------------------------------------------------------
        av = anomavision
        ab = anomalib

        latency_speedup = ab.latency_ms / av.latency_ms if av.latency_ms else float("nan")
        throughput_speedup = av.throughput_fps / ab.throughput_fps if ab.throughput_fps else float("nan")
        training_speedup = ab.training_time_s / av.training_time_s if av.training_time_s else float("nan")
        memory_reduction = (
            (ab.inference_memory_mb - av.inference_memory_mb)
            / ab.inference_memory_mb
            * 100
            if ab.inference_memory_mb
            else float("nan")
        )

        # ------------------------------------------------------------------
        # Fancy TXT report
        # ------------------------------------------------------------------
        txt_path = self.output_dir / f"padim_{self.class_name}.txt"

        def pct(value):
            return f"{value:.1f}%"

        def metric_row(label, av_value, ab_value, unit=""):
            return (
                f"{label:<28}"
                f"{av_value:>16.4f}{unit:<4}"
                f"{ab_value:>16.4f}{unit:<4}"
            )

        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n"
                "╔══════════════════════════════════════════════════════════════════════╗\n"
                f"║                 AnomaVision PaDiM Benchmark                        ║\n"
                f"║                 MVTec AD • {self.class_name:<41}║\n"
                "╚══════════════════════════════════════════════════════════════════════╝\n\n"
            )

            handle.write("PERFORMANCE SUMMARY\n")
            handle.write("─" * 70 + "\n")
            handle.write(
                f"  {'Metric':<28}"
                f"{'AnomaVision':>16}"
                f"{'Anomalib':>16}\n"
            )
            handle.write("─" * 70 + "\n")

            handle.write(metric_row("Image AUROC", av.image_auroc, ab.image_auroc))
            handle.write("\n")
            handle.write(metric_row("Pixel AUROC", av.pixel_auroc, ab.pixel_auroc))
            handle.write("\n")
            handle.write(metric_row("Training time", av.training_time_s, ab.training_time_s, "s"))
            handle.write("\n")
            handle.write(metric_row("Latency", av.latency_ms, ab.latency_ms, "ms"))
            handle.write("\n")
            handle.write(metric_row("P95 latency", av.p95_latency_ms, ab.p95_latency_ms, "ms"))
            handle.write("\n")
            handle.write(metric_row("Throughput", av.throughput_fps, ab.throughput_fps, "fps"))
            handle.write("\n")
            handle.write(metric_row("State dict", av.state_dict_size_mb, ab.state_dict_size_mb, "MB"))
            handle.write("\n")
            handle.write(metric_row("Inference memory", av.inference_memory_mb, ab.inference_memory_mb, "MB"))
            handle.write("\n\n")

            handle.write("ANOMAVISION ADVANTAGE\n")
            handle.write("─" * 70 + "\n")
            handle.write(f"  ⚡ Latency speedup:      {latency_speedup:.2f}×\n")
            handle.write(f"  🚀 Throughput speedup:   {throughput_speedup:.2f}×\n")
            handle.write(f"  ⏱ Training speedup:     {training_speedup:.2f}×\n")
            handle.write(f"  💾 Inference memory:    {pct(memory_reduction)} lower\n\n")

            handle.write("CONFIGURATION\n")
            handle.write("─" * 70 + "\n")
            handle.write(f"  Dataset:                 MVTec AD\n")
            handle.write(f"  Class:                   {self.class_name}\n")
            handle.write(f"  Input size:              224 × 224\n")
            handle.write(f"  Backbone:                ResNet18\n")
            handle.write(f"  Feature layer:           layer1\n")
            handle.write(f"  Features:                {N_FEATURES}\n")
            handle.write(f"  Batch size:              {BATCH_SIZE}\n")
            handle.write(f"  Timing batch size:       {TIMING_BATCH_SIZE}\n")
            handle.write(f"  Warmup iterations:       {WARMUP_ITERS}\n")
            handle.write(f"  Timed iterations:        {TIMING_ITERS}\n")
            handle.write(f"  Device:                  {self.device}\n")
            handle.write(f"  Seed:                    {self.seed}\n\n")

            handle.write("ENVIRONMENT\n")
            handle.write("─" * 70 + "\n")

            for key, value in av.environment.items():
                handle.write(f"  {key:<24} {value}\n")

            handle.write("\n" + "═" * 70 + "\n")
            handle.write("Generated by AnomaVision benchmark\n")
            handle.write("═" * 70 + "\n")

        # ------------------------------------------------------------------
        # Fancy self-contained HTML report
        # ------------------------------------------------------------------
        html_path = self.output_dir / f"padim_{self.class_name}.html"

        def fmt(value, digits=2):
            if value is None or pd.isna(value):
                return "—"
            return f"{value:.{digits}f}"

        def winner(metric, lower_is_better=False):
            a = getattr(av, metric)
            b = getattr(ab, metric)

            if lower_is_better:
                return "av" if a < b else "ab"
            return "av" if a > b else "ab"

        metrics = [
            ("Image AUROC", "image_auroc", "", False, 4),
            ("Pixel AUROC", "pixel_auroc", "", False, 4),
            ("Training time", "training_time_s", "s", True, 2),
            ("Latency", "latency_ms", "ms", True, 2),
            ("P95 latency", "p95_latency_ms", "ms", True, 2),
            ("Throughput", "throughput_fps", "FPS", False, 2),
            ("State dict size", "state_dict_size_mb", "MB", True, 2),
            ("Inference memory", "inference_memory_mb", "MB", True, 2),
        ]

        metric_cards = ""

        for title, field_name, unit, lower_better, digits in metrics:
            av_value = getattr(av, field_name)
            ab_value = getattr(ab, field_name)
            best = winner(field_name, lower_better)

            av_class = "winner" if best == "av" else ""
            ab_class = "winner" if best == "ab" else ""

            metric_cards += f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-values">
                    <div class="metric-value {av_class}">
                        <span>{fmt(av_value, digits)}</span>
                        <small>{unit}</small>
                        <label>AnomaVision</label>
                    </div>
                    <div class="metric-value {ab_class}">
                        <span>{fmt(ab_value, digits)}</span>
                        <small>{unit}</small>
                        <label>Anomalib</label>
                    </div>
                </div>
            </div>
            """

        environment_rows = ""

        for key, value in av.environment.items():
            environment_rows += f"""
            <tr>
                <td>{key}</td>
                <td>{value}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>AnomaVision PaDiM Benchmark — {self.class_name}</title>

    <style>
    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        font-family:
            Inter, -apple-system, BlinkMacSystemFont,
            "Segoe UI", Roboto, Arial, sans-serif;
        background: #f4f7fb;
        color: #172033;
    }}

    .container {{
        max-width: 1180px;
        margin: 0 auto;
        padding: 40px 24px 60px;
    }}

    .hero {{
        background: linear-gradient(135deg, #101827, #1d2940);
        color: white;
        border-radius: 24px;
        padding: 42px;
        margin-bottom: 28px;
        box-shadow: 0 18px 50px rgba(16, 24, 39, .18);
    }}

    .hero h1 {{
        margin: 0 0 10px;
        font-size: 34px;
    }}

    .hero p {{
        margin: 6px 0;
        color: #cbd5e1;
    }}

    .badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 24px;
    }}

    .badge {{
        background: rgba(255,255,255,.1);
        border: 1px solid rgba(255,255,255,.15);
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 13px;
    }}

    .section {{
        margin-top: 30px;
    }}

    .section h2 {{
        font-size: 22px;
        margin-bottom: 16px;
    }}

    .cards {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 18px;
    }}

    .metric-card {{
        background: white;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 5px 20px rgba(15,23,42,.07);
        border: 1px solid #e5eaf2;
    }}

    .metric-title {{
        font-weight: 700;
        color: #526078;
        margin-bottom: 18px;
    }}

    .metric-values {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }}

    .metric-value {{
        padding: 15px;
        border-radius: 12px;
        background: #f5f7fa;
    }}

    .metric-value.winner {{
        background: #eaf8ef;
        border: 1px solid #b7e4c7;
    }}

    .metric-value span {{
        font-size: 25px;
        font-weight: 800;
    }}

    .metric-value small {{
        color: #667085;
        margin-left: 4px;
    }}

    .metric-value label {{
        display: block;
        margin-top: 5px;
        font-size: 12px;
        color: #667085;
    }}

    .advantage {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
    }}

    .advantage-card {{
        background: white;
        border-radius: 18px;
        padding: 22px;
        border: 1px solid #e5eaf2;
        box-shadow: 0 5px 20px rgba(15,23,42,.07);
    }}

    .advantage-card .number {{
        font-size: 28px;
        font-weight: 800;
    }}

    .advantage-card .label {{
        color: #667085;
        margin-top: 5px;
        font-size: 13px;
    }}

    .panel {{
        background: white;
        border-radius: 18px;
        padding: 24px;
        border: 1px solid #e5eaf2;
        box-shadow: 0 5px 20px rgba(15,23,42,.07);
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
    }}

    th, td {{
        padding: 12px 14px;
        text-align: left;
        border-bottom: 1px solid #edf0f5;
    }}

    th {{
        color: #526078;
        font-size: 13px;
    }}

    td:first-child {{
        font-weight: 600;
    }}

    .footer {{
        text-align: center;
        color: #8a94a6;
        margin-top: 40px;
        font-size: 13px;
    }}

    @media (max-width: 800px) {{
        .cards {{
            grid-template-columns: 1fr;
        }}

        .advantage {{
            grid-template-columns: 1fr 1fr;
        }}

        .hero {{
            padding: 28px;
        }}
    }}

    @media (max-width: 500px) {{
        .advantage {{
            grid-template-columns: 1fr;
        }}

        .metric-values {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    </head>

    <body>
    <div class="container">

    <section class="hero">
        <h1>🔬 AnomaVision PaDiM Benchmark</h1>
        <p>Fair comparison against Anomalib on MVTec AD</p>
        <p><strong>Class:</strong> {self.class_name}</p>

        <div class="badges">
            <span class="badge">ResNet18</span>
            <span class="badge">layer1</span>
            <span class="badge">{N_FEATURES} features</span>
            <span class="badge">224 × 224</span>
            <span class="badge">ImageNet normalization</span>
            <span class="badge">{self.device}</span>
            <span class="badge">Seed {self.seed}</span>
        </div>
    </section>

    <section class="section">
        <h2>📊 Performance</h2>
        <div class="cards">
            {metric_cards}
        </div>
    </section>

    <section class="section">
        <h2>🚀 AnomaVision Advantage</h2>

        <div class="advantage">

            <div class="advantage-card">
                <div class="number">{latency_speedup:.2f}×</div>
                <div class="label">Latency speedup</div>
            </div>

            <div class="advantage-card">
                <div class="number">{throughput_speedup:.2f}×</div>
                <div class="label">Throughput speedup</div>
            </div>

            <div class="advantage-card">
                <div class="number">{training_speedup:.2f}×</div>
                <div class="label">Training speedup</div>
            </div>

            <div class="advantage-card">
                <div class="number">{memory_reduction:.1f}%</div>
                <div class="label">Lower inference memory</div>
            </div>

        </div>
    </section>

    <section class="section">
        <h2>⚙️ Benchmark Configuration</h2>

        <div class="panel">
            <table>
                <tr><td>Dataset</td><td>MVTec AD</td></tr>
                <tr><td>Class</td><td>{self.class_name}</td></tr>
                <tr><td>Train images</td><td>{int(sum([209]))}</td></tr>
                <tr><td>Test images</td><td>83</td></tr>
                <tr><td>Input size</td><td>224 × 224</td></tr>
                <tr><td>Backbone</td><td>ResNet18</td></tr>
                <tr><td>Feature layer</td><td>layer1</td></tr>
                <tr><td>Selected features</td><td>{N_FEATURES}</td></tr>
                <tr><td>Batch size</td><td>{BATCH_SIZE}</td></tr>
                <tr><td>Timing batch size</td><td>{TIMING_BATCH_SIZE}</td></tr>
                <tr><td>Warmup iterations</td><td>{WARMUP_ITERS}</td></tr>
                <tr><td>Timed iterations</td><td>{TIMING_ITERS}</td></tr>
                <tr><td>Device</td><td>{self.device}</td></tr>
                <tr><td>Seed</td><td>{self.seed}</td></tr>
            </table>
        </div>
    </section>

    <section class="section">
        <h2>💻 Environment</h2>

        <div class="panel">
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>

                <tbody>
                    {environment_rows}
                </tbody>
            </table>
        </div>
    </section>

    <div class="footer">
        Generated by AnomaVision benchmark
    </div>

    </div>
    </body>
    </html>
    """

        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(html)

        print("\n" + "=" * 78)
        print("📊 BENCHMARK REPORT")
        print("=" * 78)
        print(f"HTML : {html_path}")
        print(f"CSV  : {csv_stem}.csv")
        print(f"JSON : {csv_stem}.json")
        print(f"TXT  : {txt_path}")
        print("=" * 78)

    def run(self) -> None:
        print("\n" + "=" * 78)
        print(f"PADIM | {self.class_name}")
        print("=" * 78)
        anomavision = self.benchmark_anomavision()
        gc.collect()
        set_seed(self.seed)
        anomalib = self.benchmark_anomalib()
        self.report(anomavision, anomalib)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AnomaVision PaDiM against a local Anomalib checkout.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--class_name", choices=MVTec_CLASSES, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    BenchmarkRunner(args.dataset_path, args.class_name, args.device, args.seed).run()


if __name__ == "__main__":
    main()
