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


def _resize_masks_to_benchmark_size(masks: np.ndarray) -> np.ndarray:
    """Normalize ground-truth masks to the same 224x224 evaluation resolution."""
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (N,H,W), got {masks.shape}")
    if masks.shape[-2:] == IMAGE_SIZE:
        return masks
    tensor = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)
    resized = F.interpolate(tensor, size=IMAGE_SIZE, mode="nearest")
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
                images, labels, gt_masks = batch
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
        rows = []
        for item in (anomavision, anomalib):
            row = asdict(item)
            row["environment"] = json.dumps(row["environment"], sort_keys=True)
            rows.append(row)
        frame = pd.DataFrame(rows)
        stem = self.output_dir / f"padim_{self.class_name}"
        frame.to_csv(f"{stem}.csv", index=False)
        frame.to_json(f"{stem}.json", orient="records", indent=2)
        with open(f"{stem}.txt", "w", encoding="utf-8") as handle:
            handle.write(tabulate(frame, headers="keys", tablefmt="github", showindex=False))
            handle.write("\n")
        print("\n" + tabulate(frame, headers="keys", tablefmt="github", showindex=False))

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
