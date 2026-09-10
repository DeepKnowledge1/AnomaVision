"""Fair AnomaVision vs Anomalib benchmark for PaDiM, PatchCore and EfficientAD."""

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
    """Add the sibling Anomalib source checkout to sys.path."""
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
ALGORITHMS = ("padim", "patchcore", "efficientad")


@dataclass
class ModelMetrics:
    name: str
    algorithm: str
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
            raise TypeError(f"Could not extract Anomalib outputs from {type(output)!r}")
    score = score if isinstance(score, torch.Tensor) else torch.as_tensor(score)
    anomaly_map = (
        anomaly_map
        if isinstance(anomaly_map, torch.Tensor)
        else torch.as_tensor(anomaly_map)
    )
    if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
        anomaly_map = anomaly_map[:, 0]
    if score.ndim > 1:
        score = score.reshape(score.shape[0], -1).amax(dim=1)
    return score, anomaly_map


def extract_anomavision_outputs(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError(f"Unexpected AnomaVision output: {type(output)!r}")
    score, anomaly_map = output[0], output[1]
    score = score if isinstance(score, torch.Tensor) else torch.as_tensor(score)
    anomaly_map = (
        anomaly_map
        if isinstance(anomaly_map, torch.Tensor)
        else torch.as_tensor(anomaly_map)
    )
    if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
        anomaly_map = anomaly_map[:, 0]
    if score.ndim > 1:
        score = score.reshape(score.shape[0], -1).amax(dim=1)
    return score, anomaly_map


def _resize_maps_to_benchmark_size(maps: np.ndarray) -> np.ndarray:
    maps = np.asarray(maps)
    if maps.ndim != 3:
        raise ValueError(f"Expected maps with shape (N,H,W), got {maps.shape}")
    if maps.shape[-2:] == IMAGE_SIZE:
        return maps
    tensor = torch.from_numpy(maps).float().unsqueeze(1)
    return F.interpolate(tensor, size=IMAGE_SIZE, mode="bilinear", align_corners=False)[
        :, 0
    ].numpy()


def _resize_masks_to_benchmark_size(masks: np.ndarray) -> np.ndarray:
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
    return F.interpolate(tensor, size=IMAGE_SIZE, mode="nearest")[:, 0].numpy()


def compute_auroc(
    image_labels, image_scores, masks, anomaly_maps
) -> Tuple[float, float]:
    image_labels = np.asarray(image_labels).reshape(-1).astype(np.uint8)
    image_scores = np.asarray(image_scores).reshape(-1)
    image_auroc = (
        float(roc_auc_score(image_labels, image_scores))
        if np.unique(image_labels).size >= 2
        else float("nan")
    )
    masks = _resize_masks_to_benchmark_size(masks)
    anomaly_maps = _resize_maps_to_benchmark_size(anomaly_maps)
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


class BenchmarkEngineMixin:
    def _setup_anomalib_callbacks(self) -> None:
        from lightning.pytorch.callbacks import ModelCheckpoint

        super()._setup_anomalib_callbacks()
        self._cache.args["callbacks"] = [
            callback
            for callback in self._cache.args["callbacks"]
            if not isinstance(callback, ModelCheckpoint)
        ]


class BenchmarkRunner:
    def __init__(
        self, dataset_path: str, class_name: str, device: str = "auto", seed: int = SEED
    ) -> None:
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

        train = MVTecDataset(
            self.dataset_path,
            self.class_name,
            is_train=True,
            resize=IMAGE_SIZE,
            crop_size=IMAGE_SIZE,
            normalize=NORMALIZE,
        )
        test = MVTecDataset(
            self.dataset_path,
            self.class_name,
            is_train=False,
            resize=IMAGE_SIZE,
            crop_size=IMAGE_SIZE,
            normalize=NORMALIZE,
        )
        return train, test

    def _build_anomalib_datamodule(self, algorithm: str):
        from anomalib.data import MVTecAD

        kwargs = {
            "root": str(self.dataset_path),
            "category": self.class_name,
            "train_batch_size": 1 if algorithm == "efficientad" else BATCH_SIZE,
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
        torch.save(model.state_dict(), path)
        return path.stat().st_size / 1024**2

    def _benchmark_latency(
        self, model, batch: torch.Tensor, call_model
    ) -> Tuple[float, float, float, float]:
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
        times = np.asarray(times)
        return (
            float(times.mean() * 1000),
            float(np.percentile(times, 95) * 1000),
            float(1.0 / times.mean()),
            float(self._memory_now_mb()),
        )

    def _build_anomavision_model(self, algorithm: str):
        if algorithm == "padim":
            from anomavision import Padim

            return Padim(
                backbone=BACKBONE,
                device=self.device,
                feat_dim=N_FEATURES,
                layer_indices=LAYER_INDICES,
            )
        if algorithm == "patchcore":
            from anomavision import PatchCore

            return PatchCore(
                backbone=BACKBONE,
                device=self.device,
                layer_indices=LAYER_INDICES,
                coreset_ratio=0.02,
                max_memory_patches=2048,
                patch_grid=14,
                n_neighbors=1,
                coreset_seed=self.seed,
            )
        if algorithm == "efficientad":
            from anomavision import EfficientAD

            return EfficientAD(
                backbone=BACKBONE, device=self.device, epochs=5, learning_rate=1e-3
            )
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _build_anomalib_model(self, algorithm: str):
        add_local_anomalib_to_path()
        from anomalib.models import EfficientAd, Padim, Patchcore
        from anomalib.models.image.efficient_ad.lightning_model import (
            EfficientAdModelSize,
        )

        if algorithm == "padim":
            return Padim(
                backbone=BACKBONE,
                layers=LAYERS,
                pre_trained=True,
                n_features=N_FEATURES,
            )
        if algorithm == "patchcore":
            return Patchcore(
                backbone=BACKBONE,
                layers=LAYERS,
                pre_trained=True,
                coreset_sampling_ratio=0.02,
                num_neighbors=1,
                precision="float32",
            )
        if algorithm == "efficientad":
            # return EfficientAd(model_size="s", lr=1e-3, padding=False, pad_maps=True)

            return EfficientAd(
                model_size=EfficientAdModelSize.S,
                lr=1e-3,
                padding=False,
                pad_maps=True,
            )
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _evaluate_anomavision(self, model, test_loader) -> Tuple[float, float]:
        image_labels, image_scores, masks, maps = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for batch in test_loader:
                images, _, labels, gt_masks = batch
                scores, anomaly_maps = extract_anomavision_outputs(
                    model.predict(images.to(self.device))
                )
                image_labels.append(tensor_to_numpy(labels))
                image_scores.append(tensor_to_numpy(scores))
                masks.append(tensor_to_numpy(gt_masks))
                maps.append(tensor_to_numpy(anomaly_maps))
        return compute_auroc(
            np.concatenate(image_labels),
            np.concatenate(image_scores),
            np.concatenate(masks),
            np.concatenate(maps),
        )

    def _evaluate_anomalib(self, model, test_loader) -> Tuple[float, float]:
        image_labels, image_scores, masks, maps = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for batch in test_loader:
                images = _get_anomalib_batch_value(batch, "image").to(self.device)
                labels = _get_anomalib_batch_value(batch, "label")
                gt_masks = _get_anomalib_batch_value(batch, "mask")
                scores, anomaly_maps = extract_anomalib_outputs(model(images))
                image_labels.append(tensor_to_numpy(labels))
                image_scores.append(tensor_to_numpy(scores))
                masks.append(tensor_to_numpy(gt_masks))
                maps.append(tensor_to_numpy(anomaly_maps))
        return compute_auroc(
            np.concatenate(image_labels),
            np.concatenate(image_scores),
            np.concatenate(masks),
            np.concatenate(maps),
        )

    def benchmark_anomavision(self, algorithm: str) -> ModelMetrics:
        from anomavision import MVTecDataset

        set_seed(self.seed)
        train_dataset, test_dataset = self._build_anomavision_datasets()
        batch_size = 1 if algorithm == "efficientad" else BATCH_SIZE
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        model = self._build_anomavision_model(algorithm)
        metrics = ModelMetrics(
            f"AnomaVision {algorithm.title()}",
            algorithm,
            device=str(self.device),
            environment=environment(self.device, self.seed),
        )
        print(
            f"\nANOMAVISION {algorithm.upper()} | Train: {len(train_dataset)} | Test: {len(test_dataset)}"
        )
        self._reset_memory()
        start = time.perf_counter()
        model.fit(train_loader)
        sync(self.device)
        metrics.training_time_s = time.perf_counter() - start
        metrics.training_memory_mb = self._memory_now_mb()
        metrics.state_dict_size_mb = self._state_dict_size_mb(
            model,
            self.output_dir
            / f"anomavision_{algorithm}_{self.class_name}_state_dict.pt",
        )
        timing_batch = next(iter(test_loader))[0][:TIMING_BATCH_SIZE]
        (
            metrics.latency_ms,
            metrics.p95_latency_ms,
            metrics.throughput_fps,
            metrics.inference_memory_mb,
        ) = self._benchmark_latency(model, timing_batch, model.predict)
        metrics.image_auroc, metrics.pixel_auroc = self._evaluate_anomavision(
            model, test_loader
        )
        print(
            f"  image={metrics.image_auroc:.4f} pixel={metrics.pixel_auroc:.4f} latency={metrics.latency_ms:.2f}ms"
        )
        return metrics

    def benchmark_anomalib(self, algorithm: str) -> ModelMetrics:
        add_local_anomalib_to_path()
        from anomalib.engine import Engine

        set_seed(self.seed)
        datamodule = self._build_anomalib_datamodule(algorithm)
        train_loader = datamodule.train_dataloader()
        test_loader = datamodule.test_dataloader()
        model = self._build_anomalib_model(algorithm).to(self.device)
        metrics = ModelMetrics(
            f"Anomalib {algorithm.title()}",
            algorithm,
            device=str(self.device),
            environment=environment(self.device, self.seed),
        )
        print(
            f"ANOMALIB {algorithm.upper()} | Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}"
        )
        engine_cls = type("BenchmarkEngine", (BenchmarkEngineMixin, Engine), {})
        engine = engine_cls(
            max_epochs=1,
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        self._reset_memory()
        start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        sync(self.device)
        metrics.training_time_s = time.perf_counter() - start
        metrics.training_memory_mb = self._memory_now_mb()
        metrics.state_dict_size_mb = self._state_dict_size_mb(
            model,
            self.output_dir / f"anomalib_{algorithm}_{self.class_name}_state_dict.pt",
        )
        timing_batch = _get_anomalib_batch_value(next(iter(test_loader)), "image")[
            :TIMING_BATCH_SIZE
        ]
        (
            metrics.latency_ms,
            metrics.p95_latency_ms,
            metrics.throughput_fps,
            metrics.inference_memory_mb,
        ) = self._benchmark_latency(model, timing_batch, model)
        metrics.image_auroc, metrics.pixel_auroc = self._evaluate_anomalib(
            model, test_loader
        )
        print(
            f"  image={metrics.image_auroc:.4f} pixel={metrics.pixel_auroc:.4f} latency={metrics.latency_ms:.2f}ms"
        )
        return metrics

    def report(self, results: List[ModelMetrics]) -> None:
        rows = [asdict(item) for item in results]
        frame = pd.DataFrame(rows)
        frame["environment"] = frame["environment"].apply(
            lambda value: json.dumps(value, sort_keys=True)
        )
        # stem = self.output_dir /  f"benchmark_{self.class_name}"

        algorithm = results[0].algorithm

        stem = (
            self.output_dir
            / algorithm
            / self.class_name
            / f"benchmark_{algorithm}_{self.class_name}"
        )

        stem.parent.mkdir(parents=True, exist_ok=True)

        frame.to_csv(f"{stem}.csv", index=False)
        with open(f"{stem}.json", "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        summary = frame[
            [
                "algorithm",
                "name",
                "image_auroc",
                "pixel_auroc",
                "training_time_s",
                "latency_ms",
                "p95_latency_ms",
                "throughput_fps",
                "state_dict_size_mb",
                "training_memory_mb",
                "inference_memory_mb",
            ]
        ].copy()
        with open(f"{stem}.txt", "w", encoding="utf-8") as handle:
            handle.write("AnomaVision vs Anomalib Benchmark\n")
            handle.write("=" * 80 + "\n\n")
            handle.write(
                tabulate(
                    summary,
                    headers="keys",
                    tablefmt="github",
                    showindex=False,
                    floatfmt=".4f",
                )
            )
            handle.write("\n")
        html_rows = []
        for row in rows:
            html_rows.append(
                "<tr>"
                + "".join(
                    f"<td>{row[k]}</td>"
                    for k in [
                        "algorithm",
                        "name",
                        "image_auroc",
                        "pixel_auroc",
                        "training_time_s",
                        "latency_ms",
                        "p95_latency_ms",
                        "throughput_fps",
                        "state_dict_size_mb",
                    ]
                )
                + "</tr>"
            )
        html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AnomaVision Benchmark - {self.class_name}</title><style>body{{font-family:Inter,Arial,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}}main{{max-width:1400px;margin:40px auto;padding:24px}}h1{{margin-bottom:6px}}.sub{{color:#94a3b8;margin-bottom:28px}}.card{{background:#111827;border:1px solid #334155;border-radius:16px;padding:20px;margin-bottom:24px;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:12px 14px;border-bottom:1px solid #334155;text-align:left}}th{{color:#93c5fd;background:#1e293b;position:sticky;top:0}}tr:hover{{background:#1e293b}}.pill{{display:inline-block;padding:6px 10px;border-radius:999px;background:#1e293b;margin-right:6px}}</style></head><body><main><h1>🔬 AnomaVision Benchmark</h1><div class="sub">MVTec AD · {self.class_name} · 224×224 · {self.device}</div><div class="card"><span class="pill">PaDiM</span><span class="pill">PatchCore</span><span class="pill">EfficientAD</span><span class="pill">AnomaVision vs Anomalib</span></div><div class="card"><table><thead><tr><th>Algorithm</th><th>Implementation</th><th>Image AUROC</th><th>Pixel AUROC</th><th>Fit Time (s)</th><th>Latency (ms)</th><th>P95 (ms)</th><th>FPS</th><th>Model (MB)</th></tr></thead><tbody>{''.join(html_rows)}</tbody></table></div></main></body></html>"""
        with open(f"{stem}.html", "w", encoding="utf-8") as handle:
            handle.write(html)
        print("\nReports:")
        for suffix in ("csv", "json", "txt", "html"):
            print(f"  {stem}.{suffix}")

    def run(self, algorithms: List[str]) -> None:
        results = []
        for algorithm in algorithms:
            results.append(self.benchmark_anomavision(algorithm))
            gc.collect()
            set_seed(self.seed)
            results.append(self.benchmark_anomalib(algorithm))
            gc.collect()
            set_seed(self.seed)
        self.report(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AnomaVision against a local Anomalib checkout."
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--class_name", choices=MVTec_CLASSES, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    args = parser.parse_args()
    BenchmarkRunner(args.dataset_path, args.class_name, args.device, args.seed).run(
        args.algorithms
    )


if __name__ == "__main__":
    main()
