"""Fair AnomaVision vs Anomalib benchmark for PaDiM, PatchCore and EfficientAD."""

# git checkout 4207f163a556c5dbf59ed58ed372d8c47dc526a0


from __future__ import annotations

import argparse
import gc
import inspect
import io
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
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from tabulate import tabulate
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

ANOMAVISION_ROOT = Path(__file__).resolve().parents[2]
ANOMALIB_ROOT = ANOMAVISION_ROOT.parent / "anomalib"


# AnomaVision
coreset_ratio = 0.02
max_memory_patches = 2048
patch_grid = 14
n_neighbors = 1


# Anomalib
coreset_sampling_ratio = 0.02
num_neighbors = 1
precision = "float32"


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

ANOMA_BACKBONE = {
    "resnet18": "resnet18",
    "wide_resnet50": "wide_resnet50_2",
}

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
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]
ALGORITHMS = ("padim", "patchcore", "efficientad")


class AnomaVisionCompatiblePool(torch.nn.Module):
    def forward(self, features):
        features = torch.nn.functional.adaptive_avg_pool2d(
            features, (patch_grid, patch_grid)
        )
        return torch.nn.functional.normalize(features, p=2, dim=1)


def load_torchvision_weights_into_anomalib(model) -> None:
    """Load matching torchvision pretrained weights into Anomalib's extractor."""
    from torchvision.models import (
        ResNet18_Weights,
        Wide_ResNet50_2_Weights,
        resnet18,
        wide_resnet50_2,
    )

    backbones = {
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        "wide_resnet50": (
            wide_resnet50_2,
            Wide_ResNet50_2_Weights.DEFAULT,
        ),
    }

    if BACKBONE not in backbones:
        raise ValueError(f"Unsupported benchmark backbone: {BACKBONE}")

    model_fn, weights = backbones[BACKBONE]
    torchvision_model = model_fn(weights=weights)
    tv_state = torchvision_model.state_dict()

    target = model.model.feature_extractor.feature_extractor
    result = target.load_state_dict(tv_state, strict=False)

    if result.missing_keys:
        raise RuntimeError(
            f"Backbone weights not fully loaded. "
            f"Missing keys: {result.missing_keys}"
        )

    target.eval()


@dataclass
class ModelMetrics:
    name: str
    algorithm: str
    image_ap: float = float("nan")
    image_f1: float = float("nan")
    image_precision: float = float("nan")
    image_recall: float = float("nan")
    image_auroc: float = float("nan")
    image_specificity: float = float("nan")
    pixel_ap: float = float("nan")
    pixel_f1: float = float("nan")
    pixel_precision: float = float("nan")
    pixel_recall: float = float("nan")
    pixel_auroc: float = float("nan")
    pixel_aupro: float = float("nan")
    pixel_dice: float = float("nan")
    pixel_iou: float = float("nan")
    end_to_end_latency_ms: float = float("nan")
    timing_outlier_fraction: float = float("nan")
    timing_cv: float = float("nan")
    timing_anomaly_flag: bool = False
    training_time_s: float = float("nan")
    latency_ms: float = float("nan")
    p95_latency_ms: float = float("nan")
    throughput_fps: float = float("nan")
    state_dict_size_mb: float = float("nan")
    training_memory_mb: float = float("nan")
    inference_memory_mb: float = float("nan")
    backbone: str = ANOMA_BACKBONE[BACKBONE]
    image_size: str = f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
    layer_indices: list[int] = field(default_factory=lambda: list(LAYER_INDICES))
    layers: str = ",".join(LAYERS)
    n_features: int = N_FEATURES
    device: str = ""
    environment: dict[str, str] = field(default_factory=dict)


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


def compute_metrics(
    image_labels, image_scores, masks, anomaly_maps
) -> Dict[str, float]:
    image_labels = np.asarray(image_labels).reshape(-1).astype(np.uint8)
    image_scores = np.asarray(image_scores).reshape(-1)
    image_auroc = (
        float(roc_auc_score(image_labels, image_scores))
        if np.unique(image_labels).size >= 2
        else float("nan")
    )
    image_ap = (
        float(average_precision_score(image_labels, image_scores))
        if np.unique(image_labels).size >= 2
        else float("nan")
    )
    image_precision_curve, image_recall_curve, image_thresholds = (
        precision_recall_curve(image_labels, image_scores)
    )
    image_f1_curve = (
        2
        * image_precision_curve
        * image_recall_curve
        / np.maximum(image_precision_curve + image_recall_curve, 1e-12)
    )
    image_idx = int(np.nanargmax(image_f1_curve)) if image_f1_curve.size else 0
    image_threshold = (
        image_thresholds[min(image_idx, max(0, len(image_thresholds) - 1))]
        if len(image_thresholds)
        else float("inf")
    )
    image_pred = image_scores >= image_threshold
    image_tp = int(np.sum((image_pred == 1) & (image_labels == 1)))
    image_fp = int(np.sum((image_pred == 1) & (image_labels == 0)))
    image_fn = int(np.sum((image_pred == 0) & (image_labels == 1)))
    image_tn = int(np.sum((image_pred == 0) & (image_labels == 0)))
    image_precision = image_tp / max(1, image_tp + image_fp)
    image_recall = image_tp / max(1, image_tp + image_fn)
    image_f1 = (
        2 * image_precision * image_recall / max(1e-12, image_precision + image_recall)
    )
    image_specificity = image_tn / max(1, image_tn + image_fp)
    masks = _resize_masks_to_benchmark_size(masks)
    anomaly_maps = _resize_maps_to_benchmark_size(anomaly_maps)
    pixel_labels = masks.reshape(-1).astype(np.uint8)
    pixel_scores = anomaly_maps.reshape(-1)
    pixel_auroc = (
        float(roc_auc_score(pixel_labels, pixel_scores))
        if np.unique(pixel_labels).size >= 2
        else float("nan")
    )
    pixel_ap = (
        float(average_precision_score(pixel_labels, pixel_scores))
        if np.unique(pixel_labels).size >= 2
        else float("nan")
    )
    pixel_precision_curve, pixel_recall_curve, pixel_thresholds = (
        precision_recall_curve(pixel_labels, pixel_scores)
    )
    pixel_f1_curve = (
        2
        * pixel_precision_curve
        * pixel_recall_curve
        / np.maximum(pixel_precision_curve + pixel_recall_curve, 1e-12)
    )
    pixel_idx = int(np.nanargmax(pixel_f1_curve)) if pixel_f1_curve.size else 0
    pixel_threshold = (
        pixel_thresholds[min(pixel_idx, max(0, len(pixel_thresholds) - 1))]
        if len(pixel_thresholds)
        else float("inf")
    )
    pixel_pred = pixel_scores >= pixel_threshold
    pixel_tp = int(np.sum((pixel_pred == 1) & (pixel_labels == 1)))
    pixel_fp = int(np.sum((pixel_pred == 1) & (pixel_labels == 0)))
    pixel_fn = int(np.sum((pixel_pred == 0) & (pixel_labels == 1)))
    pixel_precision = pixel_tp / max(1, pixel_tp + pixel_fp)
    pixel_recall = pixel_tp / max(1, pixel_tp + pixel_fn)
    pixel_f1 = (
        2 * pixel_precision * pixel_recall / max(1e-12, pixel_precision + pixel_recall)
    )
    pixel_dice = pixel_f1
    pixel_iou = pixel_tp / max(1, pixel_tp + pixel_fp + pixel_fn)
    pixel_aupro = float("nan")
    try:
        from scipy.ndimage import label as connected_components

        gt = masks.astype(bool)
        thresholds = np.unique(np.quantile(anomaly_maps, np.linspace(0.0, 1.0, 100)))
        region_overlaps, false_positive_rates = [], []
        normal_pixels = int((~gt).sum())
        for threshold in thresholds:
            pred = anomaly_maps >= threshold
            fpr = float((pred & ~gt).sum() / max(1, normal_pixels))
            overlaps = []
            for idx in range(len(gt)):
                components, count = connected_components(gt[idx])
                for component_id in range(1, count + 1):
                    region = components == component_id
                    overlaps.append(
                        float((pred[idx] & region).sum() / max(1, region.sum()))
                    )
            if overlaps:
                false_positive_rates.append(fpr)
                region_overlaps.append(float(np.mean(overlaps)))
        if len(false_positive_rates) >= 2:
            order = np.argsort(false_positive_rates)
            x = np.asarray(false_positive_rates)[order]
            y = np.asarray(region_overlaps)[order]
            keep = x <= 0.3
            x, y = x[keep], y[keep]
            if len(x) >= 2 and x[-1] > x[0]:
                pixel_aupro = float(np.trapz(y, x) / (x[-1] - x[0]))
    except (ImportError, ValueError, FloatingPointError):
        pass
    return {
        "image_ap": image_ap,
        "image_f1": image_f1,
        "image_precision": image_precision,
        "image_recall": image_recall,
        "image_auroc": image_auroc,
        "image_specificity": image_specificity,
        "pixel_ap": pixel_ap,
        "pixel_f1": pixel_f1,
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_auroc": pixel_auroc,
        "pixel_aupro": pixel_aupro,
        "pixel_dice": pixel_dice,
        "pixel_iou": pixel_iou,
    }


def environment(device: torch.device, seed: int) -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(device),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": str(seed),
        "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "normalization": "ImageNet",
        "backbone": ANOMA_BACKBONE[BACKBONE],
        "layer_indices": json.dumps(LAYER_INDICES),
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
        self,
        dataset_path: str,
        class_name: str,
        device: str = "auto",
        seed: int = SEED,
        save_models: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.class_name = class_name
        self.device = self._setup_device(device)
        self.seed = int(seed)
        self.save_models = bool(save_models)
        set_seed(self.seed)
        self.output_root = Path("benchmark_results")

    def _result_dir(self, algorithm: str) -> Path:
        """Return the output folder for this backbone/algorithm/size/class."""
        result_dir = (
            self.output_root
            / BACKBONE
            / algorithm
            / f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
            / self.class_name
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

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

    # def _build_anomalib_datamodule(self, algorithm: str):
    #     # from anomalib.data import MVTecAD

    #     from anomalib.data import Visa as MVTecAD
    #     kwargs = {
    #         "root": str(self.dataset_path),
    #         "category": self.class_name,
    #         "train_batch_size": 1 if algorithm == "efficientad" else BATCH_SIZE,
    #         "eval_batch_size": BATCH_SIZE,
    #         "num_workers": 0,
    #         "seed": self.seed,
    #     }
    #     params = inspect.signature(MVTecAD).parameters
    #     datamodule = MVTecAD(**{k: v for k, v in kwargs.items() if k in params})
    #     datamodule.setup()
    #     return datamodule

    def _build_anomalib_datamodule(self, algorithm: str):
        from anomalib.data import MVTecAD, Visa

        dataset_name = self.dataset_path.name.lower()

        if "visa" in dataset_name:
            datamodule_cls = Visa
        else:
            datamodule_cls = MVTecAD

        kwargs = {
            "root": str(self.dataset_path),
            "category": self.class_name,
            "train_batch_size": 1 if algorithm == "efficientad" else BATCH_SIZE,
            "eval_batch_size": BATCH_SIZE,
            "num_workers": 0,
            "seed": self.seed,
        }

        params = inspect.signature(datamodule_cls).parameters

        # Apply the benchmark image size only if supported by this constructor.
        if "image_size" in params:
            kwargs["image_size"] = IMAGE_SIZE

        datamodule = datamodule_cls(
            **{key: value for key, value in kwargs.items() if key in params}
        )
        datamodule.setup()
        return datamodule

    def _state_dict_size_mb(self, model: torch.nn.Module, path: Path) -> float:
        """Measure serialized state size; save a .pt file only when requested."""
        # Count only what inference actually uses: drop ResNet stages above the
        # highest requested layer (and the classifier head). AnomaVision keeps the
        # full ResNet registered; Anomalib's timm extractor is already truncated.
        unused = tuple(f".layer{i}." for i in range(max(LAYER_INDICES) + 2, 5))
        unused += (".fc.",)
        state = {
            key: value
            for key, value in model.state_dict().items()
            if not any(tag in f".{key}" for tag in unused)
        }

        buffer = io.BytesIO()
        torch.save(state, buffer)
        size_bytes = buffer.tell()

        if self.save_models:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)

        return size_bytes / 1024**2

    def _benchmark_latency(
        self, model, batch: torch.Tensor, call_model
    ) -> Tuple[float, float, float, float, float, float, bool]:
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
        median = float(np.median(times))
        mad = float(np.median(np.abs(times - median)))
        robust_z = np.abs(times - median) / max(1.4826 * mad, 1e-12)
        outlier_fraction = float(np.mean(robust_z > 3.5))
        cv = float(times.std() / max(times.mean(), 1e-12))
        anomaly_flag = bool(outlier_fraction > 0.10 or cv > 0.25)
        return (
            float(times.mean() * 1000),
            float(np.percentile(times, 95) * 1000),
            float(1.0 / times.mean()),
            float(self._memory_now_mb()),
            outlier_fraction,
            cv,
            anomaly_flag,
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
                coreset_ratio=coreset_ratio,
                max_memory_patches=max_memory_patches,
                patch_grid=patch_grid,
                n_neighbors=n_neighbors,
                coreset_seed=self.seed,
            )

        if algorithm == "efficientad":
            from anomavision import EfficientAD

            return EfficientAD(
                backbone=BACKBONE, device=self.device, epochs=1, learning_rate=1e-3
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
                backbone=ANOMA_BACKBONE[BACKBONE],
                layers=LAYERS,
                pre_trained=True,
                n_features=N_FEATURES,
                pre_processor=Padim.configure_pre_processor(image_size=IMAGE_SIZE),
            )
        # if algorithm == "patchcore":
        #     return Patchcore(
        #         backbone=ANOMA_BACKBONE[BACKBONE],
        #         layers=LAYERS,
        #         pre_trained=True,
        #         coreset_sampling_ratio=coreset_sampling_ratio,
        #         num_neighbors=num_neighbors,
        #         precision=precision,
        #     )

        # if algorithm == "patchcore":
        #     return Patchcore(
        #         backbone=ANOMA_BACKBONE[BACKBONE],
        #         layers=LAYERS,
        #         pre_trained=True,
        #         coreset_sampling_ratio=coreset_sampling_ratio,
        #         num_neighbors=num_neighbors,
        #         precision=precision,
        #         pre_processor=Patchcore.configure_pre_processor(
        #             image_size=IMAGE_SIZE,
        #         ),
        #     )

        # if algorithm == "patchcore":
        #     model = Patchcore(
        #         backbone=ANOMA_BACKBONE[BACKBONE],
        #         layers=LAYERS,
        #         pre_trained=True,
        #         coreset_sampling_ratio=coreset_sampling_ratio,
        #         num_neighbors=num_neighbors,
        #         precision=precision,
        #         pre_processor=Patchcore.configure_pre_processor(
        #             image_size=IMAGE_SIZE,
        #         ),
        #     )

        #     # model.model.feature_pooler = AnomaVisionCompatiblePool()
        #     return model

        if algorithm == "patchcore":
            model = Patchcore(
                backbone=ANOMA_BACKBONE[BACKBONE],
                layers=LAYERS,
                pre_trained=True,
                coreset_sampling_ratio=2048 / (900 * 14 * 14),
                num_neighbors=1,
                precision="float32",
                pre_processor=Patchcore.configure_pre_processor(
                    image_size=IMAGE_SIZE,
                ),
            )

            load_torchvision_weights_into_anomalib(model)
            model.model.feature_pooler = AnomaVisionCompatiblePool()
            return model

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
        return compute_metrics(
            np.concatenate(image_labels),
            np.concatenate(image_scores),
            np.concatenate(masks),
            np.concatenate(maps),
        )

    @staticmethod
    def _anomalib_preprocess(model, images):
        """Apply the same transform used during fitting."""
        processed = model.pre_processor.transform(images)

        if not getattr(model, "_debug_shapes_printed", False):
            print(f"[DEBUG] Input images: {tuple(images.shape)}")
            print(f"[DEBUG] After preprocessing: {tuple(processed.shape)}")
            model._debug_shapes_printed = True

        return processed

    @staticmethod
    def _anomalib_raw(model, images):
        """Torch model on already pre-processed images, skipping Anomalib's
        post-processor (min-max normalisation + clamping), like AnomaVision."""
        model.model = model.model.to(images.device)
        output = model.model(images)

        if not getattr(model, "_debug_features_printed", False):
            print(f"[DEBUG] Model input: {tuple(images.shape)}")
            print(f"[DEBUG] Model output type: {type(output).__name__}")

            if isinstance(output, dict):
                print(
                    "[DEBUG] Feature/output shapes:",
                    {
                        key: tuple(value.shape)
                        for key, value in output.items()
                        if hasattr(value, "shape")
                    },
                )

            model._debug_features_printed = True

        return output

    def _evaluate_anomalib(self, model, test_loader) -> Tuple[float, float]:
        image_labels, image_scores, masks, maps = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for batch in test_loader:
                images = _get_anomalib_batch_value(batch, "image").to(self.device)
                labels = _get_anomalib_batch_value(batch, "label")
                gt_masks = _get_anomalib_batch_value(batch, "mask")
                scores, anomaly_maps = extract_anomalib_outputs(
                    self._anomalib_raw(model, self._anomalib_preprocess(model, images))
                )
                image_labels.append(tensor_to_numpy(labels))
                image_scores.append(tensor_to_numpy(scores))
                masks.append(tensor_to_numpy(gt_masks))
                maps.append(tensor_to_numpy(anomaly_maps))
        return compute_metrics(
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
            self._result_dir(algorithm)
            / f"anomavision_{algorithm}_{self.class_name}_state_dict.pt",
        )
        timing_batch = next(iter(test_loader))[0][:TIMING_BATCH_SIZE]
        (
            metrics.latency_ms,
            metrics.p95_latency_ms,
            metrics.throughput_fps,
            metrics.inference_memory_mb,
            outlier_fraction,
            timing_cv,
            anomaly_flag,
        ) = self._benchmark_latency(model, timing_batch, model.predict)
        self._last_timing_checks = (outlier_fraction, timing_cv, anomaly_flag)
        (
            metrics.timing_outlier_fraction,
            metrics.timing_cv,
            metrics.timing_anomaly_flag,
        ) = self._last_timing_checks
        metrics.end_to_end_latency_ms = float(
            "nan"
        )  # preprocessing is performed by the dataset, outside model.predict timing
        metrics.__dict__.update(self._evaluate_anomavision(model, test_loader))
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

        if algorithm == "patchcore":
            patch_count = len(train_loader.dataset) * 14 * 14
            model.coreset_sampling_ratio = min(
                0.02,
                2048 / patch_count,
            )

        self._reset_memory()
        start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        sync(self.device)
        metrics.training_time_s = time.perf_counter() - start
        metrics.training_memory_mb = self._memory_now_mb()
        metrics.state_dict_size_mb = self._state_dict_size_mb(
            model,
            self._result_dir(algorithm)
            / f"anomalib_{algorithm}_{self.class_name}_state_dict.pt",
        )
        timing_batch = _get_anomalib_batch_value(next(iter(test_loader)), "image")[
            :TIMING_BATCH_SIZE
        ]
        # Pre-process once, outside timing (AnomaVision's dataset also pre-processes
        # outside predict()), so latency measures model inference only.
        timing_batch = self._anomalib_preprocess(model, timing_batch)
        (
            metrics.latency_ms,
            metrics.p95_latency_ms,
            metrics.throughput_fps,
            metrics.inference_memory_mb,
            outlier_fraction,
            timing_cv,
            anomaly_flag,
        ) = self._benchmark_latency(
            model, timing_batch, lambda x: self._anomalib_raw(model, x)
        )
        self._last_timing_checks = (outlier_fraction, timing_cv, anomaly_flag)
        (
            metrics.timing_outlier_fraction,
            metrics.timing_cv,
            metrics.timing_anomaly_flag,
        ) = self._last_timing_checks
        metrics.end_to_end_latency_ms = float(
            "nan"
        )  # full data pipeline is intentionally not included in this inference-only benchmark
        metrics.__dict__.update(self._evaluate_anomalib(model, test_loader))
        print(
            f"  image={metrics.image_auroc:.4f} pixel={metrics.pixel_auroc:.4f} latency={metrics.latency_ms:.2f}ms"
        )
        return metrics

    def report(self, results: List[ModelMetrics]) -> None:
        rows = [asdict(item) for item in results]
        rows = [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for row in rows
        ]
        frame = pd.DataFrame(rows)
        frame["environment"] = frame["environment"].apply(
            lambda value: json.dumps(value, sort_keys=True)
        )
        algorithms = {item.algorithm for item in results}
        if len(algorithms) != 1:
            raise ValueError(
                "report() expects results for exactly one algorithm; "
                f"received: {sorted(algorithms)}"
            )
        algorithm = results[0].algorithm
        stem = self._result_dir(algorithm) / f"benchmark_{algorithm}_{self.class_name}"

        frame.to_csv(f"{stem}.csv", index=False)
        with open(f"{stem}.json", "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        summary = frame[
            [
                "algorithm",
                "name",
                "image_ap",
                "image_f1",
                "image_precision",
                "image_recall",
                "image_auroc",
                "image_specificity",
                "pixel_ap",
                "pixel_f1",
                "pixel_precision",
                "pixel_recall",
                "pixel_auroc",
                "pixel_aupro",
                "pixel_dice",
                "pixel_iou",
                "training_time_s",
                "latency_ms",
                "p95_latency_ms",
                "throughput_fps",
                "state_dict_size_mb",
                "training_memory_mb",
                "inference_memory_mb",
                "end_to_end_latency_ms",
                "timing_outlier_fraction",
                "timing_cv",
                "timing_anomaly_flag",
            ]
        ].copy()
        with open(f"{stem}.txt", "w", encoding="utf-8") as handle:
            handle.write("AnomaVision vs Anomalib Benchmark\n")
            handle.write("=" * 80 + "\n\n")
            config = rows[0]
            handle.write("Benchmark Configuration\n")
            handle.write("-" * 80 + "\n")
            handle.write(f"Backbone      : {config['backbone']}\n")
            handle.write(f"Image size    : {config['image_size']}\n")
            handle.write(f"Layer indices : {config['layer_indices']}\n")
            handle.write(f"Layers        : {config['layers']}\n")
            handle.write(f"N features    : {config['n_features']}\n\n")
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
                    (
                        f"<td>{row[k]:.4f}</td>"
                        if isinstance(row[k], float)
                        else f"<td>{row[k]}</td>"
                    )
                    for k in [
                        "algorithm",
                        "name",
                        "image_ap",
                        "image_f1",
                        "image_auroc",
                        "image_specificity",
                        "pixel_ap",
                        "pixel_f1",
                        "pixel_auroc",
                        "pixel_aupro",
                        "pixel_dice",
                        "pixel_iou",
                        "training_time_s",
                        "latency_ms",
                        "p95_latency_ms",
                        "throughput_fps",
                        "state_dict_size_mb",
                        "training_memory_mb",
                        "inference_memory_mb",
                        "timing_anomaly_flag",
                    ]
                )
                + "</tr>"
            )
        config = rows[0]
        html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AnomaVision Benchmark - {self.class_name}</title><style>body{{font-family:Inter,Arial,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}}main{{max-width:1400px;margin:40px auto;padding:24px}}h1{{margin-bottom:6px}}.sub{{color:#94a3b8;margin-bottom:28px}}.card{{background:#111827;border:1px solid #334155;border-radius:16px;padding:20px;margin-bottom:24px;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:12px 14px;border-bottom:1px solid #334155;text-align:left}}th{{color:#93c5fd;background:#1e293b;position:sticky;top:0}}tr:hover{{background:#1e293b}}.pill{{display:inline-block;padding:6px 10px;border-radius:999px;background:#1e293b;margin-right:6px}}.config{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}}.config span{{display:block;color:#94a3b8;font-size:12px;margin-bottom:5px}}.config strong{{font-size:15px}}</style></head><body><main><h1>🔬 AnomaVision Benchmark</h1><div class="sub">{self.class_name} · {config['image_size']} · {self.device}</div><div class="card"><span class="pill">PaDiM</span><span class="pill">PatchCore</span><span class="pill">EfficientAD</span><span class="pill">AnomaVision vs Anomalib</span></div><div class="card"><h2>Benchmark Configuration</h2><div class="config"><div><span>Backbone</span><strong>{config['backbone']}</strong></div><div><span>Image size</span><strong>{config['image_size']}</strong></div><div><span>Layer indices</span><strong>{config['layer_indices']}</strong></div><div><span>Layers</span><strong>{config['layers']}</strong></div><div><span>N features</span><strong>{config['n_features']}</strong></div></div></div><div class="card"><table><thead><tr><th>Algorithm</th><th>Implementation</th><th>Image AP</th><th>Image F1</th><th>Image AUROC</th><th>Image Specificity</th><th>Pixel AP</th><th>Pixel F1</th><th>Pixel AUROC</th><th>AUPRO</th><th>Dice</th><th>IoU</th><th>Fit Time (s)</th><th>Latency (ms)</th><th>P95 (ms)</th><th>FPS</th><th>Model (MB)</th><th>Train RAM/VRAM (MB)</th><th>Inference RAM/VRAM (MB)</th><th>Timing anomaly</th></tr></thead><tbody>{''.join(html_rows)}</tbody></table></div></main></body></html>"""
        with open(f"{stem}.html", "w", encoding="utf-8") as handle:
            handle.write(html)
        print("\nReports:")
        for suffix in ("csv", "json", "txt", "html"):
            print(f"  {stem}.{suffix}")

    def run(self, algorithms: List[str]) -> None:
        for algorithm in algorithms:
            # Save each algorithm independently under its own result directory.
            results = [self.benchmark_anomavision(algorithm)]
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
    parser.add_argument("--class_name", choices=MVTec_CLASSES)
    parser.add_argument(
        "--all_classes",
        action="store_true",
        help="Run the benchmark for all available MVTec classes in the dataset.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Also save model state dictionaries as .pt files (disabled by default).",
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    args = parser.parse_args()

    if args.class_name and args.all_classes:
        parser.error("Use either --class_name or --all_classes, not both.")

    if not args.class_name and not args.all_classes:
        parser.error("You must specify either --class_name or --all_classes.")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_dir():
        parser.error(f"Dataset path does not exist: {dataset_path}")

    if args.all_classes:
        classes = [
            class_name
            for class_name in MVTec_CLASSES
            if (dataset_path / class_name).is_dir()
        ]

        if not classes:
            parser.error(f"No supported MVTec classes found under: {dataset_path}")

        print("\n" + "=" * 80)
        print("AnomaVision vs Anomalib — ALL CLASSES")
        print("=" * 80)
        print(f"Dataset  : {dataset_path}")
        print(f"Classes  : {len(classes)}")
        print(f"Algorithms: {', '.join(args.algorithms)}")
        print(f"Backbone : {ANOMA_BACKBONE[BACKBONE]}")
        print(f"Device   : {args.device}")
        print("\nClasses:")
        for class_name in classes:
            print(f"  - {class_name}")
        print("=" * 80)

        for index, class_name in enumerate(classes, start=1):
            print("\n" + "#" * 80)
            print(f"CLASS {index}/{len(classes)}: {class_name}")
            print("#" * 80)

            BenchmarkRunner(
                str(dataset_path),
                class_name,
                args.device,
                args.seed,
                args.save_models,
            ).run(args.algorithms)

            gc.collect()
            set_seed(args.seed)

        print("\n" + "=" * 80)
        print("ALL CLASS BENCHMARKS COMPLETED")
        print("=" * 80)
    else:
        BenchmarkRunner(
            str(dataset_path),
            args.class_name,
            args.device,
            args.seed,
            args.save_models,
        ).run(args.algorithms)


if __name__ == "__main__":
    main()
