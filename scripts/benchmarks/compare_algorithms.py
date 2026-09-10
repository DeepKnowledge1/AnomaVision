"""Benchmark AnomaVision against a local Anomalib checkout.

Algorithms: PaDiM, PatchCore and EfficientAD.

The benchmark keeps the evaluation protocol common where the algorithm allows it:
MVTec AD official split, 224x224 inputs, ImageNet normalization, CPU/CUDA selected
by the CLI, deterministic seed, raw sklearn AUROC, batch-1 latency, P95 latency,
throughput, memory and serialized state size.

Anomalib is intentionally loaded from a sibling source checkout, not PyPI.
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

ROOT = Path(__file__).resolve().parents[2]
ANOMALIB_ROOT = ROOT.parent / "anomalib"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
TIMING_BATCH_SIZE = 1
WARMUP_ITERS = 10
TIMING_ITERS = 100
SEED = 42
BACKBONE = "resnet18"

ALGORITHMS = ("padim", "patchcore", "efficientad")
MVTec_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


@dataclass
class ModelMetrics:
    algorithm: str
    implementation: str
    image_auroc: float = float("nan")
    pixel_auroc: float = float("nan")
    fit_time_s: float = float("nan")
    latency_ms: float = float("nan")
    p95_latency_ms: float = float("nan")
    throughput_fps: float = float("nan")
    model_size_mb: float = float("nan")
    fit_memory_mb: float = float("nan")
    inference_memory_mb: float = float("nan")
    train_images: int = 0
    test_images: int = 0
    batch_size: int = BATCH_SIZE
    device: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)


def set_seed(seed: int) -> None:
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


def add_local_anomalib_to_path() -> None:
    for candidate in (ANOMALIB_ROOT / "src", ANOMALIB_ROOT):
        if (candidate / "anomalib").is_dir():
            value = str(candidate.resolve())
            if value not in sys.path:
                sys.path.insert(0, value)
            return
    raise ModuleNotFoundError(f"Local Anomalib checkout not found at {ANOMALIB_ROOT}")


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def batch_value(batch: Any, name: str) -> Any:
    aliases = {"image": ("image",), "label": ("gt_label", "label"), "mask": ("gt_mask", "mask")}
    for key in aliases[name]:
        if hasattr(batch, key):
            return getattr(batch, key)
        if isinstance(batch, dict) and key in batch:
            return batch[key]
    if isinstance(batch, (tuple, list)):
        return batch[{"image": 0, "label": 1, "mask": 2}[name]]
    raise TypeError(f"Cannot read {name} from batch type {type(batch)!r}")


def normalize_map(value: Any) -> torch.Tensor:
    value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"Expected anomaly map (N,H,W), got {tuple(value.shape)}")
    return value


def outputs(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    score = amap = None
    for key in ("pred_score", "anomaly_score", "image_score"):
        if hasattr(output, key):
            score = getattr(output, key); break
        if isinstance(output, dict) and key in output:
            score = output[key]; break
    for key in ("anomaly_map", "pred_mask", "score_map"):
        if hasattr(output, key):
            amap = getattr(output, key); break
        if isinstance(output, dict) and key in output:
            amap = output[key]; break
    if score is None or amap is None:
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            score, amap = output[:2]
        else:
            raise TypeError(f"Cannot extract predictions from {type(output)!r}")
    score = score if isinstance(score, torch.Tensor) else torch.as_tensor(score)
    if score.ndim > 1:
        score = score.reshape(score.shape[0], -1).amax(1)
    return score, normalize_map(amap)


def resize_maps(maps: np.ndarray, mode: str) -> np.ndarray:
    maps = np.asarray(maps)
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    if maps.ndim != 3:
        raise ValueError(f"Expected (N,H,W), got {maps.shape}")
    if maps.shape[-2:] == IMAGE_SIZE:
        return maps
    x = torch.from_numpy(maps.astype(np.float32)).unsqueeze(1)
    kwargs = {"size": IMAGE_SIZE, "mode": mode}
    if mode == "bilinear":
        kwargs["align_corners"] = False
    return F.interpolate(x, **kwargs)[:, 0].numpy()


def auroc(labels: np.ndarray, scores: np.ndarray, masks: np.ndarray, maps: np.ndarray) -> Tuple[float, float]:
    labels = np.asarray(labels).reshape(-1).astype(np.uint8)
    scores = np.asarray(scores).reshape(-1)
    image = float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else float("nan")
    masks = resize_maps(masks, "nearest")
    maps = resize_maps(maps, "bilinear")
    pixel_labels = masks.reshape(-1).astype(np.uint8)
    pixel_scores = maps.reshape(-1)
    pixel = float(roc_auc_score(pixel_labels, pixel_scores)) if np.unique(pixel_labels).size > 1 else float("nan")
    return image, pixel


def env(device: torch.device, seed: int) -> Dict[str, str]:
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": str(seed), "input": "224x224", "normalization": "ImageNet",
    }


class Runner:
    def __init__(self, dataset_path: str, class_name: str, device: str, seed: int) -> None:
        self.dataset_path = Path(dataset_path)
        self.class_name = class_name
        self.seed = seed
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = torch.device(device)
        self.out = Path("benchmark_results")
        self.out.mkdir(exist_ok=True)

    def reset(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1024**2
        return psutil.Process().memory_info().rss / 1024**2

    def av_data(self):
        from anomavision import MVTecDataset
        train = MVTecDataset(self.dataset_path, self.class_name, is_train=True,
                             resize=IMAGE_SIZE, crop_size=IMAGE_SIZE, normalize=True)
        test = MVTecDataset(self.dataset_path, self.class_name, is_train=False,
                            resize=IMAGE_SIZE, crop_size=IMAGE_SIZE, normalize=True)
        return train, test

    def ab_data(self, train_batch: int = BATCH_SIZE):
        add_local_anomalib_to_path()
        from anomalib.data import MVTecAD
        kwargs = {"root": str(self.dataset_path), "category": self.class_name,
                  "train_batch_size": train_batch, "eval_batch_size": BATCH_SIZE,
                  "num_workers": 0, "seed": self.seed}
        params = inspect.signature(MVTecAD).parameters
        dm = MVTecAD(**{k: v for k, v in kwargs.items() if k in params})
        dm.setup()
        return dm

    def latency(self, model, batch: torch.Tensor, call) -> Tuple[float, float, float, float]:
        model.eval(); batch = batch.to(self.device); self.reset()
        with torch.inference_mode():
            for _ in range(WARMUP_ITERS): call(batch)
            sync(self.device); times = []
            for _ in range(TIMING_ITERS):
                sync(self.device); t = time.perf_counter(); call(batch); sync(self.device)
                times.append(time.perf_counter() - t)
        x = np.asarray(times)
        return float(x.mean()*1000), float(np.percentile(x, 95)*1000), float(1/x.mean()), self.memory()

    def save_size(self, model, filename: str) -> float:
        path = self.out / filename
        torch.save(model.state_dict(), path)
        size = path.stat().st_size / 1024**2
        path.unlink(missing_ok=True)
        return size

    def evaluate(self, model, loader, anomalib: bool) -> Tuple[float, float]:
        labels, scores, masks, maps = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                images = batch_value(batch, "image").to(self.device) if anomalib else batch_value(batch, "image")
                label = batch_value(batch, "label"); mask = batch_value(batch, "mask")
                score, amap = outputs(model(images))
                labels.append(to_numpy(label)); scores.append(to_numpy(score))
                masks.append(to_numpy(mask)); maps.append(to_numpy(amap))
        return auroc(np.concatenate(labels), np.concatenate(scores), np.concatenate(masks), np.concatenate(maps))

    def engine(self):
        add_local_anomalib_to_path()
        from anomalib.engine import Engine
        class Mixin:
            def _setup_anomalib_callbacks(self):
                from lightning.pytorch.callbacks import ModelCheckpoint
                super()._setup_anomalib_callbacks()
                self._cache.args["callbacks"] = [c for c in self._cache.args["callbacks"] if not isinstance(c, ModelCheckpoint)]
        return type("BenchmarkEngine", (Mixin, Engine), {})(
            max_epochs=1, accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1, logger=False, enable_progress_bar=False, enable_checkpointing=False)

    def anomavision(self, algorithm: str) -> ModelMetrics:
        from anomavision import EfficientAD, Padim, PatchCore
        train, test = self.av_data()
        train_loader = DataLoader(train, batch_size=(1 if algorithm == "efficientad" else BATCH_SIZE), shuffle=False, num_workers=0)
        test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        cfg = {"backbone": BACKBONE, "input": "224x224", "normalization": "ImageNet"}
        if algorithm == "padim":
            model = Padim(backbone=BACKBONE, device=self.device, feat_dim=50, layer_indices=[0]); cfg.update(layer="layer1", n_features=50)
        elif algorithm == "patchcore":
            model = PatchCore(backbone=BACKBONE, device=self.device, layer_indices=[0], coreset_ratio=0.02,
                              max_memory_patches=2048, patch_grid=14, n_neighbors=1, coreset_seed=self.seed)
            cfg.update(layer="layer1", coreset_ratio=0.02, max_memory_patches=2048, patch_grid=14)
        else:
            model = EfficientAD(backbone=BACKBONE, device=self.device, epochs=5, learning_rate=1e-3)
            cfg.update(epochs=5, learning_rate=1e-3, layer="layer1")
        model = model.to(self.device)
        self.reset(); t = time.perf_counter(); model.fit(train_loader); sync(self.device)
        fit_time = time.perf_counter() - t; fit_mem = self.memory()
        timing = next(iter(test_loader)); batch = batch_value(timing, "image")[:1]
        lat, p95, fps, inf_mem = self.latency(model, batch, model.predict)
        image, pixel = self.evaluate(model, test_loader, False)
        return ModelMetrics(algorithm, "AnomaVision", image, pixel, fit_time, lat, p95, fps,
                            self.save_size(model, f"av_{algorithm}_tmp.pt"), fit_mem, inf_mem,
                            len(train), len(test), 1 if algorithm == "efficientad" else BATCH_SIZE,
                            str(self.device), cfg, env(self.device, self.seed))

    def anomalib(self, algorithm: str) -> ModelMetrics:
        add_local_anomalib_to_path()
        from anomalib.models import Padim, Patchcore, EfficientAd
        train_batch = 1 if algorithm == "efficientad" else BATCH_SIZE
        dm = self.ab_data(train_batch)
        train_loader, test_loader = dm.train_dataloader(), dm.test_dataloader()
        if algorithm == "padim":
            model = Padim(backbone=BACKBONE, layers=["layer1"], pre_trained=True, n_features=50)
            cfg = {"backbone": BACKBONE, "layer": "layer1", "n_features": 50}
        elif algorithm == "patchcore":
            model = Patchcore(backbone=BACKBONE, layers=["layer1"], pre_trained=True,
                              coreset_sampling_ratio=0.02, num_neighbors=1, precision="float32")
            cfg = {"backbone": BACKBONE, "layer": "layer1", "coreset_ratio": 0.02, "neighbors": 1, "precision": "float32"}
        else:
            model = EfficientAd(model_size="s", lr=1e-3, padding=False, pad_maps=True)
            cfg = {"model_size": "s", "lr": 1e-3, "padding": False, "pad_maps": True, "train_batch_size": 1}
        model = model.to(self.device)
        self.reset(); t = time.perf_counter(); self.engine().fit(model=model, datamodule=dm); sync(self.device)
        fit_time = time.perf_counter() - t; fit_mem = self.memory()
        first = next(iter(test_loader)); batch = batch_value(first, "image")[:1]
        lat, p95, fps, inf_mem = self.latency(model, batch, model)
        image, pixel = self.evaluate(model, test_loader, True)
        return ModelMetrics(algorithm, "Anomalib", image, pixel, fit_time, lat, p95, fps,
                            self.save_size(model, f"ab_{algorithm}_tmp.pt"), fit_mem, inf_mem,
                            len(train_loader.dataset), len(test_loader.dataset), train_batch,
                            str(self.device), cfg, env(self.device, self.seed))

    def report(self, results: List[ModelMetrics]) -> None:
        rows = [asdict(x) for x in results]
        frame = pd.DataFrame(rows)
        stem = self.out / f"mvtec_{self.class_name}"
        frame.to_csv(f"{stem}.csv", index=False)
        frame.to_json(f"{stem}.json", orient="records", indent=2)

        simple = frame[["algorithm", "implementation", "image_auroc", "pixel_auroc", "fit_time_s",
                        "latency_ms", "p95_latency_ms", "throughput_fps", "model_size_mb",
                        "fit_memory_mb", "inference_memory_mb"]].copy()
        with open(f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write(f"AnomaVision Benchmark | MVTec AD | {self.class_name}\n")
            f.write("=" * 78 + "\n\n")
            f.write(tabulate(simple.round(4), headers="keys", tablefmt="github", showindex=False))
            f.write("\n\nProtocol: 224x224 | ImageNet | ResNet18/layer1 where applicable | seed=42\n")
            f.write("Note: PatchCore fit time is memory-bank/coreset build time; EfficientAD uses batch size 1.\n")

        cards = []
        for alg in ALGORITHMS:
            part = [r for r in results if r.algorithm == alg]
            cards.append(self.html_algorithm(alg, part))
        overall = simple.to_html(index=False, classes="data", float_format=lambda x: f"{x:.4f}")
        html = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AnomaVision Benchmark — {self.class_name}</title><style>
:root{{--bg:#0b1020;--card:#121a2e;--muted:#9aa6bf;--text:#f4f7fb;--line:#27324a;--accent:#62d6ff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#182544 0,#0b1020 45%);color:var(--text);font:15px/1.5 Inter,Segoe UI,Arial,sans-serif}}.wrap{{max-width:1250px;margin:auto;padding:42px 22px 60px}}h1{{font-size:38px;margin:0 0 8px}}h2{{margin-top:34px}}.sub{{color:var(--muted);margin-bottom:28px}}.hero{{padding:28px;border:1px solid var(--line);border-radius:22px;background:rgba(18,26,46,.88);box-shadow:0 20px 60px rgba(0,0,0,.28)}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:22px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px}}.tag{{display:inline-block;padding:4px 9px;border-radius:999px;background:#1c2b49;color:var(--accent);font-size:12px;font-weight:700}}.metric{{font-size:28px;font-weight:800;margin-top:8px}}.muted{{color:var(--muted)}}.winner{{border-color:var(--accent);box-shadow:0 0 0 1px rgba(98,214,255,.12)}}table{{width:100%;border-collapse:collapse;margin-top:14px;background:var(--card);border-radius:14px;overflow:hidden}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);font-size:12px;text-transform:uppercase}}.section{{margin-top:28px}}.pill{{color:#b9c5da;font-size:13px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}h1{{font-size:30px}}table{{display:block;overflow:auto}}}}
</style></head><body><div class="wrap"><div class="hero"><div class="tag">VISION ANOMALY DETECTION</div><h1>AnomaVision Benchmark</h1><div class="sub">MVTec AD · <b>{self.class_name}</b> · {self.device} · seed {self.seed}</div><div class="pill">PaDiM · PatchCore · EfficientAD &nbsp;|&nbsp; AnomaVision vs Anomalib</div></div><div class="grid">{''.join(cards)}</div><div class="section"><h2>All results</h2>{overall}</div><div class="section"><h2>Benchmark protocol</h2><div class="card"><b>Input</b> 224×224 · ImageNet normalization · ResNet18/layer1 where applicable<br><b>Timing</b> 10 warmups + 100 batch-1 iterations · mean + P95 · data loading excluded<br><b>Metrics</b> raw sklearn image/pixel AUROC · throughput · memory · serialized state size<br><b>Fairness</b> Anomalib is loaded from the local sibling checkout; ONNX is intentionally outside this benchmark.<br><b>Important</b> PatchCore fit time means memory-bank/coreset construction. EfficientAD uses batch size 1 for training.</div></div></div></body></html>'''
        (Path(f"{stem}.html")).write_text(html, encoding="utf-8")
        print("\n" + "=" * 78)
        print(f"Benchmark complete: {self.class_name}")
        print("=" * 78)
        for r in results:
            print(f"{r.algorithm:<12} {r.implementation:<12} image={r.image_auroc:.4f} pixel={r.pixel_auroc:.4f} latency={r.latency_ms:.2f} ms")
        print(f"\nHTML : {stem}.html")
        print(f"CSV  : {stem}.csv")
        print(f"JSON : {stem}.json")
        print(f"TXT  : {stem}.txt")

    @staticmethod
    def html_algorithm(alg: str, rows: List[ModelMetrics]) -> str:
        best_img = max(r.image_auroc for r in rows); best_lat = min(r.latency_ms for r in rows)
        body = []
        for r in rows:
            cls = "card winner" if r.image_auroc == best_img else "card"
            body.append(f'<div class="{cls}"><span class="tag">{r.implementation}</span><div class="metric">{r.image_auroc:.4f}</div><div class="muted">Image AUROC</div><p>Pixel AUROC <b>{r.pixel_auroc:.4f}</b><br>Latency <b>{r.latency_ms:.2f} ms</b><br>Throughput <b>{r.throughput_fps:.2f} FPS</b><br>Fit <b>{r.fit_time_s:.2f} s</b></p></div>')
        return '<div class="card" style="grid-column:1/-1"><h2 style="margin:0 0 14px">' + alg.title() + '</h2><div class="grid">' + ''.join(body) + '</div></div>'

    def run(self, algorithms: List[str]) -> None:
        results = []
        for alg in algorithms:
            print(f"\n{'='*70}\n{alg.upper()}\n{'='*70}")
            set_seed(self.seed); results.append(self.anomavision(alg)); gc.collect(); set_seed(self.seed); results.append(self.anomalib(alg)); gc.collect()
        self.report(results)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark PaDiM, PatchCore and EfficientAD")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--class_name", choices=MVTec_CLASSES, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    a = p.parse_args()
    Runner(a.dataset_path, a.class_name, a.device, a.seed).run(a.algorithms)


if __name__ == "__main__":
    main()
