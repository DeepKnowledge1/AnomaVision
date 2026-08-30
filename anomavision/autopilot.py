"""Production Autopilot for calibrated, hardware-aware anomaly deployment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import anomavision
from anomavision.config import load_config
from anomavision.general import determine_device
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.utils import compute_metrics, find_optimal_threshold, make_localization_mask


ALGORITHMS = ("padim", "patchcore", "efficientad")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select, calibrate, profile, and package a production anomaly model.",
        add_help=add_help,
    )
    parser.add_argument("--config", type=str, required=True, help="Base AnomaVision config file.")
    parser.add_argument("--dataset_path", type=str, default=None, help="MVTec-style dataset root.")
    parser.add_argument("--class_name", type=str, default=None, help="Dataset class to evaluate.")
    parser.add_argument("--padim_model", type=str, default=None, help="PaDiM model artifact (.pt/.pth/.onnx).")
    parser.add_argument("--patchcore_model", type=str, default=None, help="PatchCore model artifact (.pt/.pth/.onnx).")
    parser.add_argument("--efficientad_model", type=str, default=None, help="EfficientAD model artifact (.pt/.pth/.onnx).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing_batches", type=int, default=20)
    parser.add_argument("--target_latency_ms", type=float, default=None)
    parser.add_argument("--validation_split", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./production_package")
    parser.add_argument("--copy_config", action="store_true", default=True)
    return parser


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.1%}"


def _profile_model(model_path: str, dataloader: DataLoader, device: str, warmup: int, timing_batches: int) -> Dict[str, Any]:
    wrapper = ModelWrapper(model_path, device)
    try:
        first = next(iter(dataloader))
    except StopIteration:
        wrapper.close()
        raise ValueError("The evaluation dataset is empty.")
    first_batch = first[0].to(device)
    for _ in range(max(0, warmup)):
        wrapper.predict(first_batch)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    all_scores, all_maps, all_labels, all_masks = [], [], [], []
    for count, item in enumerate(dataloader):
        batch, _, labels, masks = item
        batch = batch.to(device)
        measure = count < max(1, timing_batches)
        if measure:
            start = time.perf_counter()
        scores, maps = wrapper.predict(batch)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        if measure:
            timings.append(time.perf_counter() - start)
        all_scores.extend(_to_numpy(scores).reshape(-1).tolist())
        if maps is not None:
            all_maps.extend(_to_numpy(maps))
        all_labels.extend(_to_numpy(labels).reshape(-1).tolist())
        all_masks.extend(_to_numpy(masks))
    wrapper.close()

    scores_np = np.asarray(all_scores, dtype=np.float32)
    labels_np = np.asarray(all_labels, dtype=np.int64)
    maps_np = np.asarray(all_maps, dtype=np.float32) if all_maps else np.empty((0, 0, 0), dtype=np.float32)
    threshold, threshold_f1 = (
        find_optimal_threshold(labels_np, scores_np)
        if len(np.unique(labels_np)) > 1
        else (float(np.median(scores_np)), 0.0)
    )
    image_metrics = compute_metrics(labels_np, scores_np, thresh=threshold)
    image_auroc = image_metrics.get("auc_score") if len(np.unique(labels_np)) > 1 else None

    masks_np = np.asarray(all_masks, dtype=np.float32)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    pixel_auroc = None
    localization = {
        "available": bool(len(maps_np) == len(labels_np) and maps_np.ndim == 3),
        "non_empty_fraction": None,
        "mean_mask_area_fraction": None,
        "anomaly_non_empty_fraction": None,
        "normal_false_positive_fraction": None,
        "anomaly_mean_mask_area_fraction": None,
        "normal_mean_mask_area_fraction": None,
        "verdict": "unavailable",
    }
    if localization["available"] and masks_np.shape == maps_np.shape and np.unique(masks_np).size > 1:
        try:
            pixel_auroc = float(roc_auc_score(masks_np.reshape(-1) > 0.5, maps_np.reshape(-1)))
        except ValueError:
            pass
    image_metrics["image_auroc"] = image_auroc
    image_metrics["pixel_auroc"] = pixel_auroc

    if localization["available"]:
        anomaly_labels = (scores_np >= threshold).astype(np.uint8)
        loc_masks = make_localization_mask(maps_np, anomaly_labels).astype(bool)
        non_empty = loc_masks.reshape(len(loc_masks), -1).any(axis=1)
        area = loc_masks.reshape(len(loc_masks), -1).mean(axis=1)
        anomaly_idx, normal_idx = labels_np == 1, labels_np == 0
        localization["non_empty_fraction"] = float(non_empty.mean())
        localization["mean_mask_area_fraction"] = float(area.mean())
        localization["anomaly_non_empty_fraction"] = float(non_empty[anomaly_idx].mean()) if anomaly_idx.any() else None
        localization["normal_false_positive_fraction"] = float(non_empty[normal_idx].mean()) if normal_idx.any() else None
        localization["anomaly_mean_mask_area_fraction"] = float(area[anomaly_idx].mean()) if anomaly_idx.any() else None
        localization["normal_mean_mask_area_fraction"] = float(area[normal_idx].mean()) if normal_idx.any() else None
        localization["verdict"] = (
            "healthy"
            if (localization["normal_false_positive_fraction"] or 0.0) <= 0.10
            else "review false positives"
        ) if pixel_auroc is not None else "maps available; pixel AUROC unavailable"

    batch_size = max(1, dataloader.batch_size or 1)
    median_ms = float(np.median(timings) * 1000 / batch_size) if timings else 0.0
    p95_ms = float(np.percentile(timings, 95) * 1000 / batch_size) if timings else 0.0
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_format": Path(model_path).suffix.lower(),
        "threshold": float(threshold),
        "threshold_f1": float(threshold_f1),
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in image_metrics.items()},
        "latency_ms": {"median": median_ms, "p95": p95_ms},
        "throughput_images_per_second": float(1000.0 / median_ms) if median_ms > 0 else 0.0,
        "localization": localization,
        "samples": int(len(labels_np)),
    }


def _select(results: Dict[str, Dict[str, Any]], target_latency_ms: Optional[float]) -> str:
    eligible = results
    if target_latency_ms is not None:
        eligible = {name: result for name, result in results.items() if result["latency_ms"]["p95"] <= target_latency_ms}
    if not eligible:
        eligible = results
    return max(eligible, key=lambda name: (eligible[name]["metrics"].get("image_auroc") or 0.0, -eligible[name]["latency_ms"]["p95"]))


def _write_report(manifest: Dict[str, Any], output_dir: Path) -> None:
    selected = manifest["selected_model"]
    cards = []
    rows = []
    for name, result in manifest["candidates"].items():
        metrics, loc = result["metrics"], result["localization"]
        active = name == selected
        cards.append(
            f'<article class="model-card {"selected" if active else ""}"><b>{escape(name.upper())}</b>'
            f'<h2>{_format_metric(metrics.get("image_auroc"))}</h2><span>image AUROC</span>'
            f'<p>p95: {result["latency_ms"]["p95"]:.2f} ms · pixel AUROC: {_format_metric(metrics.get("pixel_auroc"))}</p></article>'
        )
        rows.append(
            f'<tr><td>{escape(name)}</td><td>{_format_metric(metrics.get("image_auroc"))}</td>'
            f'<td>{_format_metric(metrics.get("pixel_auroc"))}</td><td>{result["latency_ms"]["median"]:.2f}</td>'
            f'<td>{result["latency_ms"]["p95"]:.2f}</td><td>{_format_percent(loc.get("normal_false_positive_fraction"))}</td>'
            f'<td>{result["threshold"]:.6f}</td></tr>'
        )
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>AnomaVision Production Autopilot</title>
<style>body{{font:15px system-ui;margin:40px;background:#f5f7fb;color:#14213d}}main{{max-width:1100px;margin:auto}}header,.card,.table{{background:white;border-radius:16px;padding:24px;margin-bottom:20px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.model-card{{background:white;border:1px solid #ddd;border-radius:16px;padding:20px;min-width:260px}}.selected{{border:2px solid #315efb}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}</style></head>
<body><main><header><h1>Deployment confidence, before production.</h1><p>Selected model: <b>{escape(selected)}</b> · Class: <b>{escape(str(manifest["dataset"]["class_name"]))}</b></p></header>
<div class="cards">{"".join(cards)}</div><section class="table"><h2>Model comparison</h2><table><tr><th>Model</th><th>Image AUROC</th><th>Pixel AUROC</th><th>Median ms</th><th>P95 ms</th><th>Normal FP</th><th>Threshold</th></tr>{"".join(rows)}</table></section></main></body></html>'''
    (output_dir / "production_autopilot_report.html").write_text(html, encoding="utf-8")
    (output_dir / "localization_report.md").write_text(
        f"# AnomaVision Production Autopilot Report\n\n**Selected model:** `{selected}`\n",
        encoding="utf-8",
    )


def _config_model(cfg: Dict[str, Any], name: str) -> Optional[str]:
    section = cfg.get("autopilot", {}) or {}
    value = section.get(f"{name}_model")
    return str(value) if value else None


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    dataset_path = args.dataset_path or cfg.get("dataset_path") or cfg.get("img_path")
    class_name = args.class_name or cfg.get("class_name")
    if not dataset_path or not class_name:
        raise ValueError("dataset_path and class_name are required in the CLI or config.")
    device = determine_device(args.device)
    dataset = anomavision.MVTecDataset(
        dataset_path, class_name, is_train=False,
        resize=cfg.get("resize", 224), crop_size=cfg.get("crop_size", 224),
        normalize=cfg.get("normalize", True), mean=cfg.get("norm_mean"), std=cfg.get("norm_std"),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    paths = {
        "padim": args.padim_model or _config_model(cfg, "padim"),
        "patchcore": args.patchcore_model or _config_model(cfg, "patchcore"),
        "efficientad": args.efficientad_model or _config_model(cfg, "efficientad"),
    }
    candidates = {
        name: _profile_model(path, dataloader, device, args.warmup, args.timing_batches)
        for name, path in paths.items() if path
    }
    if not candidates:
        raise ValueError("Provide at least one model: --padim_model, --patchcore_model, or --efficientad_model.")

    selected = _select(candidates, args.target_latency_ms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_source = Path(candidates[selected]["model_path"])
    packaged_model = output_dir / f"model{selected_source.suffix}"
    shutil.copy2(selected_source, packaged_model)

    manifest = {
        "schema_version": 3,
        "selected_model": selected,
        "selected_artifact": packaged_model.name,
        "dataset": {"path": str(Path(dataset_path).resolve()), "class_name": class_name, "samples": len(dataset)},
        "preprocessing": {"resize": cfg.get("resize", 224), "crop_size": cfg.get("crop_size", 224), "normalize": cfg.get("normalize", True), "mean": cfg.get("norm_mean"), "std": cfg.get("norm_std")},
        "candidates": candidates,
        "target_latency_ms": args.target_latency_ms,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "device": device},
    }
    (output_dir / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_config:
        shutil.copy2(args.config, output_dir / Path(args.config).name)
    _write_report(manifest, output_dir)
    return manifest


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or create_parser().parse_args()
    manifest = run(args)
    print(json.dumps({"selected_model": manifest["selected_model"], "output_dir": str(Path(args.output_dir).resolve())}, indent=2))


if __name__ == "__main__":
    main()
