"""Production Autopilot for calibrated, hardware-aware anomaly deployment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
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


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select, calibrate, profile, and package a production anomaly model.",
        add_help=add_help,
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--padim_model", type=str, default=None, help="PaDiM model artifact.")
    parser.add_argument("--patchcore_model", type=str, default=None, help="PatchCore model artifact.")
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
    for batch_index, (batch, _, labels, masks) in enumerate(dataloader):
        batch = batch.to(device)
        measure = batch_index < max(1, timing_batches)
        start = time.perf_counter() if measure else 0.0
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
    masks_np = np.asarray(all_masks, dtype=np.float32)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]

    if len(np.unique(labels_np)) > 1:
        threshold, threshold_f1 = find_optimal_threshold(labels_np, scores_np)
    else:
        threshold, threshold_f1 = float(np.median(scores_np)), 0.0
    metrics = compute_metrics(labels_np, scores_np, thresh=threshold)
    image_auroc = metrics.get("auc_score") if len(np.unique(labels_np)) > 1 else None

    localization = {
        "available": bool(len(maps_np) == len(labels_np) and maps_np.ndim == 3),
        "pixel_auroc": None,
        "anomaly_non_empty_fraction": None,
        "normal_false_positive_fraction": None,
        "anomaly_mean_mask_area_fraction": None,
        "normal_mean_mask_area_fraction": None,
    }
    if localization["available"] and masks_np.shape == maps_np.shape and np.unique(masks_np).size > 1:
        try:
            localization["pixel_auroc"] = float(roc_auc_score(masks_np.reshape(-1) > 0.5, maps_np.reshape(-1)))
        except ValueError:
            pass
        loc_masks = make_localization_mask(maps_np, (scores_np >= threshold).astype(np.uint8)).astype(bool)
        flat = loc_masks.reshape(len(loc_masks), -1)
        non_empty = flat.any(axis=1)
        area = flat.mean(axis=1)
        anomaly_idx = labels_np == 1
        normal_idx = labels_np == 0
        localization["anomaly_non_empty_fraction"] = float(non_empty[anomaly_idx].mean()) if anomaly_idx.any() else None
        localization["normal_false_positive_fraction"] = float(non_empty[normal_idx].mean()) if normal_idx.any() else None
        localization["anomaly_mean_mask_area_fraction"] = float(area[anomaly_idx].mean()) if anomaly_idx.any() else None
        localization["normal_mean_mask_area_fraction"] = float(area[normal_idx].mean()) if normal_idx.any() else None

    batch_size = max(1, int(dataloader.batch_size or 1))
    median_ms = float(np.median(timings) * 1000.0 / batch_size) if timings else 0.0
    p95_ms = float(np.percentile(timings, 95) * 1000.0 / batch_size) if timings else 0.0
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_format": Path(model_path).suffix.lower(),
        "threshold": float(threshold),
        "threshold_f1": float(threshold_f1),
        "metrics": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()},
        "metrics": {**{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()}, "image_auroc": image_auroc, "pixel_auroc": localization["pixel_auroc"]},
        "latency_ms": {"median": median_ms, "p95": p95_ms},
        "throughput_images_per_second": float(1000.0 / median_ms) if median_ms > 0 else 0.0,
        "localization": localization,
        "samples": int(len(labels_np)),
    }


def _select(results: Dict[str, Dict[str, Any]], target_latency_ms: Optional[float]) -> str:
    eligible = results
    if target_latency_ms is not None:
        eligible = {n: r for n, r in results.items() if r["latency_ms"]["p95"] <= target_latency_ms}
    if not eligible:
        eligible = results
    return max(eligible, key=lambda n: (eligible[n]["metrics"].get("image_auroc") or 0.0, -eligible[n]["latency_ms"]["p95"]))


def _write_report(manifest: Dict[str, Any], output_dir: Path) -> None:
    selected = manifest["selected_model"]
    lines = [
        "# AnomaVision Production Autopilot Report",
        "",
        f"**Selected model:** `{selected}`",
        f"**Class:** `{manifest['dataset']['class_name']}`",
        f"**Samples:** `{manifest['dataset']['samples']}`",
        "",
        "| Model | Image AUROC | Pixel AUROC | Median ms | P95 ms | Threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in manifest["candidates"].items():
        lines.append(
            f"| {name} | {result['metrics'].get('image_auroc', 'N/A')} | {result['metrics'].get('pixel_auroc', 'N/A')} | "
            f"{result['latency_ms']['median']:.2f} | {result['latency_ms']['p95']:.2f} | {result['threshold']:.6f} |"
        )
    lines.extend(["", f"**Selected artifact:** `{manifest['selected_artifact']}`", ""])
    (output_dir / "localization_report.md").write_text("\n".join(lines), encoding="utf-8")


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

    candidates = {}
    for name, model_path in (
        ("padim", args.padim_model),
        ("patchcore", args.patchcore_model),
        ("efficientad", args.efficientad_model),
    ):
        if model_path:
            candidates[name] = _profile_model(model_path, dataloader, device, args.warmup, args.timing_batches)
    if not candidates:
        raise ValueError("Provide at least one of --padim_model, --patchcore_model, or --efficientad_model.")

    selected = _select(candidates, args.target_latency_ms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_source = Path(candidates[selected]["model_path"])
    packaged_model = output_dir / f"model{selected_source.suffix}"
    shutil.copy2(selected_source, packaged_model)

    # EfficientAD ONNX inference requires its calibrated training sidecar.
    packaged_sidecar = None
    if selected == "efficientad":
        sidecar_candidates = [selected_source.with_suffix(".pth"), selected_source.parent / "model.pth"]
        for sidecar in sidecar_candidates:
            if sidecar.exists():
                packaged_sidecar = output_dir / sidecar.name
                shutil.copy2(sidecar, packaged_sidecar)
                break

    manifest = {
        "schema_version": 3,
        "selected_model": selected,
        "selected_artifact": packaged_model.name,
        "calibration_artifact": packaged_sidecar.name if packaged_sidecar else None,
        "dataset": {"path": str(Path(dataset_path).resolve()), "class_name": class_name, "samples": len(dataset)},
        "preprocessing": {
            "resize": cfg.get("resize", 224), "crop_size": cfg.get("crop_size", 224),
            "normalize": cfg.get("normalize", True), "mean": cfg.get("norm_mean"), "std": cfg.get("norm_std"),
        },
        "candidates": candidates,
        "target_latency_ms": args.target_latency_ms,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "device": device},
    }
    (output_dir / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(manifest, output_dir)
    return manifest


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or create_parser().parse_args()
    manifest = run(args)
    print(json.dumps({"selected_model": manifest["selected_model"], "output_dir": str(Path(args.output_dir).resolve())}, indent=2))


if __name__ == "__main__":
    main()
