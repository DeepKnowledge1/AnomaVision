"""Production Autopilot for calibrated, hardware-aware anomaly deployment."""

from __future__ import annotations

import argparse
import html
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
    parser = argparse.ArgumentParser(description="Select, calibrate, profile, and package a production anomaly model.", add_help=add_help)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--padim_model", type=str, default=None)
    parser.add_argument("--patchcore_model", type=str, default=None)
    parser.add_argument("--efficientad_model", type=str, default=None)
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
    timings: list[float] = []
    all_scores: list[float] = []
    all_maps: list[np.ndarray] = []
    all_labels: list[int] = []
    all_masks: list[np.ndarray] = []
    try:
        first_batch = next(iter(dataloader))[0].to(device)
        for _ in range(max(0, warmup)):
            wrapper.predict(first_batch)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

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
                all_maps.extend(list(_to_numpy(maps)))
            all_labels.extend(_to_numpy(labels).reshape(-1).astype(int).tolist())
            all_masks.extend(list(_to_numpy(masks)))
    finally:
        wrapper.close()

    scores = np.asarray(all_scores, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    maps = np.asarray(all_maps, dtype=np.float32) if all_maps else np.empty((0, 0, 0), dtype=np.float32)
    masks = np.asarray(all_masks, dtype=np.float32)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if len(np.unique(labels)) > 1:
        threshold, threshold_f1 = find_optimal_threshold(labels, scores)
    else:
        threshold, threshold_f1 = (float(np.median(scores)) if len(scores) else 0.0), 0.0
    metrics = compute_metrics(labels, scores, thresh=threshold) if len(labels) else {}
    image_auroc = float(metrics["auc_score"]) if metrics.get("auc_score") is not None else None

    localization = {"available": bool(len(maps) == len(labels) and maps.ndim == 3), "pixel_auroc": None,
                    "anomaly_non_empty_fraction": None, "normal_false_positive_fraction": None,
                    "anomaly_mean_mask_area_fraction": None, "normal_mean_mask_area_fraction": None}
    if localization["available"] and masks.shape == maps.shape and np.unique(masks).size > 1:
        try:
            localization["pixel_auroc"] = float(roc_auc_score(masks.reshape(-1) > 0.5, maps.reshape(-1)))
        except ValueError:
            pass
        loc_masks = make_localization_mask(maps, (scores >= threshold).astype(np.uint8)).astype(bool)
        area = loc_masks.reshape(len(loc_masks), -1).mean(axis=1)
        non_empty = loc_masks.reshape(len(loc_masks), -1).any(axis=1)
        anomaly = labels == 1
        normal = labels == 0
        if anomaly.any():
            localization["anomaly_non_empty_fraction"] = float(non_empty[anomaly].mean())
            localization["anomaly_mean_mask_area_fraction"] = float(area[anomaly].mean())
        if normal.any():
            localization["normal_false_positive_fraction"] = float(non_empty[normal].mean())
            localization["normal_mean_mask_area_fraction"] = float(area[normal].mean())

    batch_size = max(1, int(dataloader.batch_size or 1))
    median_ms = float(np.median(timings) * 1000.0 / batch_size) if timings else 0.0
    p95_ms = float(np.percentile(timings, 95) * 1000.0 / batch_size) if timings else 0.0
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_format": Path(model_path).suffix.lower(),
        "threshold": float(threshold),
        "threshold_f1": float(threshold_f1),
        "metrics": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()},
        "image_auroc": image_auroc,
        "pixel_auroc": localization["pixel_auroc"],
        "latency_ms": {"median": median_ms, "p95": p95_ms},
        "throughput_images_per_second": float(1000.0 / median_ms) if median_ms > 0 else 0.0,
        "localization": localization,
        "samples": int(len(labels)),
    }


def _select(results: Dict[str, Dict[str, Any]], target_latency_ms: Optional[float]) -> str:
    eligible = results
    if target_latency_ms is not None:
        eligible = {n: r for n, r in results.items() if r["latency_ms"]["p95"] <= target_latency_ms}
    if not eligible:
        eligible = results
    return max(eligible, key=lambda n: (eligible[n].get("image_auroc") or 0.0, -eligible[n]["latency_ms"]["p95"]))


def _write_report(manifest: Dict[str, Any], output_dir: Path) -> None:
    rows = []
    for name, result in manifest["candidates"].items():
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{result.get('image_auroc', 'N/A')}</td>"
                    f"<td>{result.get('pixel_auroc', 'N/A')}</td><td>{result['latency_ms']['median']:.2f}</td>"
                    f"<td>{result['latency_ms']['p95']:.2f}</td><td>{result['threshold']:.6f}</td></tr>")
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>AnomaVision Production Autopilot Report</title>
<style>body{{font-family:Arial,sans-serif;margin:40px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:8px;text-align:left}}th{{background:#eee}}.selected{{font-size:20px;font-weight:bold}}</style></head><body>
<h1>AnomaVision Production Autopilot Report</h1><p class='selected'>Selected model: {html.escape(manifest['selected_model'])}</p>
<p>Class: {html.escape(str(manifest['dataset']['class_name']))} &nbsp; Samples: {manifest['dataset']['samples']}</p>
<table><thead><tr><th>Model</th><th>Image AUROC</th><th>Pixel AUROC</th><th>Median ms</th><th>P95 ms</th><th>Threshold</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>Selected artifact: <code>{html.escape(manifest['selected_artifact'])}</code></p>
<p>Target latency: {manifest['target_latency_ms'] if manifest['target_latency_ms'] is not None else 'not set'} ms</p>
</body></html>"""
    (output_dir / "production_autopilot_report.html").write_text(document, encoding="utf-8")
    (output_dir / "localization_report.md").write_text(
        "# AnomaVision Production Autopilot Report\n\n" +
        f"**Selected model:** `{manifest['selected_model']}`\n\n" +
        "| Model | Image AUROC | Pixel AUROC | Median ms | P95 ms | Threshold |\n|---|---:|---:|---:|---:|---:|\n" +
        "\n".join(f"| {n} | {r.get('image_auroc', 'N/A')} | {r.get('pixel_auroc', 'N/A')} | {r['latency_ms']['median']:.2f} | {r['latency_ms']['p95']:.2f} | {r['threshold']:.6f} |" for n, r in manifest['candidates'].items()) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    dataset_path = args.dataset_path or cfg.get("dataset_path") or cfg.get("img_path")
    class_name = args.class_name or cfg.get("class_name")
    if not dataset_path or not class_name:
        raise ValueError("dataset_path and class_name are required in the CLI or config.")

    device = determine_device(args.device)
    dataset = anomavision.MVTecDataset(dataset_path, class_name, is_train=False,
        resize=cfg.get("resize", 224), crop_size=cfg.get("crop_size", 224),
        normalize=cfg.get("normalize", True), mean=cfg.get("norm_mean"), std=cfg.get("norm_std"))
    dataloader = DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=False,
                            num_workers=max(0, args.num_workers), pin_memory=(device.startswith("cuda")))

    candidates: Dict[str, Dict[str, Any]] = {}
    for name, model_path in (("padim", args.padim_model), ("patchcore", args.patchcore_model), ("efficientad", args.efficientad_model)):
        if model_path:
            candidates[name] = _profile_model(model_path, dataloader, device, args.warmup, args.timing_batches)
    if not candidates:
        raise ValueError("Provide at least one model: --padim_model, --patchcore_model, or --efficientad_model.")

    selected = _select(candidates, args.target_latency_ms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(candidates[selected]["model_path"])
    packaged_model = output_dir / f"model{source.suffix}"
    shutil.copy2(source, packaged_model)

    packaged_sidecar = None
    if selected == "efficientad":
        for sidecar in (source.with_suffix(".pth"), source.parent / "model.pth"):
            if sidecar.exists():
                packaged_sidecar = output_dir / sidecar.name
                shutil.copy2(sidecar, packaged_sidecar)
                break

    if args.copy_config:
        shutil.copy2(args.config, output_dir / Path(args.config).name)
    manifest = {
        "schema_version": 4, "selected_model": selected, "selected_artifact": packaged_model.name,
        "calibration_artifact": packaged_sidecar.name if packaged_sidecar else None,
        "dataset": {"path": str(Path(dataset_path).resolve()), "class_name": class_name, "samples": len(dataset)},
        "preprocessing": {"resize": cfg.get("resize", 224), "crop_size": cfg.get("crop_size", 224),
                          "normalize": cfg.get("normalize", True), "mean": cfg.get("norm_mean"), "std": cfg.get("norm_std")},
        "candidates": candidates, "target_latency_ms": args.target_latency_ms,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "device": device},
    }
    (output_dir / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(manifest, output_dir)
    return manifest


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or create_parser().parse_args()
    manifest = run(args)
    print(json.dumps({"selected_model": manifest["selected_model"], "output_dir": str(Path(args.output_dir).resolve()),
                      "report": str(Path(args.output_dir).resolve() / "production_autopilot_report.html")}, indent=2))


if __name__ == "__main__":
    main()
