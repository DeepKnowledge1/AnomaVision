"""Production Autopilot for calibrated, hardware-aware anomaly deployment."""

from __future__ import annotations

import argparse
from html import escape
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
    """Build the ``anomavision autopilot`` argument parser."""
    parser = argparse.ArgumentParser(
        description="Select, calibrate, profile, and package a production anomaly model.",
        add_help=add_help,
    )
    parser.add_argument("--config", type=str, required=True, help="Base AnomaVision config file.")
    parser.add_argument("--dataset_path", type=str, default=None, help="MVTec-style dataset root.")
    parser.add_argument("--class_name", type=str, default=None, help="Dataset class to evaluate.")
    parser.add_argument("--padim_model", type=str, default=None, help="PaDiM model artifact (.pt/.pth/.onnx).")
    parser.add_argument("--patchcore_model", type=str, default=None, help="PatchCore model artifact (.pt/.pth/.onnx).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing_batches", type=int, default=20)
    parser.add_argument("--target_latency_ms", type=float, default=None)
    parser.add_argument("--validation_split", type=float, default=1.0, help="Fraction of the complete labeled test split used for calibration; 1.0 uses every sample.")
    parser.add_argument("--output_dir", type=str, default="./production_package")
    parser.add_argument("--copy_config", action="store_true", default=True)
    return parser


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _profile_model(model_path: str, dataloader: DataLoader, device: str, warmup: int, timing_batches: int) -> Dict[str, Any]:
    wrapper = ModelWrapper(model_path, device)
    iterator = iter(dataloader)
    try:
        first = next(iterator)
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
    count = 0
    for item in dataloader:
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
        count += 1
    wrapper.close()
    scores_np = np.asarray(all_scores, dtype=np.float32)
    labels_np = np.asarray(all_labels, dtype=np.int64)
    maps_np = np.asarray(all_maps, dtype=np.float32) if all_maps else np.empty((0, 0, 0), dtype=np.float32)
    threshold, threshold_f1 = find_optimal_threshold(labels_np, scores_np) if len(np.unique(labels_np)) > 1 else (float(np.median(scores_np)), 0.0)
    image_metrics = compute_metrics(labels_np, scores_np, thresh=threshold)
    image_auroc = image_metrics.get("auc_score") if len(np.unique(labels_np)) > 1 else None
    pixel_auroc = None
    masks_np = np.asarray(all_masks, dtype=np.float32)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    localization = {"available": bool(len(maps_np) == len(labels_np) and maps_np.ndim == 3), "non_empty_fraction": 0.0, "mean_mask_area_fraction": 0.0}
    if localization["available"] and masks_np.shape == maps_np.shape and np.unique(masks_np).size > 1:
        try:
            pixel_auroc = float(roc_auc_score(masks_np.reshape(-1) > 0.5, maps_np.reshape(-1)))
        except ValueError:
            pixel_auroc = None
    image_metrics["image_auroc"] = image_auroc
    image_metrics["pixel_auroc"] = pixel_auroc
    anomaly_labels = (scores_np >= threshold).astype(np.uint8)
    if localization["available"]:
        loc_masks = make_localization_mask(maps_np, anomaly_labels)
        localization["non_empty_fraction"] = float(np.mean(loc_masks.reshape(len(loc_masks), -1).sum(axis=1) > 0))
        localization["mean_mask_area_fraction"] = float(loc_masks.mean())
    median_ms = float(np.median(timings) * 1000 / max(1, dataloader.batch_size)) if timings else 0.0
    p95_ms = float(np.percentile(timings, 95) * 1000 / max(1, dataloader.batch_size)) if timings else 0.0
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_format": Path(model_path).suffix.lower(),
        "threshold": float(threshold),
        "threshold_f1": float(threshold_f1),
        "metrics": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in image_metrics.items()},
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
    """Write a self-contained HTML dashboard and a Markdown fallback report."""
    selected = manifest["selected_model"]
    selected_result = manifest["candidates"][selected]
    target = manifest.get("target_latency_ms")
    cards = []
    rows = []
    for name, result in manifest["candidates"].items():
        metrics = result["metrics"]
        loc = result["localization"]
        is_selected = name == selected
        status = "Selected" if is_selected else "Candidate"
        status_class = "selected" if is_selected else "candidate"
        cards.append(
            f'<article class="model-card {status_class}"><div class="card-top"><span class="model-name">{escape(name.upper())}</span><span class="badge">{status}</span></div>'
            f'<div class="score">{_format_metric(metrics.get("image_auroc"))}<small> image AUROC</small></div>'
            f'<div class="mini-grid"><div><b>{_format_metric(metrics.get("pixel_auroc"))}</b><span>pixel AUROC</span></div><div><b>{result["latency_ms"]["p95"]:.1f} ms</b><span>p95 latency</span></div><div><b>{loc["non_empty_fraction"]:.1%}</b><span>non-empty maps</span></div></div></article>'
        )
        rows.append(
            f'<tr class="{"active" if is_selected else ""}"><td><strong>{escape(name)}</strong></td><td>{_format_metric(metrics.get("image_auroc"))}</td><td>{_format_metric(metrics.get("pixel_auroc"))}</td><td>{result["latency_ms"]["median"]:.2f}</td><td>{result["latency_ms"]["p95"]:.2f}</td><td>{loc["non_empty_fraction"]:.1%}</td><td><code>{result["threshold"]:.6f}</code></td></tr>'
        )
    target_text = f"under {target:.1f} ms p95" if target is not None else "with the strongest measured accuracy/latency balance"
    environment_json = escape(json.dumps(manifest["environment"], indent=2))
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AnomaVision Production Autopilot</title>
<style>
:root{{--ink:#14213d;--muted:#667085;--line:#e6eaf0;--blue:#315efb;--teal:#0f9d8a;--bg:#f5f7fb;--card:#fff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}} .wrap{{max-width:1180px;margin:auto;padding:34px 24px 60px}}
.hero{{background:linear-gradient(135deg,#172554,#315efb 58%,#5b8cff);color:white;border-radius:24px;padding:34px 38px;box-shadow:0 18px 42px #315efb2b}} .eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.18em;opacity:.75;font-weight:700}} h1{{font-size:clamp(30px,5vw,52px);line-height:1.04;margin:12px 0}} .hero p{{max-width:690px;margin:0;color:#e6ecff;font-size:17px}} .hero-meta{{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px}} .pill{{background:#ffffff22;border:1px solid #ffffff3b;border-radius:999px;padding:7px 12px;font-size:13px}}
.section{{margin-top:28px}} .section-title{{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:12px}} h2{{font-size:21px;margin:0}} .muted{{color:var(--muted)}} .recommend{{background:#fff;border:1px solid #dbe3ff;border-left:5px solid var(--blue);border-radius:16px;padding:20px 22px;box-shadow:0 8px 25px #14213d0b}} .recommend strong{{color:var(--blue)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}} .model-card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 8px 22px #14213d08}} .model-card.selected{{border-color:#9eb0ff;box-shadow:0 10px 28px #315efb20}} .card-top{{display:flex;justify-content:space-between;align-items:center}} .model-name{{font-weight:800;letter-spacing:.08em}} .badge{{font-size:12px;border-radius:999px;padding:4px 9px;background:#eef1f6;color:var(--muted)}} .selected .badge{{background:#e5eaff;color:var(--blue)}} .score{{font-size:42px;font-weight:800;margin:18px 0 12px;letter-spacing:-.04em}} .score small{{font-size:12px;letter-spacing:0;color:var(--muted);font-weight:600}} .mini-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} .mini-grid div{{border-top:1px solid var(--line);padding-top:8px}} .mini-grid b,.mini-grid span{{display:block}} .mini-grid b{{font-size:16px}} .mini-grid span{{font-size:11px;color:var(--muted)}}
.table-wrap{{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:16px}} table{{border-collapse:collapse;width:100%;min-width:720px}} th,td{{padding:14px 16px;text-align:left;border-bottom:1px solid var(--line)}} th{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:#fafbfc}} tr:last-child td{{border-bottom:0}} tr.active td{{background:#f3f6ff}} code{{background:#eef1f6;border-radius:6px;padding:3px 6px;font-size:12px}} .two-col{{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}} .panel{{background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px}} .check{{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--line)}} .check:last-child{{border-bottom:0}} .ok{{color:var(--teal);font-weight:700}} pre{{overflow:auto;background:#111827;color:#dbeafe;padding:16px;border-radius:12px;font-size:12px}} footer{{margin-top:28px;color:var(--muted);font-size:12px;text-align:center}} @media(max-width:760px){{.wrap{{padding:20px 14px 40px}}.hero{{padding:26px 22px;border-radius:18px}}.two-col{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap">
<section class="hero"><div class="eyebrow">AnomaVision / Production Autopilot</div><h1>Deployment confidence, before production.</h1><p>Calibrated thresholds, hardware-aware profiling, and localization health checks in one reproducible report.</p><div class="hero-meta"><span class="pill">Selected: <b>{escape(selected)}</b></span><span class="pill">Class: <b>{escape(str(manifest["dataset"]["class_name"]))}</b></span><span class="pill">Samples: <b>{manifest["dataset"]["samples"]}</b></span><span class="pill">Device: <b>{escape(str(manifest["environment"].get("device", "unknown")))}</b></span></div></section>
<section class="section"><div class="recommend"><strong>Recommendation</strong><br>Deploy <b>{escape(selected)}</b> with threshold <code>{selected_result["threshold"]:.6f}</code>. It was selected {escape(target_text)}. Recheck this threshold on a production validation set before release.</div></section>
<section class="section"><div class="section-title"><h2>Candidate overview</h2><span class="muted">Measured on the same validation data</span></div><div class="cards">{"".join(cards)}</div></section>
<section class="section"><div class="section-title"><h2>Detailed comparison</h2><span class="muted">Higher AUROC and lower latency are better</span></div><div class="table-wrap"><table><thead><tr><th>Model</th><th>Image AUROC</th><th>Pixel AUROC</th><th>Median ms</th><th>P95 ms</th><th>Maps non-empty</th><th>Threshold</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>
<section class="section two-col"><div class="panel"><h2>Localization health</h2><div class="check"><span>Selected model maps available</span><span class="ok">{"PASS" if selected_result["localization"]["available"] else "CHECK"}</span></div><div class="check"><span>Selected model non-empty maps</span><span class="ok">{selected_result["localization"]["non_empty_fraction"]:.1%}</span></div><div class="check"><span>Selected model mean mask area</span><span>{selected_result["localization"]["mean_mask_area_fraction"]:.1%}</span></div><p class="muted">A low non-empty rate can indicate a threshold, score-scale, export, or model-sensitivity problem.</p></div><div class="panel"><h2>Deployment artifact</h2><div class="check"><span>Artifact</span><code>{escape(str(manifest["selected_artifact"]))}</code></div><div class="check"><span>Format</span><code>{escape(str(selected_result["model_format"]))}</code></div><div class="check"><span>Preprocessing</span><span>{escape(str(manifest["preprocessing"].get("resize")))} px</span></div><div class="check"><span>Target latency</span><span>{escape(str(target)) if target is not None else "not set"}</span></div></div></section>
<section class="section"><div class="panel"><h2>Reproducibility environment</h2><pre>{environment_json}</pre></div></section><footer>Generated by AnomaVision Production Autopilot · manifest schema {manifest["schema_version"]}</footer>
</main></body></html>'''
    (output_dir / "production_autopilot_report.html").write_text(html, encoding="utf-8")

    markdown = ["# AnomaVision Production Autopilot Report", "", f"**Selected model:** `{selected}`", "", "See `production_autopilot_report.html` for the full dashboard.", ""]
    (output_dir / "localization_report.md").write_text("\\n".join(markdown), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run Autopilot and create a deployment package."""
    cfg = load_config(args.config)
    dataset_path = args.dataset_path or cfg.get("dataset_path") or cfg.get("img_path")
    class_name = args.class_name or cfg.get("class_name")
    if not dataset_path or not class_name:
        raise ValueError("dataset_path and class_name are required in the CLI or config.")
    device = determine_device(args.device)
    dataset = anomavision.MVTecDataset(dataset_path, class_name, is_train=False, resize=cfg.get("resize", 224), crop_size=cfg.get("crop_size", 224), normalize=cfg.get("normalize", True), mean=cfg.get("norm_mean"), std=cfg.get("norm_std"))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    candidates = {}
    for name, model_path in (("padim", args.padim_model), ("patchcore", args.patchcore_model)):
        if model_path:
            candidates[name] = _profile_model(model_path, dataloader, device, args.warmup, args.timing_batches)
    if not candidates:
        raise ValueError("Provide at least one of --padim_model or --patchcore_model.")
    selected = _select(candidates, args.target_latency_ms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_source = Path(candidates[selected]["model_path"])
    packaged_model = output_dir / f"model{selected_source.suffix}"
    shutil.copy2(selected_source, packaged_model)
    manifest = {
        "schema_version": 1,
        "selected_model": selected,
        "selected_artifact": str(packaged_model.name),
        "dataset": {"path": str(Path(dataset_path).resolve()), "class_name": class_name, "samples": len(dataset)},
        "preprocessing": {"resize": cfg.get("resize", 224), "crop_size": cfg.get("crop_size", 224), "normalize": cfg.get("normalize", True), "mean": cfg.get("norm_mean"), "std": cfg.get("norm_std")},
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
