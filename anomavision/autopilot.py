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
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--class_name", default=None)
    parser.add_argument("--padim_model", default=None)
    parser.add_argument("--patchcore_model", default=None)
    parser.add_argument("--efficientad_model", default=None, help="EfficientAD model artifact (.pt/.pth/.onnx).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing_batches", type=int, default=20)
    parser.add_argument("--target_latency_ms", type=float, default=None)
    parser.add_argument("--validation_split", type=float, default=1.0)
    parser.add_argument("--output_dir", default="./production_package")
    parser.add_argument("--copy_config", action="store_true", default=True)
    return parser


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.1%}"


def _json_default(value: Any) -> Any:
    """Convert NumPy/PyTorch scalar values to JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _profile_model(model_path: str, dataloader: DataLoader, device: str, warmup: int, timing_batches: int) -> Dict[str, Any]:
    wrapper = ModelWrapper(model_path, device)
    timings = []
    scores_all, maps_all, labels_all, masks_all = [], [], [], []
    try:
        first = next(iter(dataloader))
        first_batch = first[0].to(device)
        for _ in range(max(0, warmup)):
            wrapper.predict(first_batch)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        for index, (batch, _, labels, masks) in enumerate(dataloader):
            batch = batch.to(device)
            measure = index < max(1, timing_batches)
            start = time.perf_counter() if measure else 0.0
            scores, maps = wrapper.predict(batch)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            if measure:
                timings.append(time.perf_counter() - start)
            scores_all.extend(_to_numpy(scores).reshape(-1).tolist())
            if maps is not None:
                maps_all.extend(_to_numpy(maps))
            labels_all.extend(_to_numpy(labels).reshape(-1).astype(int).tolist())
            masks_all.extend(_to_numpy(masks))
    finally:
        wrapper.close()

    scores = np.asarray(scores_all, dtype=np.float32)
    labels = np.asarray(labels_all, dtype=np.int64)
    maps = np.asarray(maps_all, dtype=np.float32) if maps_all else np.empty((0, 0, 0), dtype=np.float32)
    masks = np.asarray(masks_all, dtype=np.float32)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if len(np.unique(labels)) > 1:
        threshold, threshold_f1 = find_optimal_threshold(labels, scores)
    else:
        threshold, threshold_f1 = (float(np.median(scores)) if len(scores) else 0.0), 0.0
    metrics = compute_metrics(labels, scores, thresh=threshold) if len(labels) else {}
    pixel_auroc = None
    localization = {"available": bool(len(maps) == len(labels) and maps.ndim == 3), "non_empty_fraction": None,
                    "mean_mask_area_fraction": None, "anomaly_non_empty_fraction": None,
                    "normal_false_positive_fraction": None, "anomaly_mean_mask_area_fraction": None,
                    "normal_mean_mask_area_fraction": None, "verdict": "unavailable"}
    if localization["available"] and masks.shape == maps.shape and np.unique(masks).size > 1:
        try:
            pixel_auroc = float(roc_auc_score(masks.reshape(-1) > 0.5, maps.reshape(-1)))
        except ValueError:
            pass
        loc_masks = make_localization_mask(maps, (scores >= threshold).astype(np.uint8)).astype(bool)
        flat = loc_masks.reshape(len(loc_masks), -1)
        non_empty, area = flat.any(axis=1), flat.mean(axis=1)
        anomaly, normal = labels == 1, labels == 0
        localization["non_empty_fraction"] = float(non_empty.mean())
        localization["mean_mask_area_fraction"] = float(area.mean())
        if anomaly.any():
            localization["anomaly_non_empty_fraction"] = float(non_empty[anomaly].mean())
            localization["anomaly_mean_mask_area_fraction"] = float(area[anomaly].mean())
        if normal.any():
            localization["normal_false_positive_fraction"] = float(non_empty[normal].mean())
            localization["normal_mean_mask_area_fraction"] = float(area[normal].mean())
        localization["verdict"] = "healthy" if (localization["normal_false_positive_fraction"] or 0.0) <= 0.10 else "review false positives"
    metrics["image_auroc"] = float(metrics["auc_score"]) if metrics.get("auc_score") is not None else None
    metrics["pixel_auroc"] = pixel_auroc
    bs = max(1, int(dataloader.batch_size or 1))
    median_ms = float(np.median(timings) * 1000 / bs) if timings else 0.0
    p95_ms = float(np.percentile(timings, 95) * 1000 / bs) if timings else 0.0
    return {"model_path": str(Path(model_path).resolve()), "model_format": Path(model_path).suffix.lower(),
            "threshold": float(threshold), "threshold_f1": float(threshold_f1), "metrics": metrics,
            "latency_ms": {"median": median_ms, "p95": p95_ms},
            "throughput_images_per_second": float(1000 / median_ms) if median_ms else 0.0,
            "localization": localization, "samples": int(len(labels))}


def _select(results: Dict[str, Dict[str, Any]], target_latency_ms: Optional[float]) -> str:
    eligible = results
    if target_latency_ms is not None:
        eligible = {n: r for n, r in results.items() if r["latency_ms"]["p95"] <= target_latency_ms}
    if not eligible:
        eligible = results
    return max(eligible, key=lambda n: (eligible[n]["metrics"].get("image_auroc") or 0.0, -eligible[n]["latency_ms"]["p95"]))


def _write_report(manifest: Dict[str, Any], output_dir: Path) -> None:
    selected = manifest["selected_model"]
    selected_result = manifest["candidates"][selected]
    target = manifest.get("target_latency_ms")
    cards, rows = [], []
    for name, result in manifest["candidates"].items():
        metrics, loc = result["metrics"], result["localization"]
        active = name == selected
        cards.append(f'''<article class="card {"selected" if active else ""}"><div class="top"><b>{html.escape(name.upper())}</b><span class="badge">{"SELECTED" if active else "CANDIDATE"}</span></div><div class="big">{_fmt(metrics.get("image_auroc"))}<small>Image AUROC</small></div><div class="stats"><div><b>{_fmt(metrics.get("pixel_auroc"))}</b><small>Pixel AUROC</small></div><div><b>{result["latency_ms"]["p95"]:.1f} ms</b><small>P95 latency</small></div><div><b>{_pct(loc.get("anomaly_non_empty_fraction"))}</b><small>Anomaly coverage</small></div></div></article>''')
        rows.append(f'''<tr class="{"active" if active else ""}"><td><strong>{html.escape(name)}</strong></td><td>{_fmt(metrics.get("image_auroc"))}</td><td>{_fmt(metrics.get("pixel_auroc"))}</td><td>{result["latency_ms"]["median"]:.2f}</td><td>{result["latency_ms"]["p95"]:.2f}</td><td>{_pct(loc.get("anomaly_non_empty_fraction"))}</td><td>{_pct(loc.get("normal_false_positive_fraction"))}</td><td><code>{result["threshold"]:.6f}</code></td></tr>''')
    target_text = f"under {target:.1f} ms p95" if target is not None else "with the strongest measured accuracy/latency balance"
    env = html.escape(json.dumps(manifest["environment"], indent=2))
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AnomaVision Production Autopilot</title><style>
:root{{--ink:#14213d;--muted:#667085;--line:#e6eaf0;--blue:#315efb;--teal:#0f9d8a;--bg:#f5f7fb;--card:#fff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,system-ui,-apple-system,Segoe UI,sans-serif}}.wrap{{max-width:1180px;margin:auto;padding:34px 24px 60px}}.hero{{background:linear-gradient(135deg,#172554,#315efb 58%,#5b8cff);color:white;border-radius:24px;padding:36px 40px;box-shadow:0 18px 42px #315efb2b}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.18em;opacity:.75;font-weight:700}}h1{{font-size:clamp(32px,5vw,52px);line-height:1.05;margin:12px 0}}.hero p{{max-width:720px;color:#e6ecff;font-size:17px}}.pills{{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px}}.pill{{background:#ffffff22;border:1px solid #ffffff3b;border-radius:999px;padding:7px 12px}}section{{margin-top:28px}}h2{{font-size:21px;margin:0 0 12px}}.recommend,.panel{{background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px 22px;box-shadow:0 8px 25px #14213d0b}}.recommend{{border-left:5px solid var(--blue)}}.recommend strong{{color:var(--blue)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 8px 22px #14213d08}}.card.selected{{border-color:#9eb0ff;box-shadow:0 10px 28px #315efb20}}.top{{display:flex;justify-content:space-between;align-items:center}}.badge{{font-size:11px;border-radius:999px;padding:4px 9px;background:#eef1f6;color:var(--muted)}}.selected .badge{{background:#e5eaff;color:var(--blue)}}.big{{font-size:42px;font-weight:800;margin:18px 0 12px}}small{{display:block;color:var(--muted);font-size:11px;font-weight:600}}.big small{{display:inline;font-size:12px;margin-left:6px}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.stats>div{{border-top:1px solid var(--line);padding-top:9px}}.stats b{{font-size:16px}}.table{{overflow:auto;border:1px solid var(--line);border-radius:16px;background:#fff}}table{{border-collapse:collapse;width:100%;min-width:800px}}th,td{{padding:14px 16px;text-align:left;border-bottom:1px solid var(--line)}}th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:#fafbfc}}tr.active td{{background:#f3f6ff}}tr:last-child td{{border:0}}code{{background:#eef1f6;border-radius:6px;padding:3px 6px;font-size:12px}}.grid2{{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}}.check{{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--line)}}.check:last-child{{border:0}}.ok{{color:var(--teal);font-weight:700}}pre{{overflow:auto;background:#111827;color:#dbeafe;padding:16px;border-radius:12px;font-size:12px}}footer{{margin-top:30px;text-align:center;color:var(--muted);font-size:12px}}@media(max-width:760px){{.wrap{{padding:20px 14px 40px}}.hero{{padding:28px 22px;border-radius:18px}}.grid2{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap"><section class="hero"><div class="eyebrow">AnomaVision / Production Autopilot</div><h1>Deployment confidence, before production.</h1><p>Calibrated thresholds, hardware-aware profiling, and localization health checks in one reproducible report.</p><div class="pills"><span class="pill">Selected: <b>{html.escape(selected)}</b></span><span class="pill">Class: <b>{html.escape(str(manifest["dataset"]["class_name"]))}</b></span><span class="pill">Samples: <b>{manifest["dataset"]["samples"]}</b></span><span class="pill">Device: <b>{html.escape(str(manifest["environment"].get("device","unknown")))}</b></span></div></section><section><div class="recommend"><strong>Recommendation</strong><br>Deploy <b>{html.escape(selected)}</b> with threshold <code>{selected_result["threshold"]:.6f}</code>. It was selected {html.escape(target_text)}.</div></section><section><h2>Candidate overview</h2><div class="cards">{"".join(cards)}</div></section><section><h2>Detailed comparison</h2><div class="table"><table><thead><tr><th>Model</th><th>Image AUROC</th><th>Pixel AUROC</th><th>Median ms</th><th>P95 ms</th><th>Anomaly coverage</th><th>Normal false positives</th><th>Threshold</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section><section class="grid2"><div class="panel"><h2>Localization health</h2><div class="check"><span>Maps available</span><span class="ok">{"PASS" if selected_result["localization"]["available"] else "CHECK"}</span></div><div class="check"><span>Anomaly localization</span><span>{_pct(selected_result["localization"].get("anomaly_non_empty_fraction"))}</span></div><div class="check"><span>Normal false-positive maps</span><span>{_pct(selected_result["localization"].get("normal_false_positive_fraction"))}</span></div><div class="check"><span>Mean anomaly mask area</span><span>{_pct(selected_result["localization"].get("anomaly_mean_mask_area_fraction"))}</span></div><div class="check"><span>Verdict</span><span class="ok">{html.escape(str(selected_result["localization"].get("verdict","N/A")))}</span></div></div><div class="panel"><h2>Deployment artifact</h2><div class="check"><span>Artifact</span><code>{html.escape(str(manifest["selected_artifact"]))}</code></div><div class="check"><span>Format</span><code>{html.escape(str(selected_result["model_format"]))}</code></div><div class="check"><span>Preprocessing</span><span>{manifest["preprocessing"].get("resize",224)} px</span></div><div class="check"><span>Target latency</span><span>{target if target is not None else "not set"}</span></div></div></section><section><div class="panel"><h2>Reproducibility environment</h2><pre>{env}</pre></div></section><footer>Generated by AnomaVision Production Autopilot · manifest schema {manifest["schema_version"]}</footer></main></body></html>'''
    (output_dir / "production_autopilot_report.html").write_text(document, encoding="utf-8")
    (output_dir / "localization_report.md").write_text(f"# AnomaVision Production Autopilot Report\n\n**Selected model:** `{selected}`\n\nSee `production_autopilot_report.html` for the full dashboard.\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    dataset_path = args.dataset_path or cfg.get("dataset_path") or cfg.get("img_path")
    class_name = args.class_name or cfg.get("class_name")
    if not dataset_path or not class_name:
        raise ValueError("dataset_path and class_name are required in the CLI or config.")
    device = determine_device(args.device)
    dataset = anomavision.MVTecDataset(dataset_path, class_name, is_train=False, resize=cfg.get("resize",224), crop_size=cfg.get("crop_size",224), normalize=cfg.get("normalize",True), mean=cfg.get("norm_mean"), std=cfg.get("norm_std"))
    dataloader = DataLoader(dataset, batch_size=max(1,args.batch_size), shuffle=False, num_workers=max(0,args.num_workers), pin_memory=device.startswith("cuda"))
    candidates = {}
    for name, model_path in (("padim",args.padim_model),("patchcore",args.patchcore_model),("efficientad",args.efficientad_model)):
        if model_path:
            candidates[name] = _profile_model(model_path,dataloader,device,args.warmup,args.timing_batches)
    if not candidates:
        raise ValueError("Provide at least one model: --padim_model, --patchcore_model, or --efficientad_model.")
    selected = _select(candidates,args.target_latency_ms)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    source = Path(candidates[selected]["model_path"]); packaged_model = output_dir / f"model{source.suffix}"; shutil.copy2(source,packaged_model)
    sidecar = None
    if selected == "efficientad":
        for candidate in (source.with_suffix(".pth"),source.parent/"model.pth"):
            if candidate.exists():
                sidecar = output_dir/candidate.name; shutil.copy2(candidate,sidecar); break
    if args.copy_config:
        shutil.copy2(args.config,output_dir/Path(args.config).name)
    manifest = {"schema_version":5,"selected_model":selected,"selected_artifact":packaged_model.name,"calibration_artifact":sidecar.name if sidecar else None,"dataset":{"path":str(Path(dataset_path).resolve()),"class_name":class_name,"samples":len(dataset)},"preprocessing":{"resize":cfg.get("resize",224),"crop_size":cfg.get("crop_size",224),"normalize":cfg.get("normalize",True),"mean":cfg.get("norm_mean"),"std":cfg.get("norm_std")},"candidates":candidates,"target_latency_ms":args.target_latency_ms,"environment":{"python":sys.version.split()[0],"platform":platform.platform(),"torch":torch.__version__,"device":device}}
    (output_dir/"deployment_manifest.json").write_text(json.dumps(manifest,indent=2,default=_json_default),encoding="utf-8")
    _write_report(manifest,output_dir)
    return manifest


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or create_parser().parse_args()
    manifest = run(args)
    print(json.dumps({"selected_model":manifest["selected_model"],"output_dir":str(Path(args.output_dir).resolve()),"report":str(Path(args.output_dir).resolve()/"production_autopilot_report.html")},indent=2))


if __name__ == "__main__":
    main()
