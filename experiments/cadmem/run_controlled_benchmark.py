"""Controlled CAD-Mem benchmark with clean/drift/contaminated adaptation.

EXPLORATORY: requires a trusted pre-trained PatchCore checkpoint. Calibration
uses train/good, which may overlap checkpoint training. Test images are split
into nonoverlapping adaptation and evaluation partitions. Test labels are used
to construct contamination ratios, never by the adaptation algorithms.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from anomavision.algorithm.cadmem import CADMemConfig, CADMemPatchCore
from anomavision.datasets.mvtec_dataset import MVTecDataset


def shifted(x, mode, strength):
    """Apply deterministic brightness/contrast changes in RGB before normalization."""
    if mode == "none":
        return x
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rgb = (x * std + mean).clamp(0, 1)
    if mode == "brightness":
        rgb = (rgb + strength).clamp(0, 1)
    elif mode == "contrast":
        rgb = ((rgb - 0.5) * (1 + strength) + 0.5).clamp(0, 1)
    else:
        raise ValueError(mode)
    return (rgb - mean) / std


def scores(model, loader, device, drift, strength):
    rows = []
    with torch.no_grad():
        for images, _, labels, _ in loader:
            images = shifted(images.to(device), drift, strength)
            result, _ = model.predict(images)
            rows.extend((int(y), float(s)) for y, s in zip(labels.tolist(), result.cpu().tolist()))
    return rows


def cutoff(normal_scores, target_fpr):
    """Conservative finite-sample quantile; may produce a threshold above max."""
    values = np.sort(np.asarray(normal_scores, dtype=float))
    k = int(np.ceil((len(values) + 1) * (1 - target_fpr))) - 1
    return float(values[k]) if k < len(values) else float(np.nextafter(values[-1], np.inf))


def measures(rows, threshold):
    labels = np.asarray([r[0] for r in rows], dtype=int)
    values = np.asarray([r[1] for r in rows], dtype=float)
    pos, neg = labels == 1, labels == 0
    return {
        "n": len(rows), "n_normal": int(neg.sum()), "n_anomalous": int(pos.sum()),
        "auroc": float(roc_auc_score(labels, values)) if pos.any() and neg.any() else None,
        "ap": float(average_precision_score(labels, values)) if pos.any() else None,
        "recall": float(np.mean(values[pos] > threshold)) if pos.any() else None,
        "fpr": float(np.mean(values[neg] > threshold)) if neg.any() else None,
        "threshold": threshold,
    }


def update_naive(model, images):
    """Unprotected FIFO-free append baseline; bounded to CAD-Mem's memory budget."""
    patches, _, _ = model.patchcore._extract(images)
    x = patches.reshape(-1, patches.shape[-1]).detach().cpu().numpy()
    memory = model.memory
    room = memory.config.max_adaptive - len(memory.adaptive)
    if room <= 0:
        return 0
    chosen = x[:room].copy()
    memory.adaptive = np.concatenate([memory.adaptive, chosen], axis=0)
    return len(chosen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--class-name", default="cable")
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", default="results/cadmem/controlled")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--drift", choices=["none", "brightness", "contrast"], default="brightness")
    p.add_argument("--strength", type=float, default=0.15)
    p.add_argument("--contamination", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.2])
    p.add_argument("--adapt-normal-fraction", type=float, default=0.4)
    args = p.parse_args()
    if not 0 < args.target_fpr < 1 or not 0 < args.adapt_normal_fraction < 1:
        p.error("target-fpr and adapt-normal-fraction must be between 0 and 1")
    if args.batch_size < 1 or not 0 <= args.strength <= 1:
        p.error("batch-size must be positive and strength must be between 0 and 1")
    if any(c < 0 or c >= 1 for c in args.contamination):
        p.error("contamination rates must be in [0,1)")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # Only load checkpoints from trusted sources.
    base = torch.load(args.model_path, map_location=device, weights_only=False)
    base.to_device(device)
    base.eval()
    if not base.is_fitted:
        raise RuntimeError("PatchCore checkpoint is not fitted")

    train = MVTecDataset(args.dataset_path, args.class_name, is_train=True,
                         resize=(224, 224), crop_size=None, normalize=True)
    test = MVTecDataset(args.dataset_path, args.class_name, is_train=False,
                        resize=(224, 224), crop_size=None, normalize=True)
    train_ids = rng.permutation(len(train)).tolist()
    if len(train_ids) < 4:
        raise RuntimeError("At least four train/good images are required")
    split = len(train_ids) // 2
    cal_ids, val_ids = train_ids[:split], train_ids[split:]
    calibration = DataLoader(Subset(train, cal_ids), batch_size=args.batch_size)
    validation = DataLoader(Subset(train, val_ids), batch_size=args.batch_size)

    normal = rng.permutation([i for i, y in enumerate(test.y) if int(y) == 0]).tolist()
    anomalies = rng.permutation([i for i, y in enumerate(test.y) if int(y) == 1]).tolist()
    if len(normal) < 3 or len(anomalies) < 3:
        raise RuntimeError("Need at least three normal and three anomalous test images")
    n_adapt = max(1, min(len(normal) - 2, int(len(normal) * args.adapt_normal_fraction)))
    adapt_normals = normal[:n_adapt]
    eval_normals = normal[n_adapt:]

    # Reserve anomalies for evaluation; a separate pool is used for contamination.
    max_rate = max(args.contamination)
    needed = int(np.ceil(n_adapt * max_rate / (1 - max_rate))) if max_rate else 0
    if needed >= len(anomalies):
        raise RuntimeError("Not enough anomalies to reserve both contamination and evaluation pools")
    contam_pool = anomalies[:needed]
    eval_anomalies = anomalies[needed:]
    eval_ids = sorted(eval_normals + eval_anomalies)
    eval_loader = DataLoader(Subset(test, eval_ids), batch_size=args.batch_size)
    normal_val_scores = [s for _, s in scores(base, validation, device, "none", 0)]
    threshold = cutoff(normal_val_scores, args.target_fpr)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    predictions = []
    for rate in args.contamination:
        n_bad = int(round(n_adapt * rate / (1 - rate))) if rate else 0
        n_bad = min(n_bad, len(contam_pool))
        stream = rng.permutation(adapt_normals + contam_pool[:n_bad]).tolist()
        loader = DataLoader(Subset(test, stream), batch_size=args.batch_size)
        cad = CADMemPatchCore(base, CADMemConfig())
        naive = CADMemPatchCore(base, CADMemConfig())
        cad.calibrate(calibration)
        naive.calibrate(calibration)
        # All three methods use the same frozen reference backbone and memory.
        # Their adaptation paths differ only in admission policy.
        cad_admitted = naive_admitted = 0
        for images, _, _, _ in loader:
            images = shifted(images.to(device), args.drift, args.strength)
            cad_admitted += cad.update(images)
            naive_admitted += update_naive(naive, images)
        for method, model in [("patchcore", base), ("naive", naive), ("cadmem", cad)]:
            rows = scores(model, eval_loader, device, args.drift, args.strength)
            item = {"method": method, "contamination_requested": rate,
                    "contamination_realized": n_bad / len(stream),
                    "drift": args.drift, "strength": args.strength,
                    "seed": args.seed, "adaptation_images": len(stream),
                    "admitted_patches": 0 if method == "patchcore" else
                        (naive_admitted if method == "naive" else cad_admitted),
                    "memory_size": 0 if method == "patchcore" else
                        len(model.memory.adaptive)}
            item.update(measures(rows, threshold))
            results.append(item)
            for idx, (label, score) in zip(eval_ids, rows):
                predictions.append({"method": method, "contamination": rate,
                                    "eval_index": idx, "label": label, "score": score})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    report = {
        "protocol": "controlled_exploratory_not_publication_ready",
        "class_name": args.class_name, "seed": args.seed,
        "normal_calibration_images": len(cal_ids),
        "normal_threshold_validation_images": len(val_ids),
        "normal_adaptation_pool": len(adapt_normals),
        "normal_evaluation_images": len(eval_normals),
        "anomalous_evaluation_images": len(eval_anomalies),
        "threshold": threshold, "target_validation_fpr": args.target_fpr,
        "results": results,
        "limitations": [
            "Existing PatchCore checkpoint may have trained on calibration and threshold-validation images.",
            "Test labels construct contamination and evaluation pools; neither adaptation algorithm sees labels.",
            "Drift is synthetic brightness/contrast, not measured industrial domain shift.",
            "Naive baseline appends the earliest stream patches until full; this is one specific naive policy.",
            "Threshold is calibrated on unshifted train/good images and held fixed across scenarios.",
            "One seed/category is not sufficient for journal claims.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} method/scenario rows to {output}")
    for row in results:
        print(f"{row['method']:9} contam={row['contamination_realized']:.3f} "
              f"AUROC={row['auroc']:.4f} FPR={row['fpr']:.3f} recall={row['recall']:.3f} "
              f"memory={row['memory_size']}")


if __name__ == "__main__":
    main()
