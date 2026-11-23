"""Exploratory frozen PatchCore vs CAD-Mem benchmark (not a publication protocol).

Uses a held-out portion of test/good for adaptation and never evaluates on
those images. Calibration may overlap PatchCore training; see README.
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


def evaluate(base, cadmem, loader, device):
    records = []
    with torch.no_grad():
        for images, _, labels, _ in loader:
            images = images.to(device)
            frozen, _ = base.predict(images)
            adapted, _ = cadmem.predict(images)
            for label, a, b in zip(labels.tolist(), frozen.cpu().tolist(), adapted.cpu().tolist()):
                records.append((int(label), float(a), float(b)))
    return records


def metrics(records, column, threshold):
    labels = np.asarray([r[0] for r in records], dtype=int)
    scores = np.asarray([r[column] for r in records], dtype=float)
    predicted = scores >= threshold
    positives, negatives = labels == 1, labels == 0
    result = {
        "n": len(labels),
        "auroc": float(roc_auc_score(labels, scores)) if positives.any() and negatives.any() else None,
        "ap": float(average_precision_score(labels, scores)) if positives.any() else None,
        "recall": float(predicted[positives].mean()) if positives.any() else None,
        "fpr": float(predicted[negatives].mean()) if negatives.any() else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--class-name", default="bottle")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="results/cadmem")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--adapt-fraction", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.adapt_fraction < 1:
        parser.error("--adapt-fraction must be in (0, 1)")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # This is a trusted local checkpoint. Never unpickle untrusted models.
    base = torch.load(args.model_path, map_location=device, weights_only=False)
    base.to_device(device)
    base.eval()
    if not base.is_fitted:
        raise RuntimeError("Loaded PatchCore has no fitted memory bank")

    train = MVTecDataset(args.dataset_path, args.class_name, is_train=True,
                         resize=(224, 224), crop_size=None, normalize=True)
    test = MVTecDataset(args.dataset_path, args.class_name, is_train=False,
                        resize=(224, 224), crop_size=None, normalize=True)
    if not len(train):
        raise RuntimeError("Empty calibration dataset")
    # Normal-only calibration. Existing checkpoint may have been trained on these.
    generator = torch.Generator().manual_seed(args.seed)
    train_idx = torch.randperm(len(train), generator=generator).tolist()
    cal_idx = train_idx[:max(1, len(train_idx) // 5)]
    calibration = DataLoader(Subset(train, cal_idx), batch_size=args.batch_size)
    cadmem = CADMemPatchCore(base, CADMemConfig())
    cadmem.calibrate(calibration)

    good_idx = [i for i, y in enumerate(test.y) if int(y) == 0]
    if len(good_idx) < 2:
        raise RuntimeError("Need at least two test/good images for disjoint adaptation/evaluation")
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(good_idx).tolist()
    count = min(len(shuffled) - 1, max(1, int(len(shuffled) * args.adapt_fraction)))
    adapt_idx = shuffled[:count]
    eval_idx = sorted(set(range(len(test))) - set(adapt_idx))
    if not any(int(test.y[i]) == 1 for i in eval_idx):
        raise RuntimeError("Evaluation set has no anomalous images")
    adaptation = DataLoader(Subset(test, adapt_idx), batch_size=args.batch_size, shuffle=False)
    evaluation = DataLoader(Subset(test, eval_idx), batch_size=args.batch_size, shuffle=False)

    before = evaluate(base, cadmem, evaluation, device)
    # Separate test/good adaptation subset; its labels are used only to construct
    # this oracle-clean exploratory stream, NOT for CAD-Mem admission.
    admitted = 0
    for images, _, _, _ in adaptation:
        admitted += cadmem.update(images.to(device))
    after = evaluate(base, cadmem, evaluation, device)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_index", "label", "patchcore", "cadmem_before", "cadmem_after"])
        for idx, a, b in zip(eval_idx, before, after):
            writer.writerow([idx, a[0], a[1], a[2], b[2]])

    report = {
        "protocol": "exploratory_oracle_clean_adaptation_NOT_publication_ready",
        "class_name": args.class_name, "seed": args.seed,
        "calibration_images": len(cal_idx), "adaptation_images": len(adapt_idx),
        "evaluation_images": len(eval_idx), "admitted_patches": admitted,
        "adaptive_bank_size": len(cadmem.memory.adaptive),
        "frozen_patchcore": metrics(after, 1, args.threshold),
        "cadmem_before": metrics(before, 2, args.threshold),
        "cadmem_after": metrics(after, 2, args.threshold),
        "threshold": args.threshold,
        "caveats": [
            "Calibration may overlap the original PatchCore training set.",
            "Adaptation images were selected using test labels; this is an oracle-clean exploratory scenario.",
            "Adaptation and evaluation images are disjoint, but share the original test split.",
            "No drift, contamination, independent train split, or multi-seed study is implemented.",
            "CPU NumPy CAD-Mem scoring may differ slightly from torch PatchCore scoring."
        ]
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {output / 'predictions.csv'} and {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
