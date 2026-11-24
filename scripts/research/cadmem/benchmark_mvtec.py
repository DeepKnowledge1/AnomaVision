"""Matched MVTec AD pilot: identical features/prototypes, point memory vs CADMem.

Run:
 python scripts/research/cadmem/benchmark_mvtec.py \
   --dataset-root /data/mvtec --category bottle --device cuda \
   --output results/cadmem_bottle.json

This is an exploratory, single-category benchmark. It is NOT evidence of
scientific novelty or robustness under real acquisition changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from anomavision.algorithm.cadmem import BoundedNuisanceMemory
from anomavision.algorithm.cadmem.ablations import (
    fit_local_normal_pca,
    without_budgets,
)
from anomavision.algorithm.patchcore.patchcore import PatchCore
from anomavision.datasets.dataset import AnodetDataset, collate_anodet_batch
from anomavision.datasets.mvtec_dataset import MVTecDataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def brightness(images, shift):
    """Brightness shift on RGB values before re-applying ImageNet normalization."""
    mean = images.new_tensor(MEAN)[None, :, None, None]
    std = images.new_tensor(STD)[None, :, None, None]
    rgb = (images * std + mean + shift).clamp(0, 1)
    return (rgb - mean) / std


@torch.no_grad()
def extract(model, images):
    z, _, _ = model._extract(images)
    return z.reshape(-1, z.shape[-1]).cpu()


@torch.no_grad()
def score_patches(model, bank, images, kind):
    patches, width, height = model._extract(images)
    patches = patches.reshape(-1, patches.shape[-1])
    if isinstance(bank, BoundedNuisanceMemory):
        scores = bank.score(patches)
    else:
        # Point-memory score with the exact SAME stored prototypes.
        scores = torch.cat(
            [
                torch.cdist(part, bank.to(part)).min(dim=1).values
                for part in patches.split(256)
            ]
        )
    return scores.reshape(images.shape[0], width, height)


def metrics(labels, scores, masks, maps, threshold):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores)
    mask = np.concatenate(masks)
    pixels = np.concatenate(maps)
    normal, defective = labels == 0, labels == 1
    return {
        "image_auroc": (
            float(roc_auc_score(labels, scores))
            if len(np.unique(labels)) == 2
            else None
        ),
        "image_ap": (
            float(average_precision_score(labels, scores)) if defective.any() else None
        ),
        "pixel_auroc": (
            float(roc_auc_score(mask, pixels)) if len(np.unique(mask)) == 2 else None
        ),
        "pixel_ap": (
            float(average_precision_score(mask, pixels)) if mask.sum() else None
        ),
        "fpr_frozen": (
            float((scores[normal] > threshold).mean())
            if normal.any() and threshold is not None
            else None
        ),
        "defect_recall_frozen": (
            float((scores[defective] > threshold).mean()) if defective.any() else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prototypes", type=int, default=256)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--counterfactuals", type=int, default=4)
    parser.add_argument("--shift", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or not 0 < args.calibration_fraction < 1:
        parser.error("alpha and calibration-fraction must be in (0, 1)")
    if min(args.prototypes, args.counterfactuals, args.batch_size) < 1:
        parser.error("prototypes, counterfactuals and batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but is not available")

    rng = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.dataset_root)
    folder = root / args.category / "train" / "good"
    if not folder.is_dir():
        parser.error(f"Missing MVTec normal training images: {folder}")

    train = AnodetDataset(
        str(folder),
        resize=(args.size, args.size),
        crop_size=(args.size, args.size),
    )
    if len(train) < 5:
        parser.error("At least five independent normal images are needed")
    order = torch.randperm(len(train), generator=rng).tolist()
    n_cal = max(1, min(len(train) - 1, round(len(train) * args.calibration_fraction)))
    fit_ids, cal_ids = order[n_cal:], order[:n_cal]

    def loader(indices):
        return DataLoader(
            Subset(train, indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_anodet_batch,
        )

    test = MVTecDataset(
        str(root),
        args.category,
        is_train=False,
        resize=(args.size, args.size),
        crop_size=(args.size, args.size),
    )
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    extractor = PatchCore(
        backbone="resnet18",
        device=torch.device(args.device),
        layer_indices=[1, 2],
        patch_grid=14,
        max_memory_patches=args.prototypes,
        coreset_seed=args.seed,
    )
    extractor.eval()

    originals, paired = [], []
    for images, _, _, _ in loader(fit_ids):
        originals.append(extract(extractor, images))
        transformed = []
        for _ in range(args.counterfactuals):
            value = float((torch.rand((), generator=rng) * 2 - 1) * 0.25)
            transformed.append(extract(extractor, brightness(images, value)))
        paired.append(torch.stack(transformed))
    normal = torch.cat(originals)
    counterfactual = torch.cat(paired, dim=1)
    centers = extractor._select_coreset(normal, min(args.prototypes, len(normal))).to(
        args.device
    )
    cadmem = BoundedNuisanceMemory(
        rank=args.rank,
        max_prototypes=args.prototypes,
        neighbors=args.neighbors,
    ).to(args.device)
    cadmem.fit_features(
        normal.to(args.device),
        counterfactual.to(args.device),
        prototypes=centers,
    )
    # Equal prototype-count controls for ordinary augmentation and local PCA.
    augmented = torch.cat(
        [normal, counterfactual.reshape(-1, normal.shape[-1])], dim=0
    )
    augmented_centers = extractor._select_coreset(
        augmented, min(args.prototypes, len(augmented))
    ).to(args.device)
    local_pca = fit_local_normal_pca(
        normal.to(args.device),
        centers,
        rank=args.rank,
        neighbors=args.neighbors,
    )
    unbounded = without_budgets(cadmem)
    methods = {
        "point_memory": centers,
        "augmentation_only": augmented_centers,
        "local_normal_pca": local_pca,
        "unbounded_projection": unbounded,
        "cadmem": cadmem,
    }

    # Calibration uses held-out NORMAL training images; no test labels.
    thresholds = {}
    for name, bank in methods.items():
        values = []
        for images, _, _, _ in loader(cal_ids):
            score_map = score_patches(extractor, bank, images, name)
            values.extend(score_map.flatten(1).amax(1).tolist())
        n = len(values)
        order_stat = int(np.ceil((n + 1) * (1 - args.alpha)))
        # Insufficient calibration samples -> abstain rather than fabricate
        # a supported low-FPR operating threshold.
        thresholds[name] = (
            float(np.sort(values)[order_stat - 1]) if order_stat <= n else None
        )

    output = {
        "configuration": vars(args),
        "normal_fit_images": len(fit_ids),
        "normal_calibration_images": len(cal_ids),
        "test_images": len(test),
        "thresholds": thresholds,
        "results": {},
        "status": "exploratory; no external robustness or novelty claim",
    }
    for name, bank in methods.items():
        output["results"][name] = {}
        for condition, shift in (("clean", 0.0), ("synthetic_brightness", args.shift)):
            labels, image_scores, gt_masks, pred_maps = [], [], [], []
            for images, _, label, mask in test_loader:
                images = brightness(images, shift)
                patch_map = score_patches(extractor, bank, images, name)
                image_score = patch_map.flatten(1).amax(1)
                interpolated = F.interpolate(
                    patch_map[:, None],
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                labels.append(int(label.item()))
                image_scores.append(float(image_score.item()))
                gt_masks.append((mask > 0.5).numpy().reshape(-1))
                pred_maps.append(interpolated.cpu().numpy().reshape(-1))
            output["results"][name][condition] = metrics(
                labels,
                image_scores,
                gt_masks,
                pred_maps,
                thresholds[name],
            )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
