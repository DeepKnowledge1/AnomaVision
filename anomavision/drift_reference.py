"""Generate trusted reference embeddings for production drift monitoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import anomavision
from anomavision.config import _shape, load_config
from anomavision.general import determine_device
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.utils import merge_config


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reference embeddings for production drift monitoring.",
        add_help=add_help,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yml/.json")
    parser.add_argument("--img_path", type=str, default=None, help="Healthy/reference image directory")
    parser.add_argument("--model", type=str, default=None, help="Model file override")
    parser.add_argument("--model_data_path", type=str, default=None, help="Model directory override")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm override")
    parser.add_argument("--class_name", type=str, default=None, help="Dataset class override")
    parser.add_argument("--run_name", type=str, default=None, help="Training run name override")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=500, help="Maximum reference images to embed")
    parser.add_argument("--output", type=str, default="./drift/reference_embeddings.npy")
    return parser


def _collate_reference_batch(samples):
    """Batch transformed inputs while keeping original images as a list.

    AnodetDataset also returns the original PIL image as a tensor. Those images
    intentionally keep their native resolution, so stacking them with the
    default PyTorch collate function fails when reference images differ in size.
    Only the transformed model inputs need to be stacked for embedding.
    """
    batches, images, classifications, masks = zip(*samples)
    return (
        torch.stack(list(batches), dim=0),
        list(images),
        torch.as_tensor(classifications),
        torch.stack(list(masks), dim=0),
    )


def main(args: argparse.Namespace) -> None:
    cfg = load_config(str(args.config)) if args.config else {}
    config = edict(merge_config(args, cfg))

    if not config.get("img_path"):
        raise ValueError("img_path is required (healthy/reference images)")
    if not config.get("model"):
        raise ValueError("model is required")
    if not config.get("model_data_path"):
        config.model_data_path = "./distributions"
    if not config.get("algorithm"):
        raise ValueError("algorithm is required")
    if not config.get("class_name"):
        raise ValueError("class_name is required")
    if not config.get("run_name"):
        config.run_name = "anomav_exp"

    resize = _shape(config.get("resize", [224, 224]))
    crop_size = _shape(config.get("crop_size"))
    normalize = config.get("normalize", True)
    device = determine_device(config.get("device", "auto"))

    model_path = Path(config.model_data_path) / config.algorithm / config.class_name / config.run_name / config.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if int(args.max_samples) < 2:
        raise ValueError("max_samples must be at least 2")

    dataset = anomavision.AnodetDataset(
        str(config.img_path),
        resize=resize,
        crop_size=crop_size,
        normalize=normalize,
        mean=config.get("norm_mean", [0.485, 0.456, 0.406]),
        std=config.get("norm_std", [0.229, 0.224, 0.225]),
    )
    limit = min(len(dataset), int(args.max_samples))
    if limit < 2:
        raise ValueError("At least 2 reference images are required")

    class LimitedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return limit

        def __getitem__(self, index):
            return dataset[index]

    loader = DataLoader(
        LimitedDataset(),
        batch_size=int(config.get("batch_size", 2) or 2),
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=_collate_reference_batch,
    )

    model = ModelWrapper(str(model_path), device)
    embeddings = []
    seen = 0
    try:
        for batch, _, _, _ in loader:
            if device == "cuda":
                batch = batch.half()
            batch = batch.to(device)
            values = model.extract_drift_embeddings(batch)
            values = np.asarray(values, dtype=np.float32)
            if values.ndim != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {values.shape}")
            take = min(values.shape[0], limit - seen)
            embeddings.append(values[:take])
            seen += take
            if seen >= limit:
                break
    finally:
        model.close()

    reference = np.concatenate(embeddings, axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, reference)

    print("AnomaVision Drift Reference")
    print("=" * 28)
    print(f"Model:              {model_path}")
    print(f"Reference images:   {reference.shape[0]}")
    print(f"Embedding dimension: {reference.shape[1]}")
    print(f"Output:             {output}")
