"""Generate trusted reference embeddings for production drift monitoring.

The reference builder is deliberately outside the anomaly algorithms. It uses
normal AnomaVision inference and captures the same representation exposed to
production drift monitoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset

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
    """Batch transformed inputs while preserving native-resolution images."""
    batches, images, classifications, masks = zip(*samples)
    return (
        torch.stack(list(batches), dim=0),
        list(images),
        torch.as_tensor(classifications),
        torch.stack(list(masks), dim=0),
    )


class _LimitedDataset(Dataset):
    def __init__(self, dataset, limit: int):
        self.dataset = dataset
        self.limit = limit

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        return self.dataset[index]


def main(args: argparse.Namespace | None = None) -> None:
    args = args or create_parser().parse_args()
    cfg = load_config(str(args.config)) if args.config else {}
    config = edict(merge_config(args, cfg))

    required = ("img_path", "model", "algorithm", "class_name", "run_name")
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    if not config.get("model_data_path"):
        config.model_data_path = "./distributions"

    max_samples = int(args.max_samples)
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")

    resize = _shape(config.get("resize", [224, 224]))
    crop_size = _shape(config.get("crop_size"))
    dataset = anomavision.AnodetDataset(
        str(config.img_path),
        resize=resize,
        crop_size=crop_size,
        normalize=config.get("normalize", True),
        mean=config.get("norm_mean", [0.485, 0.456, 0.406]),
        std=config.get("norm_std", [0.229, 0.224, 0.225]),
    )
    limit = min(len(dataset), max_samples)
    if limit < 2:
        raise ValueError("At least 2 reference images are required")

    loader = DataLoader(
        _LimitedDataset(dataset, limit),
        batch_size=int(config.get("batch_size", 2) or 2),
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=_collate_reference_batch,
    )

    device = determine_device(config.get("device", "auto"))
    model_path = (
        Path(config.model_data_path)
        / config.algorithm
        / config.class_name
        / config.run_name
        / config.model
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = ModelWrapper(str(model_path), device)
    chunks: list[np.ndarray] = []
    seen = 0
    try:
        for batch, _, _, _ in loader:
            inference_batch = batch.half() if device == "cuda" else batch
            inference_batch = inference_batch.to(device)

            # Run the normal inference path first. Backends such as PatchCore
            # can then reuse the exact representation captured by prediction.
            model.predict(inference_batch)
            values = np.asarray(
                model.extract_drift_embeddings(inference_batch), dtype=np.float32
            )
            if values.ndim != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError("Generated embeddings contain NaN or infinite values")

            take = min(values.shape[0], limit - seen)
            if take:
                chunks.append(values[:take])
                seen += take
            if seen >= limit:
                break
    finally:
        model.close()

    if not chunks:
        raise ValueError("No drift reference embeddings were generated")

    reference = np.concatenate(chunks, axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, reference)

    metadata = {
        "samples": int(reference.shape[0]),
        "feature_dimensions": int(reference.shape[1]),
        "model_path": str(model_path),
        "algorithm": str(config.algorithm),
        "class_name": str(config.class_name),
        "run_name": str(config.run_name),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("AnomaVision Drift Reference")
    print("=" * 28)
    print(f"Model:               {model_path}")
    print(f"Reference samples:   {reference.shape[0]}")
    print(f"Embedding dimension: {reference.shape[1]}")
    print(f"Output:              {output}")


if __name__ == "__main__":
    main()
