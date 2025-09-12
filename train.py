# train.py
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import anomavision
from anomavision.config import load_config
from anomavision.general import GitStatusChecker, increment_path
from anomavision.utils import get_logger, merge_config, save_args_to_yaml, setup_logging

checker = GitStatusChecker()
checker.check_status()


def parse_args():
    parser = argparse.ArgumentParser(description="Train PaDiM (args OR config).")
    # meta
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to config.yml/.json"
    )
    # dataset
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help='Path to the dataset folder containing "train/good" images.',
    )

    # preprocessing
    parser.add_argument(
        "--resize",
        type=int,
        nargs="*",
        default=None,
        metavar=("W", "H"),
        help="Resize before processing. Provide one value for a square resize (e.g., 256) or two values for width and height (e.g., 256 192). Omit to keep original size.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="*",
        default=None,
        metavar=("W", "H"),
        help="Apply a center (or configured) crop. One value for a square crop (e.g., 224) or two for width and height (e.g., 224 224). Omit to disable cropping.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=None,
        help="Enable input normalization. If set, inputs are normalized using --norm_mean/--norm_std if provided (commonly ImageNet stats).",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        default=None,
        help="Disable input normalization explicitly. If both --normalize and --no_normalize are set, this flag should take precedence in your code.",
    )
    parser.add_argument(
        "--norm_mean",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Per-channel RGB mean used when normalization is enabled. Example: 0.485 0.456 0.406.",
    )
    parser.add_argument(
        "--norm_std",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Per-channel RGB standard deviation used when normalization is enabled. Example: 0.229 0.224 0.225.",
    )

    # train
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "wide_resnet50"],
        default=None,
        help="Backbone network to use for feature extraction.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size used during training and inference.",
    )
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=None,
        help="Number of random feature dimensions to keep.",
    )
    parser.add_argument(
        "--layer_indices",
        type=int,
        nargs="+",
        default=None,
        help="List of layer indices to extract features from, e.g., 0 1 2.",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=None,
        help="Filename to save the PT model.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Experiment name for this training run.",
    )
    parser.add_argument(
        "--model_data_path",
        type=str,
        default=None,
        help="Directory to save model distributions and PT file.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level (default: INFO).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Merge config with CLI args
    config = edict(merge_config(args, cfg))

    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger(__name__)

    if not config.dataset_path:
        logger.error(
            "dataset.path is required (via --dataset_path or config.common.dataset_path)"
        )
        sys.exit(1)

    t0 = time.perf_counter()
    try:
        logger.info(
            "Image processing: resize=%s, crop_size=%s, normalize=%s",
            config.resize,
            config.crop_size,
            config.normalize,
        )
        if config.normalize:
            logger.info(
                "Normalization: mean=%s, std=%s", config.norm_mean, config.norm_std
            )

        # Resolve output run dir once
        run_dir = increment_path(
            Path(config.model_data_path) / config.run_name, exist_ok=True, mkdir=True
        )

        # === Dataset ===
        root = os.path.join(
            os.path.realpath(config.dataset_path), config.class_name, "train", "good"
        )
        if not os.path.isdir(root):
            logger.error('Expected folder "%s" does not exist.', root)
            sys.exit(1)

        ds = anomavision.AnodetDataset(
            root,
            resize=config.resize,
            crop_size=config.crop_size,
            normalize=config.normalize,
            mean=config.norm_mean,
            std=config.norm_std,
        )  # uses your existing dataset class signature.

        if len(ds) == 0:
            logger.error("No training images found in %s", root)
            sys.exit(1)

        dl = DataLoader(ds, batch_size=int(config.batch_size), shuffle=False)
        logger.info("dataset: %d images | batch_size=%d", len(ds), config.batch_size)

        # === Device ===
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "device: %s (cuda_available=%s)", device.type, torch.cuda.is_available()
        )

        # === Model & Train ===
        logger.info(
            "cfg: backbone=%s | layers=%s | feat_dim=%d",
            config.backbone,
            config.layer_indices,
            config.feat_dim,
        )

        padim = anomavision.Padim(
            backbone=config.backbone,
            device=device,
            layer_indices=config.layer_indices,
            feat_dim=int(config.feat_dim),
        )

        t_fit = time.perf_counter()
        padim.fit(dl)
        logger.info("fit: completed in %.2fs", time.perf_counter() - t_fit)

        # === Save ===
        model_path = Path(run_dir) / config.output_model
        torch.save(padim, str(model_path))

        # also save a compact stats-only artifact (anomalib-style) -> ".pth"
        stats_path = model_path.with_suffix(".pth")
        try:
            padim.save_statistics(str(stats_path), half=True)
            logger.info("saved: slim statistics=%s", stats_path)
        except Exception as e:
            logger.warning("saving slim statistics failed: %s", e)

        # snapshot the effective configuration
        # save_args_to_yaml(config, str(Path(run_dir) / "config.yml"))
        save_args_to_yaml(config, str(Path(run_dir) / "config.yml"))

        logger.info(
            "saved: model=%s, config=%s", model_path, Path(run_dir) / "config.yml"
        )
        logger.info("=== Training done in %.2fs ===", time.perf_counter() - t0)

    except Exception:
        get_logger(__name__).exception("Fatal error during training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
