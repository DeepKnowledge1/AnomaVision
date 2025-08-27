import os
import sys
import time
import argparse

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from pathlib import Path

import anodet
from anodet.utils import get_logger, save_args_to_yaml, setup_logging
from anodet.general import increment_path

# pre-commit run trailing-whitespace --files .\anodet\utils.py

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PaDiM model for anomaly detection."
    )

    parser.add_argument(
        "--dataset_path",
        default=r"D:\01-DATA\bottle",
        type=str,
        required=False,
        help='Path to the dataset folder containing "train/good" images.',
    )

    parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions/",
        help="Directory to save model distributions and PT file.",
    )

    parser.add_argument(
        "--run_name",
        default="anomav_exp",
        help="experiment name for this training run"
    )

    parser.add_argument(
        "--overwrite",
        action="store_false",
        help="overwrite existing run directory without auto-incrementing",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "wide_resnet50"],
        default="resnet18",
        help="Backbone network to use for feature extraction.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size used during training and inference.",
    )

    parser.add_argument(
        "--output_model",
        type=str,
        default="padim_model.pt",
        help="Filename to save the PT model.",
    )

    parser.add_argument(
        "--layer_indices",
        nargs="+",
        type=int,
        default=[0],
        help="List of layer indices to extract features from. Default: [0].",
    )

    parser.add_argument(
        "--feat_dim",
        type=int,
        default=50,
        help="Number of random feature dimensions to keep.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO).",
    )

    return parser.parse_args()


def main(args):
    setup_logging(args.log_level)
    logger = get_logger(__name__)

    t0 = time.perf_counter()
    try:
        # Resolve paths
        DATASET_PATH = os.path.realpath(args.dataset_path)
        MODEL_DATA_PATH = increment_path(Path(args.model_data_path) / args.run_name, exist_ok=args.overwrite, mkdir=True)

        # Minimal, high-signal run header
        logger.info("=== PaDiM training started ===")
        logger.info(
            "cfg: backbone=%s | layers=%s | feat_dim=%d | batch_size=%d | out=%s",
            args.backbone,
            args.layer_indices,
            args.feat_dim,
            args.batch_size,
            os.path.join(MODEL_DATA_PATH, args.output_model),
        )
        logger.info("data: %s", DATASET_PATH)

        # Dataset
        dataset_root = os.path.join(DATASET_PATH, "train", "good")
        if not os.path.isdir(dataset_root):
            logger.error('Expected folder "%s" does not exist.', dataset_root)
            sys.exit(1)

        dataset = anodet.AnodetDataset(dataset_root)
        if len(dataset) == 0:
            logger.error("No training images found in %s", dataset_root)
            sys.exit(1)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        logger.info("dataset: %d images", len(dataset))

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("device: %s (cuda_available=%s)", device.type, torch.cuda.is_available())

        # Model
        padim = anodet.Padim(
            backbone=args.backbone,
            device=device,
            layer_indices=args.layer_indices,
            feat_dim=args.feat_dim,
        )

        # Train
        t_fit = time.perf_counter()
        padim.fit(dataloader)
        logger.info("fit: completed in %.2fs", time.perf_counter() - t_fit)

        # Save artifacts
        model_path = os.path.join(MODEL_DATA_PATH, args.output_model)
        torch.save(padim, model_path)
        save_args_to_yaml(args, os.path.join(MODEL_DATA_PATH, "config.yml"))
        logger.info("saved: model=%s, config=%s",
                    model_path, os.path.join(MODEL_DATA_PATH, "config.yml"))

        logger.info("=== Training is done in %.2fs ===", time.perf_counter() - t0)

    except Exception:
        logger.exception("Fatal error during training.")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
