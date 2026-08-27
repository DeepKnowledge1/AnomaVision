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


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AnomaVision anomaly detection models (args OR config).", add_help=add_help)
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config.yml/.json")
    parser.add_argument("--dataset_path", type=str, default=None, help='Path to the dataset folder containing "train/good" images.')
    parser.add_argument("--resize", type=int, nargs="*", default=None, metavar=("W", "H"), help="Resize before processing. Provide one value for a square resize or two values for width and height.")
    parser.add_argument("--crop_size", type=int, nargs="*", default=None, metavar=("W", "H"), help="Apply a center (or configured) crop.")
    parser.add_argument("--normalize", action="store_true", default=None, help="Enable input normalization.")
    parser.add_argument("--no_normalize", action="store_true", default=None, help="Disable input normalization explicitly.")
    parser.add_argument("--norm_mean", type=float, nargs=3, default=None, metavar=("R", "G", "B"), help="Per-channel RGB mean.")
    parser.add_argument("--norm_std", type=float, nargs=3, default=None, metavar=("R", "G", "B"), help="Per-channel RGB standard deviation.")
    parser.add_argument("--backbone", type=str, choices=["resnet18", "wide_resnet50"], default=None, help="Backbone network for PaDiM/PatchCore.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size used during training and inference.")
    parser.add_argument("--feat_dim", type=int, default=None, help="Number of random feature dimensions to keep.")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=None, help="List of feature layers to extract.")
    parser.add_argument("--coreset_ratio", type=float, default=None, help="PatchCore memory-bank fraction to retain.")
    parser.add_argument("--max_memory_patches", type=int, default=None, help="Maximum PatchCore memory-bank size.")
    parser.add_argument("--patch_grid", type=int, default=None, help="PatchCore pooled grid size.")
    parser.add_argument("--search_chunk_size", type=int, default=None, help="PatchCore query chunk size.")
    parser.add_argument("--coreset_method", type=str, choices=["kcenter", "random"], default=None, help="PatchCore coreset selection strategy.")
    parser.add_argument("--coreset_seed", type=int, default=None, help="Seed used for deterministic PatchCore coreset selection.")
    parser.add_argument("--efficientad_model_size", type=str, choices=["s", "m"], default=None, help="EfficientAD model size.")
    parser.add_argument("--efficientad_lr", type=float, default=None, help="EfficientAD learning rate.")
    parser.add_argument("--efficientad_weight_decay", type=float, default=None, help="EfficientAD weight decay.")
    parser.add_argument("--efficientad_epochs", type=int, default=None, help="EfficientAD training epochs.")
    parser.add_argument("--efficientad_pretrained_teacher", action="store_true", default=None, help="Use ImageNet-pretrained EfficientNet teacher.")
    parser.add_argument("--output_model", type=str, default=None, help="Filename to save the PT model.")
    parser.add_argument("--run_name", type=str, default=None, help="Experiment name for this training run.")
    parser.add_argument("--model_data_path", type=str, default=None, help="Directory to save model distributions and PT file.")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm name (padim, patchcore, efficientad).")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None, help="Logging level.")
    return parser


def run_training(args):
    cfg = load_config(args.config)
    config = edict(merge_config(args, cfg))
    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger("anomavision.train")

    algorithm = str(config.algorithm).lower()
    if algorithm not in {"padim", "patchcore", "efficientad"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Available: padim, patchcore, efficientad")
    if not config.dataset_path:
        raise ValueError("dataset_path is required (via --dataset_path or config)")
    if algorithm == "efficientad" and not bool(config.get("normalize", True)):
        raise ValueError("EfficientAD requires normalize: true because the teacher uses ImageNet preprocessing")

    t0 = time.perf_counter()
    logger.info("Image processing: resize=%s, crop_size=%s, normalize=%s", config.resize, config.crop_size, config.normalize)
    if config.normalize:
        logger.info("Normalization: mean=%s, std=%s", config.norm_mean, config.norm_std)

    run_dir = increment_path(
        Path(config.model_data_path) / algorithm / config.class_name / config.run_name,
        exist_ok=True,
        mkdir=True,
    )

    root = os.path.join(os.path.realpath(config.dataset_path), config.class_name, "train", "good")
    if not os.path.isdir(root):
        potential_root = os.path.join(os.path.realpath(config.dataset_path), "train", "good")
        if os.path.isdir(potential_root):
            root = potential_root
        else:
            logger.error('Expected folder "%s" does not exist.', root)
            raise FileNotFoundError(f"Dataset root not found: {root}")

    ds = anomavision.AnodetDataset(
        root,
        resize=config.resize,
        crop_size=config.crop_size,
        normalize=config.normalize,
        mean=config.norm_mean,
        std=config.norm_std,
    )
    if len(ds) == 0:
        raise ValueError(f"No training images found in {root}")

    dl = DataLoader(ds, batch_size=int(config.batch_size), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s (cuda_available=%s)", device.type, torch.cuda.is_available())
    logger.info("cfg: algorithm=%s | backbone=%s | layers=%s", algorithm, config.backbone, config.layer_indices)

    if algorithm == "patchcore":
        model = anomavision.PatchCore(
            backbone=config.backbone,
            device=device,
            layer_indices=config.layer_indices,
            coreset_ratio=float(config.coreset_ratio),
            max_memory_patches=config.max_memory_patches,
            patch_grid=config.patch_grid,
            search_chunk_size=config.search_chunk_size,
            coreset_method=config.get("coreset_method", "kcenter"),
            coreset_seed=int(config.get("coreset_seed", 42)),
        )
        model.fit(dl)
    elif algorithm == "efficientad":
        model = anomavision.EfficientAD(
            device=device,
            model_size=config.get("efficientad_model_size", "s"),
            lr=float(config.get("efficientad_lr", 1e-4)),
            weight_decay=float(config.get("efficientad_weight_decay", 1e-5)),
            pretrained_teacher=bool(config.get("efficientad_pretrained_teacher", True)),
        )
        t_fit = time.perf_counter()
        model.fit(dl, epochs=int(config.get("efficientad_epochs", 1)))
        logger.info("fit: completed in %.2fs", time.perf_counter() - t_fit)
    else:
        model = anomavision.Padim(
            backbone=config.backbone,
            device=device,
            layer_indices=config.layer_indices,
            feat_dim=int(config.feat_dim),
        )
        t_fit = time.perf_counter()
        model.fit(dl)
        logger.info("fit: completed in %.2fs", time.perf_counter() - t_fit)

    model_path = Path(run_dir) / config.output_model
    torch.save(model, str(model_path))

    stats_path = model_path.with_suffix(".pth")
    try:
        model.save_statistics(str(stats_path), half=True)
        logger.info("saved: slim statistics=%s", stats_path)
    except Exception as e:
        logger.warning("saving slim statistics failed: %s", e)

    save_args_to_yaml(config, str(Path(run_dir) / "config.yml"))
    logger.info("saved: model=%s, config=%s", model_path, Path(run_dir) / "config.yml")
    logger.info("=== Training done in %.2fs ===", time.perf_counter() - t0)
    return model, config, run_dir, {"train": dl}


def main(args=None):
    try:
        if args is None:
            args = create_parser().parse_args()
        try:
            checker = GitStatusChecker()
            if checker.is_repo():
                checker.check_status()
        except Exception:
            pass
        run_training(args)
    except Exception:
        get_logger(__name__).exception("Fatal error during training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
