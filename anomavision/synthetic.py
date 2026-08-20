"""CLI for generating production-style synthetic anomaly datasets."""

from __future__ import annotations

import argparse

from anomavision.synthetic_defects import generate_synthetic_dataset


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anomavision synthesize",
        description="Generate a reproducible synthetic defect dataset from normal images.",
        add_help=add_help,
    )
    parser.add_argument("--input_dir", required=True, help="Directory of normal images")
    parser.add_argument("--output_dir", required=True, help="Output dataset directory")
    parser.add_argument(
        "--defect_types",
        nargs="+",
        default=["scratch", "crack", "stain", "dent", "hole", "cutpaste"],
        help="Defect types to generate",
    )
    parser.add_argument(
        "--severity", choices=["low", "medium", "high"], default="medium"
    )
    parser.add_argument("--copies_per_type", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=10000)
    return parser


def main(args: argparse.Namespace) -> None:
    summary = generate_synthetic_dataset(
        args.input_dir,
        args.output_dir,
        defect_types=args.defect_types,
        severity=args.severity,
        copies_per_type=args.copies_per_type,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
    )
    print("Synthetic dataset created successfully.")
    for key, value in summary.items():
        print(f"{key}: {value}")
