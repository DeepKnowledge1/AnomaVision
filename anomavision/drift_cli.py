"""CLI implementation for AnomaVision data-drift monitoring."""

from __future__ import annotations

import argparse

from anomavision.drift import DriftMonitor, load_embeddings, save_report


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare reference and production embeddings for data drift.",
        add_help=add_help,
    )
    parser.add_argument("--reference", required=True, help="Reference embeddings (.npy/.npz).")
    parser.add_argument("--current", required=True, help="Current production window (.npy/.npz).")
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins per feature.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="PSI threshold for reporting drift (default: 0.20).",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    monitor = DriftMonitor.from_reference_file(
        args.reference,
        bins=args.bins,
        threshold=args.threshold,
    )
    report = monitor.compare(load_embeddings(args.current))

    print("AnomaVision Data Drift Report")
    print("=" * 30)
    print(f"Status:              {report.status.upper()}")
    print(f"Drift score:         {report.drift_score:.4f}")
    print(f"PSI:                 {report.psi:.4f}")
    print(f"Mean shift:          {report.mean_shift:.4f}")
    print(f"Std shift:           {report.std_shift:.4f}")
    print(f"Cosine shift:        {report.cosine_shift:.4f}")
    print(f"Reference samples:   {report.reference_samples}")
    print(f"Current samples:     {report.current_samples}")
    if report.warnings:
        print(f"Warnings:            {', '.join(report.warnings)}")

    if args.output:
        save_report(report, args.output)
        print(f"Report saved to:     {args.output}")
