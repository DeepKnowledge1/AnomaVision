"""Config-driven pure inference performance regression tests.

These tests intentionally measure only ``model.predict(batch)``. Image loading,
preprocessing, postprocessing, visualization, and result accumulation are kept
outside the timed section so the numbers match the ``Pure inference FPS`` and
``Average inference time`` reported by ``anomavision.detect``.

Enable them in ``config.yml`` with ``inference_benchmark.enabled: true``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

import anomavision
from anomavision.config import load_config
from anomavision.general import determine_device
from anomavision.inference.model.wrapper import ModelWrapper


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yml"
ALGORITHMS = ("padim", "patchcore", "efficientad")


def _cuda_sync(device: str) -> None:
    """Synchronize CUDA before/after a timed inference call."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_benchmark_config() -> dict:
    config = load_config(str(CONFIG_PATH))
    benchmark = config.get("inference_benchmark", {})
    if not benchmark.get("enabled", False):
        pytest.skip(
            "Inference benchmark disabled. Set inference_benchmark.enabled: true in config.yml."
        )
    return config


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_inference_performance(algorithm: str) -> None:
    """Benchmark PaDiM, PatchCore, and EfficientAD independently."""
    config = _load_benchmark_config()
    benchmark = config["inference_benchmark"]

    device = determine_device(str(config.get("device", "auto")))
    model_path = (
        Path(config["model_data_path"])
        / algorithm
        / config["class_name"]
        / config["run_name"]
        / config["model"]
    )
    dataset_path = Path(config["img_path"])

    if not model_path.exists():
        pytest.skip(f"{algorithm}: model not found: {model_path}")
    if not dataset_path.exists():
        pytest.skip(f"{algorithm}: dataset not found: {dataset_path}")

    batch_size = int(benchmark.get("batch_size", config.get("batch_size", 1)))
    warmup_runs = int(benchmark.get("warmup_runs", 10))
    test_runs = int(benchmark.get("test_runs", 50))

    if batch_size < 1 or warmup_runs < 0 or test_runs < 1:
        raise ValueError(
            "inference_benchmark.batch_size >= 1, warmup_runs >= 0, test_runs >= 1"
        )

    dataset = anomavision.AnodetDataset(
        str(dataset_path),
        resize=config.get("resize"),
        crop_size=config.get("crop_size"),
        normalize=config.get("normalize", True),
        mean=config.get("norm_mean"),
        std=config.get("norm_std"),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    try:
        first = next(iter(dataloader))
    except StopIteration:
        pytest.fail(f"{algorithm}: benchmark dataset is empty: {dataset_path}")

    batch = first[0]
    if device == "cuda":
        batch = batch.half()
    batch = batch.to(device)

    model = ModelWrapper(str(model_path), device)
    try:
        for _ in range(warmup_runs):
            model.predict(batch)
        _cuda_sync(device)

        timings_ms = []
        for _ in range(test_runs):
            _cuda_sync(device)
            start = time.perf_counter()
            model.predict(batch)
            _cuda_sync(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

        average_ms = sum(timings_ms) / len(timings_ms)
        pure_fps = batch_size * 1000.0 / average_ms

        max_ms = benchmark.get("max_inference_ms", {}).get(algorithm)
        baseline_ms = benchmark.get("baseline_inference_ms", {}).get(algorithm)
        max_regression = float(benchmark.get("max_regression_percent", 25.0))

        allowed_ms = None if max_ms is None else float(max_ms)
        if baseline_ms is not None:
            regression_limit = float(baseline_ms) * (1.0 + max_regression / 100.0)
            allowed_ms = (
                regression_limit
                if allowed_ms is None
                else min(allowed_ms, regression_limit)
            )

        print(
            f"\n{algorithm.upper()} | "
            f"Pure inference FPS: {pure_fps:.2f} images/sec | "
            f"Average inference time: {average_ms:.2f} ms/batch | "
            f"Throughput: {pure_fps:.2f} images/sec (batch size: {batch_size})"
        )

        if allowed_ms is not None:
            assert average_ms <= allowed_ms, (
                f"{algorithm} inference regression: {average_ms:.2f} ms/batch "
                f"> allowed {allowed_ms:.2f} ms/batch."
            )
    finally:
        model.close()
