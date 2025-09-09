# tests/conftest.py
"""
Pytest configuration and shared fixtures for AnomaVision tests.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import tempfile
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

import anodet

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _create_normal_image() -> Image.Image:
    """Generate a synthetic 'normal' image (mostly uniform)."""
    return Image.fromarray(np.random.randint(100, 150, (224, 224, 3), dtype=np.uint8))


def _create_anomaly_image() -> Image.Image:
    """Generate a synthetic 'anomaly' image with a bright patch."""
    img_array = np.random.randint(100, 150, (224, 224, 3), dtype=np.uint8)
    img_array[50:100, 50:100] = np.random.randint(200, 255, (50, 50, 3))
    return Image.fromarray(img_array)


def _create_anomaly_mask(size: int = 224) -> Image.Image:
    """Generate a binary mask for the anomalous region."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[50:100, 50:100] = 255
    return Image.fromarray(mask)


# ---------------------------------------------------------------------------
# Fixtures: Directories & Datasets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_device() -> torch.device:
    """Provide test device (CPU for CI compatibility)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def sample_images_dir() -> Iterator[Path]:
    """Create temporary directory with sample normal and anomalous images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save normal and anomalous images
        for i in range(3):
            _create_normal_image().save(temp_path / f"normal_{i}.png")
        for i in range(2):
            _create_anomaly_image().save(temp_path / f"anomaly_{i}.png")

        yield temp_path


@pytest.fixture(scope="session")
def mvtec_structure(sample_images_dir: Path) -> Iterator[Path]:
    """Create MVTec-like directory structure with sample images & masks."""
    mvtec_root = sample_images_dir / "mvtec_test"
    bottle_dir = mvtec_root / "bottle"

    # Create required directories
    train_good = bottle_dir / "train" / "good"
    test_good = bottle_dir / "test" / "good"
    test_broken = bottle_dir / "test" / "broken_large"
    gt_dir = bottle_dir / "ground_truth" / "broken_large"
    for d in (train_good, test_good, test_broken, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Populate training and test sets
    for i, img_file in enumerate(sample_images_dir.glob("normal_*.png")):
        img = Image.open(img_file)
        if i < 2:
            img.save(train_good / f"train_{i:03d}.png")
        img.save(test_good / f"test_{i:03d}.png")

    for i, img_file in enumerate(sample_images_dir.glob("anomaly_*.png")):
        img = Image.open(img_file)
        img.save(test_broken / f"broken_{i:03d}.png")
        _create_anomaly_mask().save(gt_dir / f"broken_{i:03d}_mask.png")

    yield mvtec_root


@pytest.fixture
def sample_dataset(sample_images_dir: Path) -> anodet.AnodetDataset:
    """Create a sample AnodetDataset."""
    return anodet.AnodetDataset(
        str(sample_images_dir),
        resize=[224, 224],
        crop_size=[224, 224],
        normalize=True,
    )


@pytest.fixture
def sample_dataloader(sample_dataset: anodet.AnodetDataset) -> DataLoader:
    """Create a sample DataLoader."""
    return DataLoader(sample_dataset, batch_size=2, shuffle=False)


@pytest.fixture
def mvtec_dataset(mvtec_structure: Path) -> anodet.MVTecDataset:
    """Create a sample MVTecDataset."""
    return anodet.MVTecDataset(
        str(mvtec_structure),
        class_name="bottle",
        is_train=True,
        resize=[224, 224],
        crop_size=[224, 224],
        normalize=True,
    )


# ---------------------------------------------------------------------------
# Fixtures: Models
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_padim_model(sample_dataloader: DataLoader, test_device: torch.device):
    """Train a minimal PaDiM model for testing."""
    model = anodet.Padim(
        backbone="resnet18",
        device=test_device,
        layer_indices=[0],
        feat_dim=10,  # small for speed
    )
    model.fit(sample_dataloader)
    return model


@pytest.fixture
def sample_batch(sample_dataloader: DataLoader):
    """Get a single batch for testing."""
    return next(iter(sample_dataloader))


@pytest.fixture(scope="session")
def temp_model_dir() -> Iterator[Path]:
    """Temporary directory for saving/loading models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# ---------------------------------------------------------------------------
# Test configuration constants
# ---------------------------------------------------------------------------

TEST_CONFIG = {
    "batch_size": 2,
    "image_size": 224,
    "num_test_images": 5,
    "feat_dim": 10,
    "tolerance": 1e-6,
}


@pytest.fixture
def test_config() -> dict:
    """Provide test configuration constants."""
    return TEST_CONFIG


# ---------------------------------------------------------------------------
# Utility helpers for tests
# ---------------------------------------------------------------------------


class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]):
        assert (
            tensor.shape == expected_shape
        ), f"Expected {expected_shape}, got {tensor.shape}"

    @staticmethod
    def assert_tensor_range(tensor: torch.Tensor, min_val: float, max_val: float):
        assert tensor.min() >= min_val, f"Tensor min {tensor.min()} < {min_val}"
        assert tensor.max() <= max_val, f"Tensor max {tensor.max()} > {max_val}"

    @staticmethod
    def create_temp_model_file(
        model, temp_dir: Path, filename: str = "test_model.pt"
    ) -> Path:
        model_path = temp_dir / filename
        torch.save(model, str(model_path))
        return model_path


@pytest.fixture
def test_utils() -> TestUtils:
    """Provide test utility functions."""
    return TestUtils


# ---------------------------------------------------------------------------
# Fixtures: PaDiM Lite patching
# ---------------------------------------------------------------------------


class DummyExtractor:
    """Tiny stand-in for ResnetEmbeddingsExtractor used in tests."""

    def __init__(self, backbone_name: str, device: torch.device):
        self.backbone_name = backbone_name
        self.device = device

    def __call__(self, batch, *, channel_indices, layer_indices, layer_hook=None):
        B = batch.shape[0]
        N, D = getattr(self, "_N"), getattr(self, "_D")
        torch.manual_seed(0)
        emb = torch.randn(B, N, D, device=self.device, dtype=torch.float32)

        W, H = getattr(self, "_W"), getattr(self, "_H")
        if W * H != N:
            W = int(math.sqrt(N))
            H = N // W
        return emb, W, H


@pytest.fixture
def make_stats():
    """Create a tiny valid PaDiM stats dict on CPU with fp16 storage."""

    def _factory(N=6, D=4, W=3, H=2):
        assert W * H == N, "W*H must equal N for a valid map."
        torch.manual_seed(1)
        mean = torch.randn(N, D)
        L = torch.randn(N, D, D)
        cov = L @ L.transpose(-1, -2) + 1e-3 * torch.eye(D).expand(N, D, D)
        cov_inv = torch.inverse(cov)

        stats = {
            "mean": mean.half().cpu(),
            "cov_inv": cov_inv.half().cpu(),
            "channel_indices": torch.arange(D, dtype=torch.int32),
            "layer_indices": [2, 3],
            "backbone": "resnet18",
            "model_version": "1.0",
        }
        return stats, (N, D, W, H)

    return _factory


@pytest.fixture
def patch_extractor(monkeypatch):
    """Monkeypatch DummyExtractor into anodet.padim_lite."""
    from importlib import import_module

    padim_lite = import_module("anodet.padim_lite")

    def _apply(N, D, W, H):
        class _Extractor(DummyExtractor):
            def __init__(self, backbone_name, device):
                super().__init__(backbone_name, device)
                self._N, self._D, self._W, self._H = N, D, W, H

        monkeypatch.setattr(
            padim_lite, "ResnetEmbeddingsExtractor", _Extractor, raising=True
        )
        return _Extractor

    return _apply
