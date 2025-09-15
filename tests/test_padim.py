# tests/test_padim.py
"""
Tests for core PaDiM functionality.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import anomavision


class TestPadimCore:
    """Test core PaDiM functionality."""

    def test_padim_initialization(self, test_device):
        """Test PaDiM model initialization."""
        model = anomavision.Padim(
            backbone="resnet18", device=test_device, layer_indices=[0, 1], feat_dim=50
        )

        assert model.device == test_device
        assert model.layer_indices == [0, 1]
        assert model.embeddings_extractor.backbone_name == "resnet18"

    def test_padim_training(self, sample_dataloader, test_device):
        """Test PaDiM model training."""
        model = anomavision.Padim(
            backbone="resnet18", device=test_device, layer_indices=[0], feat_dim=10
        )

        # Before training
        assert not hasattr(model, "mahalanobisDistance")

        # Train model
        model.fit(sample_dataloader)

        # After training
        assert hasattr(model, "mahalanobisDistance")
        assert model.mahalanobisDistance._mean_flat is not None
        assert model.mahalanobisDistance._cov_inv_flat is not None

    def test_padim_prediction(self, trained_padim_model, sample_batch):
        """Test PaDiM model prediction."""
        batch, _, _, _ = sample_batch

        # Run prediction
        image_scores, score_maps = trained_padim_model.predict(batch)

        # Check output shapes
        assert image_scores.shape == (batch.size(0),)
        assert score_maps.shape == (batch.size(0), batch.size(2), batch.size(3))

        # Check output types
        assert isinstance(image_scores, torch.Tensor)
        assert isinstance(score_maps, torch.Tensor)

        # Check output ranges (scores should be positive)
        assert torch.all(image_scores >= 0)
        assert torch.all(score_maps >= 0)

    # def test_padim_evaluation(self, trained_padim_model, sample_dataloader):
    #     """Test PaDiM model evaluation."""
    #     results = trained_padim_model.evaluate(sample_dataloader)
    #     images, targets, masks, scores, maps = results

    #     # Check that we get results
    #     assert len(images) > 0
    #     assert len(targets) > 0
    #     assert len(scores) > 0
    #     assert len(maps) > 0

    #     # Check shapes match
    #     assert len(images) == len(targets) == len(scores)
    #     assert len(maps) == np.prod([len(images), images[0].shape[1], images[0].shape[2]])


class TestPadimStatistics:
    """Test PaDiM statistics saving/loading functionality."""

    def test_save_statistics(self, trained_padim_model, temp_model_dir):
        """Test saving model statistics."""
        stats_path = temp_model_dir / "test_stats.pth"

        # Save statistics
        trained_padim_model.save_statistics(str(stats_path))

        # Check file exists
        assert stats_path.exists()

        # Load and check contents
        stats = torch.load(str(stats_path), weights_only=False)
        expected_keys = {
            "mean",
            "cov_inv",
            "channel_indices",
            "layer_indices",
            "backbone",
        }
        assert expected_keys.issubset(stats.keys())

    def test_load_statistics(self, trained_padim_model, temp_model_dir, test_device):
        """Test loading model statistics."""
        stats_path = temp_model_dir / "test_stats.pth"

        # Save statistics
        trained_padim_model.save_statistics(str(stats_path))

        # Load statistics
        stats = anomavision.Padim.load_statistics(
            str(stats_path), device=str(test_device)
        )

        # Check types and shapes
        assert isinstance(stats["mean"], torch.Tensor)
        assert isinstance(stats["cov_inv"], torch.Tensor)
        assert isinstance(stats["channel_indices"], torch.Tensor)
        assert isinstance(stats["layer_indices"], list)
        assert isinstance(stats["backbone"], str)


class TestPadimDimensions:
    """Test PaDiM with different image dimensions."""

    @pytest.mark.parametrize(
        "resize,crop_size",
        [
            ([224, 224], [224, 224]),
            ([256, 192], [224, 224]),
            ([320, 240], [224, 224]),
            ([256, 256], None),
        ],
    )
    def test_flexible_dimensions(
        self, sample_images_dir, test_device, resize, crop_size
    ):
        """Test PaDiM with different image dimensions."""
        # Create dataset with custom dimensions
        dataset = anomavision.AnodetDataset(
            str(sample_images_dir), resize=resize, crop_size=crop_size, normalize=True
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Create and train model
        model = anomavision.Padim(
            backbone="resnet18", device=test_device, layer_indices=[0], feat_dim=10
        )

        model.fit(dataloader)

        # Test prediction
        batch, _, _, _ = next(iter(dataloader))
        image_scores, score_maps = model.predict(batch)

        assert image_scores.shape == (batch.size(0),)
        assert score_maps.shape == (batch.size(0), batch.size(2), batch.size(3))


class TestPadimBackbones:
    """Test PaDiM with different backbones."""

    @pytest.mark.parametrize("backbone", ["resnet18", "wide_resnet50"])
    def test_different_backbones(self, sample_dataloader, test_device, backbone):
        """Test PaDiM with different backbone architectures."""
        model = anomavision.Padim(
            backbone=backbone, device=test_device, layer_indices=[0], feat_dim=10
        )

        # Should initialize without error
        assert model.embeddings_extractor.backbone_name == backbone

        # Should train without error
        model.fit(sample_dataloader)

        # Should predict without error
        batch, _, _, _ = next(iter(sample_dataloader))
        image_scores, score_maps = model.predict(batch)

        assert image_scores.shape == (batch.size(0),)


class TestPadimClassification:
    """Test classification functionality."""

    def test_classification_function(self):
        """Test the classification utility function."""
        # Create test scores
        scores = torch.tensor([5.0, 15.0, 8.0, 20.0])
        threshold = 10.0

        # Run classification
        classifications = anomavision.classification(scores, threshold)

        # Check results (below threshold = normal=1, above = anomaly=0)
        expected = torch.tensor([1, 0, 1, 0])
        assert torch.equal(classifications, expected)

    def test_classification_with_numpy(self):
        """Test classification with numpy arrays."""
        scores = np.array([5.0, 15.0, 8.0, 20.0])
        threshold = 10.0

        classifications = anomavision.classification(scores, threshold)

        expected = np.array([1, 0, 1, 0])
        np.testing.assert_array_equal(classifications, expected)


class TestPadimMemoryEfficiency:
    """Test memory-efficient operations."""

    # def test_memory_efficient_evaluation(self, trained_padim_model, sample_dataloader):
    #     """Test memory-efficient evaluation."""
    #     # This should not raise any memory errors
    #     results = trained_padim_model.evaluate_memory_efficient(sample_dataloader)
    #     images, targets, masks, scores, maps = results

    #     # Should get same structure as regular evaluation
    #     assert len(images) > 0
    #     assert len(targets) > 0
    #     assert len(scores) > 0
    #     assert len(maps) > 0

    def test_device_conversion(self, test_device):
        """Test device conversion functionality."""
        model = anomavision.Padim(
            backbone="resnet18",
            device=torch.device("cpu"),
            layer_indices=[0],
            feat_dim=10,
        )

        # Convert to test device
        model.to_device(test_device)
        assert model.device == test_device
