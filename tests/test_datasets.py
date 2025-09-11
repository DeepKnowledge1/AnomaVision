# tests/test_datasets.py
"""
Tests for dataset loading functionality.
"""
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

import anomavision


class TestAnodetDataset:
    """Test AnodetDataset functionality."""

    def test_dataset_initialization(self, sample_images_dir):
        """Test basic dataset initialization."""
        dataset = anodet.AnodetDataset(
            str(sample_images_dir),
            resize=[224, 224],
            crop_size=[224, 224],
            normalize=True,
        )

        assert len(dataset) == 5  # 3 normal + 2 anomaly images
        assert dataset.image_directory_path == str(sample_images_dir)

    def test_dataset_getitem(self, sample_dataset):
        """Test dataset __getitem__ method."""
        batch, image, classification, mask = sample_dataset[0]

        # Check types
        assert isinstance(batch, torch.Tensor)
        assert isinstance(image, np.ndarray)
        assert isinstance(classification, int)
        assert isinstance(mask, torch.Tensor)

        # Check shapes
        assert batch.shape == (3, 224, 224)  # C, H, W
        assert image.shape == (224, 224, 3)  # H, W, C
        assert mask.shape == (1, 224, 224)

    @pytest.mark.parametrize(
        "resize,crop_size,expected_shape",
        [
            ([224, 224], [224, 224], (3, 224, 224)),
            ([256, 192], [256, 192], (3, 256, 192)),
            ([320, 240], None, (3, 320, 240)),
            (256, [224, 224], (3, 224, 224)),
        ],
    )
    def test_flexible_dimensions(
        self, sample_images_dir, resize, crop_size, expected_shape
    ):
        """Test dataset with different image dimensions."""
        dataset = anodet.AnodetDataset(
            str(sample_images_dir), resize=resize, crop_size=crop_size, normalize=True
        )

        batch, _, _, _ = dataset[0]
        assert batch.shape == expected_shape

    def test_normalization_options(self, sample_images_dir):
        """Test dataset with and without normalization."""
        # With normalization
        dataset_norm = anodet.AnodetDataset(str(sample_images_dir), normalize=True)

        # Without normalization
        dataset_no_norm = anodet.AnodetDataset(str(sample_images_dir), normalize=False)

        batch_norm, _, _, _ = dataset_norm[0]
        batch_no_norm, _, _, _ = dataset_no_norm[0]

        # Normalized should be roughly in [-2, 2] range
        assert batch_norm.min() > -3 and batch_norm.max() < 3

        # Non-normalized should be in [0, 1] range
        assert batch_no_norm.min() >= 0 and batch_no_norm.max() <= 1

    def test_custom_normalization(self, sample_images_dir):
        """Test dataset with custom normalization parameters."""
        custom_mean = [0.5, 0.5, 0.5]
        custom_std = [0.5, 0.5, 0.5]

        dataset = anodet.AnodetDataset(
            str(sample_images_dir), normalize=True, mean=custom_mean, std=custom_std
        )

        batch, _, _, _ = dataset[0]

        # With these parameters, values should be roughly in [-1, 1]
        assert batch.min() > -2 and batch.max() < 2


class TestMVTecDataset:
    """Test MVTecDataset functionality."""

    def test_mvtec_dataset_train(self, mvtec_structure):
        """Test MVTec dataset in training mode."""
        dataset = anodet.MVTecDataset(
            str(mvtec_structure),
            class_name="bottle",
            is_train=True,
            resize=[224, 224],
            crop_size=[224, 224],
            normalize=True,
        )

        assert len(dataset) > 0

        # All training samples should be normal (classification = 0)
        for i in range(len(dataset)):
            _, _, classification, _ = dataset[i]
            assert classification == 0

    def test_mvtec_dataset_test(self, mvtec_structure):
        """Test MVTec dataset in test mode."""
        dataset = anodet.MVTecDataset(
            str(mvtec_structure),
            class_name="bottle",
            is_train=False,
            resize=[224, 224],
            crop_size=[224, 224],
            normalize=True,
        )

        assert len(dataset) > 0

        # Should have both normal and anomalous samples
        classifications = []
        for i in range(len(dataset)):
            _, _, classification, _ = dataset[i]
            classifications.append(classification)

        # Should have both 0 (normal) and 1 (anomalous)
        assert 0 in classifications
        assert 1 in classifications

    def test_mvtec_dataset_masks(self, mvtec_structure):
        """Test MVTec dataset mask handling."""
        dataset = anodet.MVTecDataset(
            str(mvtec_structure),
            class_name="bottle",
            is_train=False,
            resize=[224, 224],
            crop_size=[224, 224],
            normalize=True,
        )

        for i in range(len(dataset)):
            batch, _, classification, mask = dataset[i]

            if classification == 0:  # Normal image
                # Mask should be all zeros
                assert torch.all(mask == 0)
            else:  # Anomalous image
                # Mask should have some non-zero values
                assert torch.any(mask > 0)

    def test_mvtec_invalid_class(self, mvtec_structure):
        """Test MVTec dataset with invalid class name."""
        with pytest.raises(AssertionError, match="should be in"):
            anodet.MVTecDataset(
                str(mvtec_structure), class_name="invalid_class", is_train=True
            )


class TestDatasetCompatibility:
    """Test dataset compatibility with DataLoader and training."""

    def test_dataloader_compatibility(self, sample_dataset):
        """Test dataset compatibility with PyTorch DataLoader."""
        dataloader = DataLoader(
            sample_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        # Should be able to iterate
        batch, images, classifications, masks = next(iter(dataloader))

        assert batch.shape[0] == 2  # batch size
        assert len(images) == 2
        assert len(classifications) == 2
        assert masks.shape[0] == 2

    def test_dataset_consistency(self, sample_dataset):
        """Test that dataset returns consistent results."""
        # Get same item twice
        result1 = sample_dataset[0]
        result2 = sample_dataset[0]

        batch1, _, _, _ = result1
        batch2, _, _, _ = result2

        # Should be identical
        assert torch.equal(batch1, batch2)

    def test_empty_directory_handling(self, temp_model_dir):
        """Test dataset behavior with empty directory."""
        empty_dir = temp_model_dir / "empty"
        empty_dir.mkdir()

        dataset = anodet.AnodetDataset(str(empty_dir))
        assert len(dataset) == 0

    def test_mixed_image_formats(self, temp_model_dir):
        """Test dataset with different image formats."""
        # Create images in different formats
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Save in different formats
        img.save(temp_model_dir / "test.png")
        img.save(temp_model_dir / "test.jpg")
        img.save(temp_model_dir / "test.jpeg")

        dataset = anodet.AnodetDataset(str(temp_model_dir))
        assert len(dataset) == 3  # Should load all formats


class TestDatasetTransforms:
    """Test dataset transform functionality."""

    def test_transform_pipeline(self, sample_images_dir):
        """Test that transforms are applied correctly."""
        dataset = anodet.AnodetDataset(
            str(sample_images_dir),
            resize=[256, 256],
            crop_size=[224, 224],
            normalize=True,
        )

        batch, raw_image, _, _ = dataset[0]

        # Raw image should be original size (224x224 from our test data)
        assert raw_image.shape[:2] == (224, 224)

        # Processed batch should be 224x224 after resize+crop
        assert batch.shape[-2:] == (224, 224)

    def test_no_crop_transform(self, sample_images_dir):
        """Test dataset without cropping."""
        dataset = anodet.AnodetDataset(
            str(sample_images_dir), resize=[256, 192], crop_size=None, normalize=True
        )

        batch, _, _, _ = dataset[0]

        # Should have resize dimensions
        assert batch.shape[-2:] == (256, 192)  # H, W
