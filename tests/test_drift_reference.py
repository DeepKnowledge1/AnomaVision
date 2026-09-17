import numpy as np
import torch
from PIL import Image

from anomavision.drift_reference import _collate_reference_batch


def _sample(height, width):
    batch = torch.zeros((3, 224, 224), dtype=torch.float32)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    classification = 1
    mask = torch.zeros((1, 224, 224), dtype=torch.float32)
    return batch, image, classification, mask


def test_reference_collate_keeps_native_image_sizes():
    samples = [_sample(375, 500), _sample(281, 300)]
    batch, images, classifications, masks = _collate_reference_batch(samples)

    assert batch.shape == (2, 3, 224, 224)
    assert [image.shape for image in images] == [(375, 500, 3), (281, 300, 3)]
    assert classifications.tolist() == [1, 1]
    assert masks.shape == (2, 1, 224, 224)


def test_reference_collate_does_not_modify_images():
    samples = [_sample(100, 120), _sample(150, 160)]
    original = [sample[1].copy() for sample in samples]

    _, images, _, _ = _collate_reference_batch(samples)

    for actual, expected in zip(images, original):
        np.testing.assert_array_equal(actual, expected)


def test_reference_embeddings_are_valid_for_production_monitor():
    from anomavision.production_monitor import ProductionDriftMonitor

    rng = np.random.default_rng(123)
    reference = rng.normal(size=(50, 16)).astype(np.float32)
    production = reference[:10].copy()

    monitor = ProductionDriftMonitor(
        reference, window_size=20, min_samples=10, evaluation_interval=10
    )
    report = monitor.update(production)

    assert report is not None
    assert report.feature_dimensions == reference.shape[1]
    assert report.reference_samples == reference.shape[0]
    assert report.current_samples == production.shape[0]
    assert np.isfinite(report.psi)
    assert np.isfinite(report.drift_score)
