"""Small deterministic contracts for behavior that must not silently change."""

import numpy as np
import torch

from anomavision.utils import classification
from anomavision.visualization.boundary import boundary_image
from anomavision.visualization.frame import frame_by_anomalies


def test_classification_boundary_is_inclusive():
    scores = np.array([0.0, 1.0, 1.000001])
    np.testing.assert_array_equal(classification(scores, 1.0), [0, 1, 1])


def test_classification_supports_torch_scores():
    scores = torch.tensor([0.0, 2.0, 3.0])
    result = classification(scores, 2.0)
    assert torch.equal(result, torch.tensor([0, 1, 1]))


def test_anomaly_boundary_is_visible():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    result = boundary_image(image, mask, boundary_color=(255, 0, 0))
    assert np.any(np.all(result == (255, 0, 0), axis=-1))


def test_anomaly_and_normal_frame_contract():
    images = np.full((2, 12, 12, 3), 128, dtype=np.uint8)
    result = frame_by_anomalies(images, np.array([1, 0]), padding=2)

    assert tuple(result[0, 0, 0]) == (255, 0, 0)
    assert tuple(result[1, 0, 0]) == (0, 255, 0)
