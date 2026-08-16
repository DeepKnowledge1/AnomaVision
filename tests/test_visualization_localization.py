import numpy as np
import torch
from PIL import Image

from anomavision.visualization.frame import frame_by_anomalies
from anomavision.visualization.heatmap import heatmap_image
from anomavision.visualization.highlight import highlighted_image


def test_anomaly_frame_is_red_and_normal_frame_is_green():
    images = np.full((2, 12, 12, 3), 128, dtype=np.uint8)
    framed = frame_by_anomalies(images, np.array([1, 0]), padding=2)
    assert tuple(framed[0, 0, 0]) == (255, 0, 0)
    assert tuple(framed[1, 0, 0]) == (0, 255, 0)


def test_highlight_uses_anomaly_mask_directly():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    result = highlighted_image(image, mask, color=(255, 0, 0), alpha=1.0)
    assert result[3, 3, 0] > 200
    assert result[0, 0].sum() == 0


def test_heatmap_high_scores_are_rendered_in_anomaly_direction():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    scores = np.zeros((8, 8), dtype=np.float32)
    scores[2:6, 2:6] = 1.0
    result = heatmap_image(image, scores, alpha=1.0)
    assert not np.array_equal(result[3, 3], result[0, 0])
