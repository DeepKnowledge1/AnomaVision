import numpy as np
import torch

from anomavision.utils import classification


def test_numpy_threshold_marks_high_scores_anomalous():
    scores = np.array([0.0, 2.0, 13.0])
    np.testing.assert_array_equal(classification(scores, 2.0), [0, 1, 1])
    np.testing.assert_array_equal(classification(scores, 13.0), [0, 0, 1])


def test_torch_threshold_marks_high_scores_anomalous():
    scores = torch.tensor([0.0, 2.0, 13.0])
    expected = torch.tensor([0, 1, 1], dtype=torch.int64)
    assert torch.equal(classification(scores, 2.0), expected)
