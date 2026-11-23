"""Fast integration tests with a stub PatchCore (no pretrained weights)."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from anomavision.algorithm.cadmem import CADMemConfig, CADMemPatchCore


class StubPatchCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("memory_bank", torch.empty(0, 3))

    @property
    def is_fitted(self):
        return self.memory_bank.numel() > 0

    def fit(self, loader):
        self.memory_bank = torch.tensor([[1., 0., 0.], [0., 1., 0.]])

    def _extract(self, images):
        # Images Bx3x2x2 are already patch features.
        b = images.shape[0]
        patches = images.permute(0, 2, 3, 1).reshape(b, 4, 3)
        return torch.nn.functional.normalize(patches, dim=-1), 2, 2


def test_integration_fit_predict_update_reset():
    data = torch.ones(6, 3, 2, 2)
    data[:, 1:] = 0
    train = DataLoader(TensorDataset(data[:3]), batch_size=2)
    cal = DataLoader(TensorDataset(data[3:]), batch_size=2)
    model = CADMemPatchCore(StubPatchCore(), CADMemConfig(min_support=2, max_update_fraction=.25))
    model.fit(train, cal)
    scores, maps = model.predict(data[:2])
    assert scores.shape == (2,)
    assert maps.shape == (2, 2, 2)
    assert torch.isfinite(maps).all()
    before = model.memory.anchor.copy()
    model.update(data[:2])
    assert np.array_equal(before, model.memory.anchor)
    model.reset_adaptation()
    assert len(model.memory.adaptive) == 0


def test_requires_calibration_and_rejects_export():
    model = CADMemPatchCore(StubPatchCore())
    with pytest.raises(RuntimeError):
        model.predict(torch.ones(1, 3, 2, 2))
    with pytest.raises(RuntimeError):
        model.calibrate([])
    with pytest.raises(RuntimeError):
        model.predict(torch.ones(1, 3, 2, 2), export=True)
