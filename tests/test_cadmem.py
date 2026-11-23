import numpy as np
import pytest
from anomavision.algorithm.cadmem import CADMem, CADMemConfig


def make_model():
    rng = np.random.default_rng(42)
    anchor = rng.normal(size=(32, 4)).astype("float32")
    cal = anchor + rng.normal(scale=.05, size=anchor.shape).astype("float32")
    return CADMem(anchor, cal, CADMemConfig(max_update_fraction=.25, min_support=2))


def test_anchor_immutable_and_score_floor():
    model = make_model()
    anchor = model.anchor.copy()
    x = np.repeat(anchor[:1], 12, axis=0)
    model.update(x)
    assert np.array_equal(anchor, model.anchor)
    d0 = model._nearest(x, model.anchor)
    assert np.all(model.score(x) >= (1 - model.config.gate_cap) * d0 - 1e-6)


def test_far_outliers_not_admitted():
    model = make_model()
    x = np.ones((20, 4), dtype="float32") * 1000
    assert model.update(x) == 0
    assert len(model.adaptive) == 0


def test_reset_and_budget():
    model = make_model()
    x = np.repeat(model.anchor[:1], 20, axis=0)
    assert model.update(x) <= 5
    model.reset_adaptation()
    assert len(model.adaptive) == 0


def test_invalid_input():
    model = make_model()
    with pytest.raises(ValueError):
        model.score(np.array([[np.nan] * 4]))
