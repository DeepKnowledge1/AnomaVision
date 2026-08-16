import numpy as np

from anomavision.utils import make_localization_mask, resolve_threshold


def test_patchcore_does_not_reuse_padim_scale_threshold():
    config = {"algorithm": "patchcore", "thresh": 13.0}
    assert resolve_threshold(config) == 0.35


def test_patchcore_localization_mask_keeps_only_high_regions_for_anomaly():
    score_maps = np.zeros((2, 8, 8), dtype=np.float32)
    score_maps[0, 3:5, 3:5] = 1.0
    score_maps[1, 3:5, 3:5] = 1.0
    masks = make_localization_mask(score_maps, np.array([1, 0]))
    assert masks[0, 3:5, 3:5].all()
    assert masks[0].sum() > 0
    assert masks[1].sum() == 0
