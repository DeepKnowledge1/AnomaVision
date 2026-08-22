import numpy as np
import torch

from anomavision.algorithm.patchcore import PatchCore
from anomavision.utils import resolve_threshold


def test_algorithm_threshold_overrides_legacy_threshold():
    config = {
        "algorithm": "patchcore",
        "thresh": 13.0,
        "thresh_patchcore": 0.35,
        "thresh_padim": 8.0,
    }
    assert resolve_threshold(config) == 0.35
    config["algorithm"] = "padim"
    assert resolve_threshold(config) == 8.0
    config["algorithm"] = "unknown"
    assert resolve_threshold(config) == 13.0


def test_zero_algorithm_threshold_is_preserved():
    assert (
        resolve_threshold(
            {"algorithm": "patchcore", "thresh": 13.0, "thresh_patchcore": 0.0}
        )
        == 0.0
    )


def test_kcenter_selection_is_deterministic_and_bounded():
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.7, 0.7]])
    first = PatchCore(
        device=torch.device("cpu"),
        memory_bank=bank,
        max_memory_patches=3,
        search_chunk_size=2,
        coreset_seed=7,
    )
    second = PatchCore(
        device=torch.device("cpu"),
        memory_bank=bank,
        max_memory_patches=3,
        search_chunk_size=2,
        coreset_seed=7,
    )
    selected_first = first._select_coreset(bank, 3)
    selected_second = second._select_coreset(bank, 3)
    assert selected_first.shape == (3, 2)
    assert torch.equal(selected_first, selected_second)
    assert np.unique(selected_first.numpy(), axis=0).shape[0] == 3
