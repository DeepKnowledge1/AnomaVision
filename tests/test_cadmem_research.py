"""Pure mathematical regression tests for the CADMem research module."""

import pytest
import torch

from anomavision.algorithm.cadmem.ablations import (
    fit_local_normal_pca,
    without_budgets,
)
from anomavision.algorithm.cadmem.cadmem import CADMem
from anomavision.algorithm.cadmem.memory import (
    BoundedNuisanceMemory,
    conformal_p_values,
)


def example():
    normal = torch.tensor([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]])
    cf = torch.stack(
        (
            normal + torch.tensor([0.5, 0.0]),
            normal + torch.tensor([-0.5, 0.0]),
        )
    )
    return normal, cf


def test_benign_shift_is_tolerated_but_orthogonal_defect_is_not():
    normal, cf = example()
    memory = BoundedNuisanceMemory(rank=1, max_prototypes=1, neighbors=3).fit_features(
        normal, cf, prototypes=normal[:1]
    )
    assert memory.score(torch.tensor([[0.4, 0.0]])).item() < 1e-5
    assert torch.allclose(
        memory.score(torch.tensor([[0.0, 0.2]])),
        torch.tensor([0.2]),
        atol=1e-5,
    )
    assert memory.score(torch.tensor([[1.0, 0.0]])).item() > 0.49


def test_zero_nuisance_reduces_to_point_nearest_neighbor():
    normal = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    memory = BoundedNuisanceMemory(rank=1, max_prototypes=2, neighbors=2).fit_features(
        normal, normal.unsqueeze(0)
    )
    queries = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected = torch.cdist(queries, normal).min(dim=1).values
    assert torch.allclose(memory.score(queries), expected, atol=1e-5)


def test_exact_search_is_no_worse_than_candidate_approximation():
    normal, cf = example()
    exact = BoundedNuisanceMemory(rank=1, max_prototypes=3, neighbors=3).fit_features(
        normal, cf
    )
    approximate = BoundedNuisanceMemory(
        rank=1, max_prototypes=3, neighbors=3, candidate_neighbors=1
    ).fit_features(normal, cf)
    queries = torch.tensor([[0.2, 0.1], [0.9, 0.3]])
    assert torch.all(exact.score(queries) <= approximate.score(queries) + 1e-6)


def test_save_and_load_memory_buffers():
    normal, cf = example()
    fitted = BoundedNuisanceMemory(rank=1, max_prototypes=2).fit_features(normal, cf)
    state = fitted.state_dict()
    restored = BoundedNuisanceMemory(rank=1, max_prototypes=2)
    for key in ("prototypes", "bases", "budgets", "scales"):
        setattr(restored, key, torch.empty_like(state[key]))
    restored.load_state_dict(state)
    assert torch.allclose(fitted.score(normal), restored.score(normal))


def test_invalid_input_rejected():
    normal, cf = example()
    with pytest.raises(ValueError):
        BoundedNuisanceMemory(rank=3).fit_features(normal, cf)
    with pytest.raises(ValueError):
        BoundedNuisanceMemory().fit_features(normal, cf[:, :1])
    with pytest.raises(RuntimeError):
        BoundedNuisanceMemory().score(normal)


def test_conformal_p_values():
    calibration = torch.tensor([0.1, 0.2, 0.3, 0.4])
    values = conformal_p_values(calibration, torch.tensor([0.0, 0.3, 0.9]))
    assert torch.allclose(values, torch.tensor([1.0, 0.6, 0.2]))


def test_image_wrapper_contract_without_network_download():
    class StubExtractor(torch.nn.Module):
        def _extract(self, x):
            return x.flatten(2).transpose(1, 2).float(), x.shape[2], x.shape[3]

    # Exercise the wrapper contract without instantiating a pretrained ResNet.
    model = CADMem.__new__(CADMem)
    torch.nn.Module.__init__(model)
    model.feature_model = StubExtractor()
    model.memory = BoundedNuisanceMemory(rank=1, max_prototypes=1)
    normal = torch.tensor([[0.0, 0.0, 0.0]])
    model.memory.fit_features(
        normal,
        normal.unsqueeze(0) + torch.tensor([0.2, 0.0, 0.0]),
    )
    scores, maps = model.predict(torch.zeros(2, 3, 2, 2))
    assert scores.shape == (2,)
    assert maps.shape == (2, 2, 2)
    assert torch.isfinite(scores).all()


def test_paired_directions_are_distinct_from_local_normal_pca():
    normal = torch.tensor([[0.0, -0.1], [0.0, 0.0], [0.0, 0.1]])
    cf = normal.unsqueeze(0) + torch.tensor([0.5, 0.0])
    center = normal[1:2]
    paired = BoundedNuisanceMemory(
        rank=1, max_prototypes=1, neighbors=3
    ).fit_features(normal, cf, prototypes=center)
    local = fit_local_normal_pca(normal, center, rank=1, neighbors=3)
    query_x = torch.tensor([[0.3, 0.0]])
    assert paired.score(query_x).item() < 1e-5
    assert local.score(query_x).item() > 0.29


def test_unbounded_projection_can_hide_large_aligned_deviations():
    normal, cf = example()
    bounded = BoundedNuisanceMemory(
        rank=1, max_prototypes=1, neighbors=3
    ).fit_features(normal, cf, prototypes=normal[:1])
    unbounded = without_budgets(bounded)
    distant_query = torch.tensor([[5.0, 0.0]])
    assert unbounded.score(distant_query).item() < 1e-5
    assert bounded.score(distant_query).item() > 4.0
    assert bounded.budgets.max().item() < 1.0
