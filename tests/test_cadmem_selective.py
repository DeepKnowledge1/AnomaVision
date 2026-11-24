import torch

from anomavision.algorithm.cadmem import BoundedNuisanceMemory
from anomavision.algorithm.cadmem.selective import SelectiveMemory


def test_selective_score_bounds():
    normal = torch.zeros(8, 2)
    paired = normal[None] + torch.tensor([[[0.5, 0.0]], [[-0.5, 0.0]]])
    memory = BoundedNuisanceMemory(rank=1, max_prototypes=1)
    memory.fit_features(normal, paired)
    model = SelectiveMemory(memory, residual_weight=0.25)
    model.fit_reliability(normal.clone(), paired.clone())
    queries = torch.tensor([[0.3, 0.0], [0.0, 0.3], [1.0, 0.0]])
    raw = queries.norm(dim=1)
    scores = model.score(queries)
    assert torch.all(scores >= memory.score(queries) - 1e-6)
    assert torch.all(scores <= raw + 1e-6)
    assert torch.isclose(scores[0], torch.tensor(0.075), atol=1e-6)
    assert torch.isclose(scores[1], raw[1])


def test_zero_reliability_preserves_raw_distance():
    normal = torch.zeros(4, 2)
    paired = normal[None] + torch.tensor([[[0.5, 0.0]]])
    memory = BoundedNuisanceMemory(rank=1, max_prototypes=1)
    memory.fit_features(normal, paired)
    model = SelectiveMemory(memory).fit_reliability(normal, normal[None])
    queries = torch.tensor([[0.3, 0.4]])
    assert torch.allclose(model.score(queries), queries.norm(dim=1))
