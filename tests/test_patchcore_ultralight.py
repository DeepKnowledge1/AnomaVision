import torch
from torch.utils.data import DataLoader, TensorDataset

import anomavision.algorithm.patchcore as patchcore_module


class FakeExtractor(torch.nn.Module):
    backbone_name = "resnet18"

    def __init__(self, backbone_name, device):
        super().__init__()
        self.device = torch.device(device)

    def forward(self, batch, layer_indices=None):
        batch_size = batch.shape[0]
        # A 4x4 native feature map makes pooling and patch-count assertions explicit.
        embeddings = (
            batch.mean(dim=(1, 2, 3)).view(batch_size, 1, 1).expand(batch_size, 16, 3)
        )
        return embeddings, 4, 4

    def to_device(self, device):
        self.device = torch.device(device)


def test_ultralight_patchcore_bounds_patches_and_search(monkeypatch):
    monkeypatch.setattr(patchcore_module, "ResnetEmbeddingsExtractor", FakeExtractor)
    model = patchcore_module.PatchCore(
        device="cpu",
        layer_indices=[0],
        coreset_ratio=1.0,
        max_memory_patches=3,
        patch_grid=2,
        search_chunk_size=1,
    )
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 8, 8)), batch_size=2)
    model.fit(loader)

    assert model.memory_bank.shape[0] <= 3
    scores, maps = model.predict(torch.randn(2, 3, 8, 8))
    assert scores.shape == (2,)
    assert maps.shape == (2, 8, 8)


def test_patchcore_stats_round_trip_preserves_ultralight_settings(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(patchcore_module, "ResnetEmbeddingsExtractor", FakeExtractor)
    model = patchcore_module.PatchCore(
        device="cpu", patch_grid=3, search_chunk_size=7, max_memory_patches=11
    )
    model.memory_bank = torch.randn(5, 3)
    path = tmp_path / "patchcore.pth"
    model.save_statistics(str(path))
    restored = patchcore_module.build_patchcore_from_stats(
        torch.load(path, weights_only=False), device="cpu"
    )

    assert restored.patch_grid == 3
    assert restored.search_chunk_size == 7
    assert restored.max_memory_patches == 11


def test_patchcore_fit_and_inference_are_deterministic(monkeypatch):
    """Guard against regressions that make PatchCore appear random between runs."""
    monkeypatch.setattr(patchcore_module, "ResnetEmbeddingsExtractor", FakeExtractor)

    images = torch.arange(6 * 3 * 8 * 8, dtype=torch.float32).reshape(6, 3, 8, 8)
    loader = DataLoader(TensorDataset(images), batch_size=2, shuffle=False)
    query = images[:2]

    def build_and_run():
        model = patchcore_module.PatchCore(
            device="cpu",
            layer_indices=[0],
            coreset_ratio=0.5,
            max_memory_patches=5,
            patch_grid=2,
            search_chunk_size=2,
            coreset_method="kcenter",
            coreset_seed=42,
        )
        model.fit(loader)
        scores, maps = model.predict(query)
        return model.memory_bank.clone(), scores.clone(), maps.clone()

    bank_a, scores_a, maps_a = build_and_run()
    bank_b, scores_b, maps_b = build_and_run()

    assert torch.equal(bank_a, bank_b)
    assert torch.equal(scores_a, scores_b)
    assert torch.equal(maps_a, maps_b)
    assert torch.isfinite(scores_a).all()
    assert torch.isfinite(maps_a).all()
