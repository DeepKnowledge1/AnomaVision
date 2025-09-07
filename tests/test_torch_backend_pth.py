# tests/test_torch_backend_pth.py
import numpy as np
import torch


def test_torch_backend_loads_stats_pth_and_infers(
    tmp_path, make_stats, patch_extractor
):
    stats, (N, D, W, H) = make_stats()
    patch_extractor(N, D, W, H)

    # save a stats-only artifact as your saver would
    pth_path = tmp_path / "model_stats.pth"
    torch.save(stats, pth_path)

    # now load through your TorchBackend (expects .predict returning (scores, maps))
    from anodet.inference.model.backends.torch_backend import TorchBackend

    be = TorchBackend(str(pth_path), device="cpu", use_amp=False)
    assert hasattr(be, "model")
    assert be.use_amp is False
    assert getattr(be, "device").type == "cpu"

    # run inference; backend returns numpy arrays
    batch = torch.zeros(3, 3, 80, 80)  # B=3
    scores, maps = be.predict(batch)

    assert isinstance(scores, np.ndarray) and isinstance(maps, np.ndarray)
    assert scores.shape == (batch.shape[0],)
    assert maps.shape == (3, batch.shape[2], batch.shape[3])
