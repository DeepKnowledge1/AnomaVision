# tests/test_padim_lite.py
import torch


def test_build_padim_from_stats_and_predict_cpu(make_stats, patch_extractor):
    stats, (N, D, W, H) = make_stats()
    # patch extractor inside anodet.padim_lite to our dummy
    patch_extractor(N, D, W, H)

    from anodet.padim_lite import build_padim_from_stats

    # build runtime model on CPU
    model = build_padim_from_stats(stats, device="cpu")
    assert hasattr(model, "predict")
    assert str(getattr(model, "device", "cpu")) == "cpu"

    # run a dry inference (B=2, any HxW input; extractor ignores pixel content)
    x = torch.zeros(2, 3, 64, 64)
    scores, maps = model.predict(x)

    # shapes: scores -> (B,), maps -> (B, H, W)
    assert scores.shape == (2,)
    assert maps.shape == (2, x.shape[2], x.shape[3])
    assert scores.dtype == torch.float32
    assert maps.dtype == torch.float32
