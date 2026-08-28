import torch
from torch.utils.data import DataLoader, TensorDataset

from anomavision.algorithm.efficientad import EfficientAD


def test_efficientad_fit_predict_contract():
    images = torch.rand(2, 3, 224, 224)
    loader = DataLoader(TensorDataset(images), batch_size=1, shuffle=False)

    model = EfficientAD(
        device=torch.device("cpu"),
        pretrained_teacher=False,
        model_size="s",
        threshold_quantile=0.995,
    )
    model.fit(loader, epochs=1)

    scores, maps = model.predict(images[:1])
    assert scores.shape == (1,)
    assert maps.shape == (1, 224, 224)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(maps).all()
    assert torch.isfinite(model.score_mean)
    assert torch.isfinite(model.score_std)
    assert torch.isfinite(model.threshold)
    assert model.threshold.item() >= 0.0


def test_efficientad_rejects_unknown_model_size():
    try:
        EfficientAD(pretrained_teacher=False, model_size="large")
    except ValueError as exc:
        assert "model_size" in str(exc)
    else:
        raise AssertionError("Expected invalid EfficientAD model_size to fail")


def test_efficientad_rejects_invalid_threshold_quantile():
    try:
        EfficientAD(pretrained_teacher=False, threshold_quantile=1.0)
    except ValueError as exc:
        assert "threshold_quantile" in str(exc)
    else:
        raise AssertionError("Expected invalid threshold_quantile to fail")
