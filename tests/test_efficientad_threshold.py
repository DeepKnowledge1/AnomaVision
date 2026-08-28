from pathlib import Path

import torch

from anomavision.efficientad_threshold import load_calibrated_threshold


def test_load_calibrated_efficientad_threshold(tmp_path: Path):
    model_path = tmp_path / "model.onnx"
    stats_path = model_path.with_suffix(".pth")
    torch.save(
        {
            "algorithm": "efficientad",
            "model_state": {"threshold": torch.tensor(2.75)},
        },
        stats_path,
    )

    assert load_calibrated_threshold(model_path) == 2.75
