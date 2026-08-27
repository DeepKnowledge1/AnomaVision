"""Helpers for loading model-calibrated EfficientAD thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_calibrated_threshold(model_path: str | Path) -> float:
    """Load the threshold persisted by EfficientAD training.

    Training writes a compact ``.pth`` artifact next to every model. The
    artifact contains the EfficientAD state dict, including its calibrated
    threshold buffer. This keeps the threshold tied to the exact trained model
    and also makes it available when the inference model is ONNX/TensorRT/etc.
    """
    model_path = Path(model_path)
    candidates = [model_path.with_suffix(".pth")]
    if model_path.suffix == ".pth":
        candidates.insert(0, model_path)

    for candidate in candidates:
        if not candidate.is_file():
            continue
        data: Any = torch.load(candidate, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or data.get("algorithm") != "efficientad":
            continue
        state = data.get("model_state", {})
        threshold = state.get("threshold") if isinstance(state, dict) else None
        if threshold is None:
            raise ValueError(f"EfficientAD artifact has no calibrated threshold: {candidate}")
        value = float(torch.as_tensor(threshold).reshape(-1)[0].item())
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"EfficientAD calibrated threshold is not finite: {candidate}")
        return value

    raise FileNotFoundError(
        f"No calibrated EfficientAD threshold found next to model '{model_path}'. "
        "Retrain the EfficientAD model to create the .pth calibration artifact."
    )
