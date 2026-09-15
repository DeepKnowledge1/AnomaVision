"""Helpers for obtaining model-representation embeddings during inference.

For the PyTorch PatchCore backend, the same normalized patch representation used
by anomaly scoring is exposed and pooled into one vector per image. This avoids
running a second backbone pass solely for drift monitoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def extract_model_embeddings(model: Any, batch: torch.Tensor) -> np.ndarray:
    """Extract one stable feature vector per image from a fitted PyTorch model.

    PatchCore uses its actual inference feature extractor. Other models may expose
    a compatible ``_extract`` method. A clear error is raised rather than silently
    monitoring raw pixels, because raw-pixel drift is often dominated by harmless
    lighting changes and is not the model's representation space.
    """
    if not isinstance(batch, torch.Tensor):
        batch = torch.as_tensor(batch, dtype=torch.float32)

    extractor = getattr(model, "_extract", None)
    if extractor is None:
        raise TypeError(
            "The loaded model does not expose a model-representation extractor. "
            "Drift monitoring currently requires a PyTorch model with _extract()."
        )

    with torch.inference_mode():
        embeddings, _, _ = extractor(batch)
        # Patch-level representations -> one vector per image. Mean pooling is
        # deterministic, bounded, and preserves the model's feature space.
        pooled = embeddings.float().mean(dim=1)
    return pooled.detach().cpu().numpy()
