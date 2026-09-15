"""Inference-time adapter for rolling AnomaVision drift monitoring."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from anomavision.production_monitor import ProductionDriftMonitor


class InferenceDriftRuntime:
    """Attach a rolling drift monitor to a model object without changing scoring.

    The adapter is intentionally backend-agnostic at the API level. A model may
    provide ``extract_drift_embeddings(batch)``; otherwise PatchCore-like models
    exposing ``_extract(batch)`` are supported.
    """

    def __init__(self, monitor: ProductionDriftMonitor, model: Any) -> None:
        self.monitor = monitor
        self.model = model

    def update(self, batch: Any) -> Optional[dict[str, Any]]:
        """Extract model features from an inference batch and update the window."""
        embeddings = self._extract(batch)
        report = self.monitor.update(embeddings)
        if report is None:
            return None
        return report.to_dict()

    def _extract(self, batch: Any) -> np.ndarray:
        custom = getattr(self.model, "extract_drift_embeddings", None)
        if custom is not None:
            return np.asarray(custom(batch))

        extractor = getattr(self.model, "_extract", None)
        if extractor is None:
            raise TypeError(
                "Drift integration requires the model to expose "
                "extract_drift_embeddings(batch) or _extract(batch)."
            )

        import torch

        with torch.inference_mode():
            features, _, _ = extractor(batch)
            # PatchCore returns (B, patches, dimensions). Pool over patches to
            # monitor the same representation used by the detector.
            if features.ndim > 2:
                features = features.float().mean(dim=1)
            return features.detach().cpu().numpy()
