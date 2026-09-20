"""Inference-time adapter for rolling AnomaVision drift monitoring."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from anomavision.production_monitor import ProductionDriftMonitor


def input_drift_features(batch: Any) -> np.ndarray:
    """Build a small deterministic input-space representation.

    This is a backend-safe fallback for deployments whose inference backend does
    not expose internal feature embeddings (for example Hailo/KV260/ONNX). It
    does not call or modify the anomaly model; it only summarizes the already
    prepared inference tensor with per-channel moments and quantiles.
    """
    values = np.asarray(batch.detach().cpu() if hasattr(batch, "detach") else batch)
    if values.ndim == 3:
        values = values[:, None, :, :]
    if values.ndim != 4:
        raise ValueError(
            "Input drift fallback expects an image batch with shape (N, C, H, W)."
        )
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Input drift values contain NaN or infinite values")

    flattened = values.reshape(values.shape[0], values.shape[1], -1)
    features = np.concatenate(
        [
            flattened.mean(axis=2),
            flattened.std(axis=2),
            np.quantile(flattened, 0.10, axis=2),
            np.quantile(flattened, 0.50, axis=2),
            np.quantile(flattened, 0.90, axis=2),
        ],
        axis=1,
    )
    return features


class InferenceDriftRuntime:
    """Attach a rolling drift monitor to a model object without changing scoring.

    A backend-provided representation is preferred. If the backend does not
    expose one, a deterministic input-space representation is used so drift
    monitoring remains available on hardware/portable inference backends.
    """

    def __init__(self, monitor: ProductionDriftMonitor, model: Any) -> None:
        self.monitor = monitor
        self.model = model

    def update(self, batch: Any) -> Optional[dict[str, Any]]:
        """Extract a drift representation and update the rolling window."""
        embeddings = self._extract(batch)
        report = self.monitor.update(embeddings)
        if report is None:
            return None
        return report.to_dict()

    def _extract(self, batch: Any) -> np.ndarray:
        custom = getattr(self.model, "extract_drift_embeddings", None)
        if custom is not None:
            try:
                return np.asarray(custom(batch))
            except NotImplementedError:
                # Portable/hardware backends may intentionally have no internal
                # embedding API. Fall back to model-independent input telemetry.
                pass

        extractor = getattr(self.model, "_extract", None)
        if extractor is None:
            return input_drift_features(batch)

        import torch

        with torch.inference_mode():
            features, _, _ = extractor(batch)
            # PatchCore returns (B, patches, dimensions). Pool over patches to
            # monitor the same representation used by the detector.
            if hasattr(features, "ndim") and features.ndim > 2:
                if hasattr(features, "float"):
                    features = features.float().mean(dim=1)
                else:
                    features = np.asarray(features, dtype=np.float64).mean(axis=1)

            if hasattr(features, "detach"):
                return features.detach().cpu().numpy()
            return np.asarray(features)
