"""Rolling production drift monitoring for AnomaVision inference.

This module connects inference-time embeddings to the existing DriftMonitor.
It deliberately keeps state bounded: only a rolling window of feature vectors
is retained, so long-running camera processes do not grow memory indefinitely.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np

from anomavision.drift import DriftMonitor, DriftReport


@dataclass(frozen=True)
class ProductionDriftStatus:
    """Current production monitoring state."""

    samples_seen: int
    window_size: int
    window_fill: int
    ready: bool
    status: str
    drift_score: Optional[float]
    psi: Optional[float]
    mean_shift: Optional[float]
    std_shift: Optional[float]
    cosine_shift: Optional[float]
    threshold: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProductionDriftMonitor:
    """Monitor embeddings emitted by a live AnomaVision inference loop.

    Args:
        reference: Reference embeddings from trusted normal production/training data.
        window_size: Maximum number of recent production samples retained.
        min_samples: Minimum current samples before drift is evaluated.
        threshold: PSI threshold passed to :class:`DriftMonitor`.
        evaluation_interval: Evaluate every N accepted samples once ready.
    """

    def __init__(
        self,
        reference: np.ndarray,
        *,
        window_size: int = 500,
        min_samples: int = 100,
        threshold: float = 0.20,
        evaluation_interval: int = 25,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not 2 <= min_samples <= window_size:
            raise ValueError("min_samples must be between 2 and window_size")
        if evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")

        self._monitor = DriftMonitor(reference, threshold=threshold)
        self.window_size = int(window_size)
        self.min_samples = int(min_samples)
        self.evaluation_interval = int(evaluation_interval)
        self._window: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._samples_seen = 0
        self._last_report: Optional[DriftReport] = None
        self._lock = Lock()

    @property
    def reference(self) -> np.ndarray:
        """Return the immutable reference representation used for monitoring."""
        return self._monitor.reference

    def update(self, embeddings: np.ndarray) -> Optional[DriftReport]:
        """Add one batch of inference embeddings and optionally evaluate drift.

        ``embeddings`` may be ``(N, D)`` or a higher-dimensional feature tensor.
        Higher-dimensional tensors are flattened per sample and mean-pooled over
        non-feature dimensions when necessary.
        """
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim < 2:
            raise ValueError("embeddings must have at least 2 dimensions")
        if values.ndim > 2:
            values = values.reshape(values.shape[0], -1)
        if values.shape[1] != self.reference.shape[1]:
            raise ValueError(
                "embedding dimension does not match reference "
                f"({values.shape[1]} != {self.reference.shape[1]})"
            )
        if not np.isfinite(values).all():
            raise ValueError("embeddings contain NaN or infinite values")

        with self._lock:
            for row in values:
                self._window.append(row.copy())
                self._samples_seen += 1

            if len(self._window) < self.min_samples:
                return None
            if self._samples_seen % self.evaluation_interval != 0:
                return self._last_report

            self._last_report = self._monitor.compare(np.asarray(self._window))
            return self._last_report

    def status(self) -> ProductionDriftStatus:
        """Return the latest bounded-memory monitoring status."""
        with self._lock:
            report = self._last_report
            ready = len(self._window) >= self.min_samples
            return ProductionDriftStatus(
                samples_seen=self._samples_seen,
                window_size=self.window_size,
                window_fill=len(self._window),
                ready=ready,
                status=report.status if report else "warming_up",
                drift_score=report.drift_score if report else None,
                psi=report.psi if report else None,
                mean_shift=report.mean_shift if report else None,
                std_shift=report.std_shift if report else None,
                cosine_shift=report.cosine_shift if report else None,
                threshold=self._monitor.threshold,
                warnings=report.warnings if report else ("warming_up",),
            )

    def reset(self) -> None:
        """Clear the production window while preserving the reference data."""
        with self._lock:
            self._window.clear()
            self._samples_seen = 0
            self._last_report = None

    def save_status(self, path: str | Path) -> None:
        """Persist the latest status as JSON for dashboards/health checks."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.status().to_dict(), indent=2) + "\n", encoding="utf-8"
        )
