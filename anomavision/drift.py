"""Production data-drift monitoring for anomaly-detection embeddings.

The drift monitor compares a trusted reference distribution (for example, the
normal training/validation population) with a production window. It is model-
agnostic and does not require labels, making it useful for unsupervised visual
inspection systems.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class DriftReport:
    """Summary of distribution shift between reference and current data."""

    reference_samples: int
    current_samples: int
    feature_dimensions: int
    psi: float
    mean_shift: float
    std_shift: float
    cosine_shift: float
    drift_score: float
    status: str
    threshold: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DriftMonitor:
    """Detect feature-distribution drift without requiring anomaly labels.

    Inputs are sample-by-feature matrices. For image anomaly detection, these
    can be pooled backbone embeddings, model feature vectors, or any stable
    representation produced during inference.
    """

    def __init__(
        self,
        reference: np.ndarray,
        *,
        bins: int = 20,
        threshold: float = 0.20,
        epsilon: float = 1e-6,
    ) -> None:
        self.reference = self._validate(reference, "reference")
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if threshold <= 0:
            raise ValueError("threshold must be greater than 0")
        self.bins = bins
        self.threshold = threshold
        self.epsilon = epsilon

    @staticmethod
    def _validate(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array of shape (samples, features)")
        if array.shape[0] < 2:
            raise ValueError(f"{name} must contain at least 2 samples")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return array

    def compare(self, current: np.ndarray) -> DriftReport:
        """Compare a production window against the reference distribution."""
        current_array = self._validate(current, "current")
        if current_array.shape[1] != self.reference.shape[1]:
            raise ValueError(
                "reference and current must have the same feature dimension "
                f"({self.reference.shape[1]} != {current_array.shape[1]})"
            )

        psi = self._population_stability_index(current_array)
        mean_shift = self._relative_location_shift(current_array)
        std_shift = self._relative_scale_shift(current_array)
        cosine_shift = self._cosine_shift(current_array)

        # PSI is the primary distribution metric. The other signals make the
        # score sensitive to feature-space translation/scale changes as well.
        drift_score = float(
            min(
                1.0,
                0.60 * min(psi, 1.0)
                + 0.25 * min(mean_shift, 1.0)
                + 0.15 * min(std_shift, 1.0),
            )
        )
        status = "drift" if psi >= self.threshold else "stable"

        warnings = []
        if current_array.shape[0] < 100:
            warnings.append("small_current_window")
        if psi >= self.threshold:
            warnings.append("feature_distribution_shift")
        if cosine_shift >= 0.10:
            warnings.append("embedding_direction_shift")

        return DriftReport(
            reference_samples=self.reference.shape[0],
            current_samples=current_array.shape[0],
            feature_dimensions=self.reference.shape[1],
            psi=float(psi),
            mean_shift=float(mean_shift),
            std_shift=float(std_shift),
            cosine_shift=float(cosine_shift),
            drift_score=drift_score,
            status=status,
            threshold=self.threshold,
            warnings=tuple(warnings),
        )

    def _population_stability_index(self, current: np.ndarray) -> float:
        """Compute PSI using reference-derived quantile bins.

        Fixed min/max bins can report very large PSI for two small samples drawn
        from the same distribution because sparse bins receive almost all of
        their mass from sampling noise. Reference quantiles keep the expected
        distribution stable, while the bin count adapts to small production
        windows. This preserves the PSI signal for genuine distribution shifts.
        """
        total = 0.0
        used = 0
        effective_bins = min(self.bins, max(2, int(np.sqrt(current.shape[0]))))
        smoothing = max(self.epsilon, 1e-4)

        for feature in range(self.reference.shape[1]):
            ref = self.reference[:, feature]
            cur = current[:, feature]

            quantiles = np.linspace(0.0, 1.0, effective_bins + 1)
            edges = np.quantile(ref, quantiles)
            edges = np.unique(edges)
            if edges.size < 2:
                continue

            # Keep every production value in a bin, including values outside
            # the reference range, without changing the reference distribution.
            edges = edges.copy()
            edges[0] = -np.inf
            edges[-1] = np.inf

            ref_hist, _ = np.histogram(ref, bins=edges)
            cur_hist, _ = np.histogram(cur, bins=edges)
            ref_pct = (ref_hist + smoothing) / (
                ref_hist.sum() + smoothing * len(ref_hist)
            )
            cur_pct = (cur_hist + smoothing) / (
                cur_hist.sum() + smoothing * len(cur_hist)
            )
            total += float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
            used += 1

        return total / used if used else 0.0

    def _relative_location_shift(self, current: np.ndarray) -> float:
        ref_mean = self.reference.mean(axis=0)
        cur_mean = current.mean(axis=0)
        ref_scale = np.maximum(np.abs(ref_mean), self.epsilon)
        return float(np.mean(np.abs(cur_mean - ref_mean) / ref_scale))

    def _relative_scale_shift(self, current: np.ndarray) -> float:
        ref_std = self.reference.std(axis=0)
        cur_std = current.std(axis=0)
        scale = np.maximum(ref_std, self.epsilon)
        return float(np.mean(np.abs(cur_std - ref_std) / scale))

    def _cosine_shift(self, current: np.ndarray) -> float:
        ref_mean = self.reference.mean(axis=0)
        cur_mean = current.mean(axis=0)
        ref_norm = np.linalg.norm(ref_mean)
        cur_norm = np.linalg.norm(cur_mean)
        if ref_norm <= self.epsilon or cur_norm <= self.epsilon:
            return 0.0
        cosine_similarity = float(np.dot(ref_mean, cur_mean) / (ref_norm * cur_norm))
        return float(1.0 - np.clip(cosine_similarity, -1.0, 1.0))

    def save_reference(self, path: str | Path) -> None:
        """Persist the reference embedding matrix as a NumPy file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, self.reference)

    @classmethod
    def from_reference_file(cls, path: str | Path, **kwargs: Any) -> "DriftMonitor":
        """Create a monitor from a ``.npy`` reference embedding file."""
        reference = np.load(Path(path), allow_pickle=False)
        return cls(reference, **kwargs)


def save_report(report: DriftReport, path: str | Path) -> None:
    """Write a machine-readable JSON drift report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load a supported embedding matrix from ``.npy`` or ``.npz``."""
    source = Path(path)
    if source.suffix == ".npy":
        return np.load(source, allow_pickle=False)
    if source.suffix == ".npz":
        archive = np.load(source, allow_pickle=False)
        if not archive.files:
            raise ValueError(f"No arrays found in {source}")
        return archive[archive.files[0]]
    raise ValueError("Embedding files must use .npy or .npz")
