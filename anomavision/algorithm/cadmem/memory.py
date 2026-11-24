"""Bounded, conservative patch-memory adaptation (experimental).

All inputs are L2-normalized patch embeddings supplied by PatchCore.
The anchor is immutable; candidate admission is heuristic, not certified.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CADMemConfig:
    max_adaptive: int = 256
    admission_quantile: float = 0.50
    agreement_quantile: float = 0.90
    min_support: int = 3
    max_update_fraction: float = 0.05
    gate_cap: float = 0.5

    def __post_init__(self):
        if self.max_adaptive < 1 or self.min_support < 2:
            raise ValueError("Invalid memory or support limit")
        if not (0 < self.admission_quantile < 1 and 0 < self.agreement_quantile < 1):
            raise ValueError("Quantiles must be in (0, 1)")
        if not (0 <= self.max_update_fraction <= 1 and 0 <= self.gate_cap <= 1):
            raise ValueError("Fractions must be in [0, 1]")


class CADMem:
    def __init__(self, anchor, calibration, config=None):
        self.config = config or CADMemConfig()
        self.anchor = self._matrix(anchor).copy()
        calibration = self._matrix(calibration)
        if self.anchor.shape[1] != calibration.shape[1]:
            raise ValueError("Feature dimensions differ")
        self.anchor.setflags(write=False)
        self.calibration_scores = self._nearest(calibration, self.anchor)
        self.admission_limit = float(np.quantile(
            self.calibration_scores, self.config.admission_quantile))
        self.agreement_limit = float(np.quantile(
            self.calibration_scores, self.config.agreement_quantile))
        self.adaptive = np.empty((0, self.anchor.shape[1]), dtype=np.float32)
        self.last_admitted_indices = np.empty(0, dtype=np.int64)

    @staticmethod
    def _matrix(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected finite, nonempty [patches, features] array")
        return x

    @staticmethod
    def _nearest(x, bank):
        if not len(bank):
            return np.full(len(x), np.inf, dtype=np.float32)
        result = []
        for start in range(0, len(x), 256):
            part = x[start:start + 256]
            squared = ((part * part).sum(1)[:, None] +
                       (bank * bank).sum(1)[None, :] - 2 * part @ bank.T)
            result.append(np.sqrt(np.maximum(squared.min(axis=1), 0)))
        return np.concatenate(result)

    def score(self, patches):
        x = self._matrix(patches)
        if x.shape[1] != self.anchor.shape[1]:
            raise ValueError("Feature dimensions differ")
        d0 = self._nearest(x, self.anchor)
        if not len(self.adaptive):
            return d0
        da = self._nearest(x, self.adaptive)
        return d0 - self.config.gate_cap * np.maximum(d0 - da, 0)

    def update(self, patches):
        self.last_admitted_indices = np.empty(0, dtype=np.int64)
        x = self._matrix(patches)
        if x.shape[1] != self.anchor.shape[1]:
            raise ValueError("Feature dimensions differ")
        d0 = self._nearest(x, self.anchor)
        indices = np.flatnonzero(d0 <= self.admission_limit)
        eligible = x[indices]
        if len(eligible) < self.config.min_support:
            return 0
        supported = np.zeros(len(eligible), dtype=bool)
        for start in range(0, len(eligible), 32):
            stop = min(start + 32, len(eligible))
            pair = np.linalg.norm(eligible[start:stop, None, :] - eligible[None, :, :], axis=-1)
            pair[np.arange(stop - start), np.arange(start, stop)] = np.inf
            supported[start:stop] = (pair <= self.agreement_limit).sum(axis=1) >= self.config.min_support - 1
        indices = indices[supported]
        eligible = eligible[supported]
        budget = min(self.config.max_adaptive - len(self.adaptive),
                     int(np.floor(self.config.max_update_fraction * len(x))))
        if budget <= 0 or not len(eligible):
            return 0
        center = np.median(eligible, axis=0)
        order = np.argsort(np.linalg.norm(eligible - center, axis=1), kind="stable")
        self.last_admitted_indices = indices[order[:budget]]
        selected = eligible[order[:budget]]
        self.adaptive = np.concatenate((self.adaptive, selected), axis=0)
        return len(selected)

    def reset_adaptation(self):
        self.adaptive = np.empty((0, self.anchor.shape[1]), dtype=np.float32)
