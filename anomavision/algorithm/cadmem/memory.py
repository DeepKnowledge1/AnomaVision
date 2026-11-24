"""Bounded counterfactual normality memory, experimental research API.

normal: [N,D] aligned reference patch features.
counterfactual: [T,N,D] feature embeddings after benign transformations.
Patch i must correspond to the same physical image location in both arrays.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class BoundedNuisanceMemory(nn.Module):
    """Union of locally bounded, counterfactual-derived normal feature sets.

    Each memory element is (prototype m, nuisance basis U, component budgets b).
    Its anomaly score is min_j ||z-m_j-U_j clip(U_j^T(z-m_j), -b_j, b_j)||.
    """

    def __init__(
        self,
        rank: int = 4,
        max_prototypes: int = 1024,
        neighbors: int = 32,
        budget_quantile: float = 0.95,
        candidate_neighbors: Optional[int] = None,
        chunk_size: int = 256,
        normalize_residual: bool = False,
        scale_floor: float = 0.01,
    ) -> None:
        super().__init__()
        if min(rank, max_prototypes, neighbors, chunk_size) < 1:
            raise ValueError("rank, prototypes, neighbors and chunk size must be positive")
        if candidate_neighbors is not None and candidate_neighbors < 1:
            raise ValueError("candidate_neighbors must be positive or None")
        if not 0 < budget_quantile <= 1 or scale_floor <= 0:
            raise ValueError("Invalid budget_quantile or scale_floor")
        self.rank = rank
        self.max_prototypes = max_prototypes
        self.neighbors = neighbors
        self.budget_quantile = budget_quantile
        self.candidate_neighbors = candidate_neighbors
        self.chunk_size = chunk_size
        self.normalize_residual = normalize_residual
        self.scale_floor = scale_floor
        self.register_buffer("prototypes", torch.empty(0, 0))
        self.register_buffer("bases", torch.empty(0, 0, 0))
        self.register_buffer("budgets", torch.empty(0, 0))
        self.register_buffer("scales", torch.empty(0))

    @property
    def is_fitted(self) -> bool:
        return self.prototypes.ndim == 2 and self.prototypes.shape[0] > 0

    @torch.no_grad()
    def fit_features(
        self,
        normal: Tensor,
        counterfactual: Tensor,
        prototypes: Optional[Tensor] = None,
    ) -> "BoundedNuisanceMemory":
        if normal.ndim != 2 or not torch.is_floating_point(normal):
            raise ValueError("normal must be floating point [N,D]")
        if (
            counterfactual.ndim != 3
            or counterfactual.shape[1:] != normal.shape
            or counterfactual.shape[0] < 1
        ):
            raise ValueError("counterfactual must be [T,N,D], with T >= 1")
        if not torch.isfinite(normal).all() or not torch.isfinite(counterfactual).all():
            raise ValueError("Training features must be finite")
        n, d = normal.shape
        if n == 0 or self.rank > d:
            raise ValueError("Empty normal features or rank exceeds feature dimension")
        normal = normal.float()
        counterfactual = counterfactual.to(normal)
        if prototypes is None:
            indices = torch.linspace(
                0, n - 1, steps=min(n, self.max_prototypes),
                device=normal.device,
            ).round().long()
            centers = normal[indices]
        else:
            if (
                prototypes.ndim != 2
                or prototypes.shape[1] != d
                or not 0 < len(prototypes) <= self.max_prototypes
                or not torch.isfinite(prototypes).all()
            ):
                raise ValueError("Invalid [K,D] prototypes")
            centers = prototypes.to(normal)
        deltas = counterfactual - normal.unsqueeze(0)
        bases, budgets, scales = [], [], []
        for center in centers:
            closest = torch.linalg.vector_norm(normal - center, dim=1).topk(
                min(n, self.neighbors), largest=False
            ).indices
            variations = deltas[:, closest].reshape(-1, d)
            # Uncentered SVD: zero is a meaningful no-change reference.
            _, _, vh = torch.linalg.svd(variations, full_matrices=False)
            r = min(self.rank, vh.shape[0])
            basis = variations.new_zeros(d, self.rank)
            basis[:, :r] = vh[:r].T
            coefficients = variations @ basis[:, :r]
            bounds = torch.quantile(
                coefficients.abs(), self.budget_quantile, dim=0
            )
            budget = variations.new_zeros(self.rank)
            budget[:r] = bounds
            explained = coefficients.clamp(-bounds, bounds) @ basis[:, :r].T
            residuals = torch.linalg.vector_norm(variations - explained, dim=1)
            scale = residuals.quantile(self.budget_quantile).clamp_min(
                self.scale_floor
            )
            bases.append(basis)
            budgets.append(budget)
            scales.append(scale)
        self.prototypes = centers.detach()
        self.bases = torch.stack(bases).detach()
        self.budgets = torch.stack(budgets).detach()
        self.scales = torch.stack(scales).detach()
        return self

    @torch.no_grad()
    def score(self, queries: Tensor) -> Tensor:
        """Distances to bounded normal sets. Exact search is the default.

        Optional candidate selection uses point proximity and is approximate.
        It can miss the best bounded-set score; report this as a separate ablation.
        """
        if not self.is_fitted:
            raise RuntimeError("Fit the memory before scoring")
        if queries.ndim != 2 or queries.shape[1] != self.prototypes.shape[1]:
            raise ValueError("Queries must be [Q,D] with matching dimension")
        if not torch.isfinite(queries).all():
            raise ValueError("Queries must be finite")
        chunks = []
        count = len(self.prototypes)
        for q in queries.to(self.prototypes).split(self.chunk_size):
            if self.candidate_neighbors is None or self.candidate_neighbors >= count:
                minimum = q.new_full((len(q),), float("inf"))
                for start in range(0, count, self.chunk_size):
                    centers = self.prototypes[start:start + self.chunk_size]
                    basis = self.bases[start:start + self.chunk_size]
                    bounds = self.budgets[start:start + self.chunk_size]
                    delta = q[:, None, :] - centers[None]
                    coeff = torch.einsum("qkd,kdr->qkr", delta, basis)
                    coeff = torch.maximum(
                        torch.minimum(coeff, bounds[None]), -bounds[None]
                    )
                    residual = delta - torch.einsum(
                        "qkr,kdr->qkd", coeff, basis
                    )
                    distances = torch.linalg.vector_norm(residual, dim=-1)
                    if self.normalize_residual:
                        distances = distances / self.scales[
                            start:start + self.chunk_size
                        ][None]
                    minimum = torch.minimum(minimum, distances.min(1).values)
                chunks.append(minimum)
            else:
                indices = torch.cdist(q, self.prototypes).topk(
                    self.candidate_neighbors, largest=False
                ).indices
                delta = q[:, None, :] - self.prototypes[indices]
                basis, bounds = self.bases[indices], self.budgets[indices]
                coeff = torch.einsum("qkd,qkdr->qkr", delta, basis)
                coeff = torch.maximum(torch.minimum(coeff, bounds), -bounds)
                residual = delta - torch.einsum(
                    "qkr,qkdr->qkd", coeff, basis
                )
                distances = torch.linalg.vector_norm(residual, dim=-1)
                if self.normalize_residual:
                    distances = distances / self.scales[indices]
                chunks.append(distances.min(1).values)
        return torch.cat(chunks) if chunks else queries.new_empty(0)


def conformal_p_values(
    calibration_scores: Tensor, test_scores: Tensor
) -> Tensor:
    """Normal-only upper-tail split conformal p-values.

    Finite-sample false-alarm validity requires exchangeable observations.
    """
    if calibration_scores.ndim != 1 or not len(calibration_scores):
        raise ValueError("Calibration scores must be nonempty 1D")
    if test_scores.ndim != 1:
        raise ValueError("Test scores must be 1D")
    if (
        not torch.isfinite(calibration_scores).all()
        or not torch.isfinite(test_scores).all()
    ):
        raise ValueError("Scores must be finite")
    return (
        1 + (
            calibration_scores[None, :] >= test_scores[:, None]
        ).sum(dim=1)
    ).float() / (len(calibration_scores) + 1)
