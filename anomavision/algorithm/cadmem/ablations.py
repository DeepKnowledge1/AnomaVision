"""Matched normal-PCA and unbounded-projection controls for CADMem.

These baselines are intentionally separate from the proposed method. All
methods must share the same pretrained features and prototype locations.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor

from .memory import BoundedNuisanceMemory


@torch.no_grad()
def fit_local_normal_pca(
    normal: Tensor,
    prototypes: Tensor,
    rank: int = 4,
    neighbors: int = 32,
    budget_quantile: float = 0.95,
) -> BoundedNuisanceMemory:
    """Use local normal offsets (NOT paired interventions) to learn subspaces."""
    if normal.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("Expected [N,D] normal features and [K,D] prototypes")
    if normal.shape[1] != prototypes.shape[1]:
        raise ValueError("Normal and prototype dimensions must match")
    if not len(normal) or not len(prototypes):
        raise ValueError("Normal features and prototypes cannot be empty")
    if not 1 <= rank <= normal.shape[1] or neighbors < 1:
        raise ValueError("Invalid rank or neighbor count")
    model = BoundedNuisanceMemory(
        rank=rank,
        max_prototypes=len(prototypes),
        neighbors=neighbors,
        budget_quantile=budget_quantile,
    )
    normal = normal.float()
    centers = prototypes.to(normal).float()
    bases, budgets, scales = [], [], []
    for center in centers:
        indices = torch.cdist(
            center[None, :], normal
        ).flatten().topk(min(len(normal), neighbors), largest=False).indices
        deviations = normal[indices] - center
        _, _, vh = torch.linalg.svd(deviations, full_matrices=False)
        effective_rank = min(rank, vh.shape[0])
        basis = deviations.new_zeros(normal.shape[1], rank)
        basis[:, :effective_rank] = vh[:effective_rank].T
        coeff = deviations @ basis[:, :effective_rank]
        bound = torch.quantile(
            coeff.abs(), budget_quantile, dim=0
        )
        budget = deviations.new_zeros(rank)
        budget[:effective_rank] = bound
        compensated = coeff.clamp(-bound, bound) @ basis[:, :effective_rank].T
        residual = torch.linalg.vector_norm(deviations - compensated, dim=1)
        scale = residual.quantile(budget_quantile).clamp_min(0.01)
        bases.append(basis)
        budgets.append(budget)
        scales.append(scale)
    model.prototypes = centers.detach()
    model.bases = torch.stack(bases).detach()
    model.budgets = torch.stack(budgets).detach()
    model.scales = torch.stack(scales).detach()
    return model


def without_budgets(
    fitted: BoundedNuisanceMemory,
) -> BoundedNuisanceMemory:
    """Unbounded nuisance projection; demonstrates possible defect suppression.

    This ablation deliberately relaxes all learned direction budgets.
    """
    if not fitted.is_fitted:
        raise RuntimeError("Model must be fitted")
    model = deepcopy(fitted)
    model.budgets.fill_(torch.finfo(model.budgets.dtype).max)
    return model
