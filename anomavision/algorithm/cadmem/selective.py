"""Experimental selective compensation with independent normal validation."""

import torch
from torch import nn


class SelectiveMemory(nn.Module):
    """Shrink nuisance budgets using held-out paired normal displacements.

    Reliability is directional energy relative to total displacement energy,
    multiplied by agreement with the fitted budget. This is a heuristic,
    not a guarantee of defect preservation.
    """

    def __init__(self, memory, residual_weight=0.25):
        super().__init__()
        if not 0 <= residual_weight <= 1:
            raise ValueError("residual_weight must be in [0, 1]")
        self.memory = memory
        self.residual_weight = residual_weight
        self.register_buffer("reliability", torch.zeros_like(memory.budgets))

    @torch.no_grad()
    def fit_reliability(self, normal, paired):
        if normal.ndim != 2 or paired.ndim != 3 or paired.shape[1:] != normal.shape:
            raise ValueError("Expected [N,D] and [T,N,D]")
        normal = normal.to(self.memory.prototypes)
        paired = paired.to(normal)
        if not len(normal) or not len(paired):
            raise ValueError("Empty validation features")
        if not torch.isfinite(normal).all() or not torch.isfinite(paired).all():
            raise ValueError("Nonfinite validation features")
        weights = []
        for center, basis, budget in zip(
            self.memory.prototypes, self.memory.bases, self.memory.budgets
        ):
            idx = (normal - center).norm(dim=1).topk(
                min(len(normal), self.memory.neighbors), largest=False
            ).indices
            delta = (paired[:, idx] - normal[idx]).flatten(0, 1)
            coeff = delta @ basis
            energy = coeff.square().mean(0)
            total = delta.square().sum(1).mean().clamp_min(1e-12)
            agreement = (coeff.abs() <= budget).float().mean(0)
            weights.append((energy / total * agreement).clamp(0, 1))
        self.reliability = torch.stack(weights)
        return self

    @torch.no_grad()
    def score(self, queries):
        memory = self.memory
        if memory.normalize_residual:
            raise ValueError("Fusion requires unnormalized distances")
        outputs = []
        for q in queries.to(memory.prototypes).split(memory.chunk_size):
            best = q.new_full((len(q),), float("inf"))
            raw = best.clone()
            for start in range(0, len(memory.prototypes), memory.chunk_size):
                sl = slice(start, start + memory.chunk_size)
                delta = q[:, None] - memory.prototypes[sl][None]
                basis = memory.bases[sl]
                bounds = memory.budgets[sl]
                coeff = torch.einsum("qkd,kdr->qkr", delta, basis)
                coeff = coeff.clamp(-bounds, bounds) * self.reliability[sl]
                residual = delta - torch.einsum("qkr,kdr->qkd", coeff, basis)
                best = torch.minimum(best, residual.norm(dim=-1).min(1).values)
                raw = torch.minimum(raw, delta.norm(dim=-1).min(1).values)
            outputs.append(
                self.residual_weight * raw + (1 - self.residual_weight) * best
            )
        return torch.cat(outputs) if outputs else queries.new_empty(0)
