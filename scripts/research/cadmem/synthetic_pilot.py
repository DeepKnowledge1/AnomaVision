"""Synthetic CADMem mechanism check (NOT an industrial detection benchmark).

python scripts/research/cadmem/synthetic_pilot.py
"""

import json

import torch

from anomavision.algorithm.cadmem import BoundedNuisanceMemory


def main():
    torch.manual_seed(42)
    normal = torch.zeros(100, 8)
    normal[:, 0] = 0.025 * torch.randn(100)
    counterfactuals = torch.stack(
        [
            normal
            + torch.randn(100, 8)
            * torch.tensor([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for _ in range(8)
        ]
    )
    center = normal[:1]
    memory = BoundedNuisanceMemory(
        rank=1, max_prototypes=1, neighbors=100
    ).fit_features(normal, counterfactuals, prototypes=center)
    nuisance_only = torch.zeros(100, 8)
    nuisance_only[:, 0] = 0.12
    defects = nuisance_only.clone()
    defects[:, 1] = 0.3
    result = {
        "status": "synthetic feature-space sanity check only",
        "seed": 42,
        "point_benign_mean": float(torch.cdist(nuisance_only, center).mean()),
        "point_defect_mean": float(torch.cdist(defects, center).mean()),
        "cadmem_benign_mean": float(memory.score(nuisance_only).mean()),
        "cadmem_defect_mean": float(memory.score(defects).mean()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
