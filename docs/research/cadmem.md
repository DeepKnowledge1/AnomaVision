# CADMem: Counterfactual Anisotropic Defect Memory

**Status: experimental research prototype; novelty and real-data improvements NOT established.**

## Purpose and safety of integration

CADMem studies whether locally bounded nuisance subspaces learned from paired
defect-free appearance transformations can distinguish benign changes from
genuine abnormalities. It is **not** a production replacement for PaDiM,
PatchCore, or EfficientAD.

All changes reside in new files under
`anomavision/algorithm/cadmem/`,
`scripts/research/cadmem/` and `tests/test_cadmem_research.py`.
Existing algorithms, benchmark scripts, configuration, inference backends,
and export pipelines have **not** been edited. CADMem is intentionally not
registered with the production CLI or ONNX/TensorRT export: those interfaces
remain unvalidated for this method.

CADMem reuses the existing PatchCore embedding extractor but does not modify
its normality model, distance or coreset selection.

## Mathematical formulation

For a normal image `x_i`, a defect-preserving transformation `T_t`, and
aligned patch extractor `f_p`, compute paired normal displacements

`Δ(i,p,t) = f_p(T_t(x_i)) - f_p(x_i)`.

For every representative prototype `m_j`, collect displacements from
nearby normal patch observations and estimate an **uncentered** truncated
SVD basis `U_j`. Estimate finite direction budgets `b_j` as quantiles
of the absolute projected displacement components.

The allowable feature set is

`C_j = {m_j + U_j a : |a_k| <= b_jk}`.

The default proposed anomaly score is the exact distance to a union of
finite-dimensional bounded sets:

`s(z) = min_j ||z - m_j - U_j clip(U_j^T(z-m_j), -b_j, b_j)||_2`.

Columns of `U_j` are orthonormal for retained nonzero-rank directions;
zero-filled excess columns have zero budgets. This closed-form clipped
projection follows from Euclidean orthogonal decomposition.

The default is **unscaled residual scoring**. The optional
`normalize_residual=True` is a separately testable ablation because
prototype-specific scale division changes the geometric distance
interpretation. Zero budgets reproduce point-memory nearest-neighbor
scoring. Finite budgets prevent *unlimited* invariance in learned nuisance
directions, but cannot guarantee detection when defect displacements overlap
normal variations.

There is no claim of causal invariance: chosen transformations are only
controlled **assumptions** about acceptable appearance changes.

## Research API

```python
from anomavision.algorithm.cadmem import CADMem

model = CADMem(
    backbone="resnet18", device="cuda", rank=4, max_prototypes=512
)
model.fit(
    normal_dataloader,
    transform=my_defect_preserving_transform,
    transformations_per_batch=8,
)
image_scores, score_maps = model.predict(preprocessed_images)
```

`my_defect_preserving_transform` must operate in the **input
preprocessing domain**, preserve tensor shape and preserve each patch's
physical correspondence. Changing geometry or inserting defects invalidates
the paired-normal assumption. Set and log all augmentation RNG seeds.
Training accumulates feature tensors in CPU memory; use a bounded pilot
dataset before scaling.

An independent feature-level API permits matched baselines without a CNN:

```python
from anomavision.algorithm.cadmem import BoundedNuisanceMemory

# normal: [N,D], transformed: [T,N,D], prototype_bank: [K,D].
memory = BoundedNuisanceMemory(rank=4, max_prototypes=512)
memory.fit_features(normal, transformed, prototypes=prototype_bank)
scores = memory.score(query_features)  # [Q], minimum explained residual
```

With `candidate_neighbors=None` the search is exhaustive. Enabling
`candidate_neighbors` retrieves candidates based on ordinary prototype
distance, which is **approximate** for bounded-set distance. Measure exact
search agreement in an ablation.

## Validation commands

```bash
pytest -q tests/test_cadmem_research.py
python scripts/research/cadmem/synthetic_pilot.py
python scripts/research/cadmem/benchmark_mvtec.py \
  --dataset-root /data/mvtec --category bottle --device cuda \
  --seed 42 --output results/cadmem_bottle.json
```

The benchmark uses the same frozen ResNet-18 feature extractor, coreset and
224-pixel preprocessing for point-memory and CADMem. It reserves independent
normal training images for a fixed upper-tail calibration threshold; it does
not optimize a threshold using test labels. If the sample count is too small
for the specified significance level, the threshold and its operating
metrics are reported as null, not as validated false-positive rates.

The script reports clean and synthetic brightness-shift image AUROC, image
AP, pixel AUROC, pixel AP, frozen-threshold false-positive rate and defect
recall. Its JSON includes seed, split sizes and arguments. This is a
**single-category exploratory pilot**, not the complete paper evaluation.

## Required scientific ablations before publication

1. Point-memory PatchCore with identical embeddings and prototypes.
2. Augmentation-only PatchCore with matched memory/storage budget.
3. Local normal PCA with the same ranks, prototypes and normal training data.
4. Unbounded paired subspace projection (tests defect suppression).
5. Bounded paired memory (this work).
6. PaDiM, EfficientAD, FR-PatchCore, INP-Former and further current baselines.
7. Feature basis rank, nuisance budget, transformation family, coreset size
   and exact versus approximate memory search.

Evaluate MVTec AD, VisA, MVTec AD 2 and real acquisition shifts with
category-matched conditions, mask-quality validation, at least five seeds,
held-out calibration, paired bootstrap confidence intervals and
worst-condition false-negative reporting. Use AUPRO@low FPR, image AP/AUROC,
pixel AP/AUROC, TPR at fixed FPR and full end-to-end P95 latency/memory.

The central falsification test: if simple augmentation or local PCA equals
CADMem on **real shifted imagery without defect recall loss**, the
intervention-derived memory is not sufficiently supported as a contribution.

## Explicit limitations

- The pilot was not run against external benchmark datasets as part of
  the implementation. No empirical performance claims can be made yet.
- The synthetic example demonstrates algebraic behavior only.
- The method may suppress actual defects aligned with normal shifts.
- Transformation realism and physical correspondence are unproven.
- The local SVD method has precedents in tangent-distance and manifold
  approaches; the exact novelty position needs a systematic literature check.
- Dataset protocols, distribution shifts, calibration sample size, and
  significance testing require additional independent experimental runs.
- End-to-end deployment compatibility for CADMem is not established.
- Existing model files and all other algorithms are intentionally untouched.
