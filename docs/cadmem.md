# CAD-Mem (experimental PatchCore adapter)

This branch adds an **opt-in research adapter**. Existing PaDiM, PatchCore,
CLI, ONNX and deployment behavior are unchanged.

## Usage

```python
from torch.utils.data import DataLoader
from anomavision.algorithm.patchcore import PatchCore
from anomavision.algorithm.cadmem import CADMemPatchCore, CADMemConfig

base = PatchCore(backbone="resnet18", layer_indices=[0], patch_grid=14)
model = CADMemPatchCore(base, CADMemConfig(gate_cap=0.5))
model.fit(normal_train_loader, independent_normal_calibration_loader)
scores, maps = model.predict(batch)  # read-only
admitted = model.update(new_unlabeled_window)  # explicit adaptation
model.reset_adaptation()
```

**Limitations:** Calibration must be normal-only and disjoint from training.
The adapter currently copies embeddings to CPU NumPy, so its latency is not
comparable to the GPU PatchCore baseline. It does not support ONNX/TensorRT
export or the existing CLI. A score floor does not prove defect detection or
contamination resistance. Candidate support currently counts patches, not
independent images; clustered defect patches can satisfy it. Do not claim
production readiness or journal-level novelty without real-data evaluation.

Run: `pytest -q tests/test_cadmem.py tests/test_cadmem_patchcore.py`.
