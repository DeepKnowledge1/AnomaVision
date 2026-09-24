# AnomaVision Studio

AnomaVision Studio is the local-first UI and orchestration layer for AnomaVision.

## Product workflow

**Data → Train → Validate → Deploy → Inspect → Monitor**

Studio does not reimplement anomaly-detection algorithms or deployment backends. PaDiM, PatchCore, inference backends, deployment validation, benchmarking and drift monitoring remain owned by the existing AnomaVision core.

## MVP architecture

```text
Browser
  │
  ▼
Streamlit Studio
  │
  ├── ProjectStore (filesystem JSON)
  ├── Dataset / Model / Deployment services
  │
  ▼
AnomaVision core
  ├── PaDiM / PatchCore
  ├── inference backends
  ├── export / deployment validation
  ├── benchmarking
  └── drift monitoring
```

The first Studio release deliberately uses filesystem storage under `~/.anomavision/projects` and avoids introducing a database or changing the core algorithms.

## Run

```bash
streamlit run apps/studio/app.py
```

Override the project root when needed:

```bash
set ANOMAVISION_STUDIO_ROOT=C:\\path\\to\\projects
streamlit run apps/studio/app.py
```

## Implementation roadmap

1. Studio shell + project lifecycle — implemented.
2. Dataset discovery and dataset-quality checks — implemented.
3. Dataset import/curation and training integration.
3. Training jobs using the existing PaDiM/PatchCore APIs.
4. Model artifacts + evaluation results.
5. Deployment orchestration using existing validation/export code.
6. Live inference using existing runtime/backends.
7. Production monitoring and drift visualization.


### Training integration

Studio now exposes a **Training** workspace. It accepts a dataset root/class folder containing
`train/good`, writes a project-local experiment config, and calls the existing
`anomavision.train.run_training()` implementation.

Studio does not duplicate PaDiM or PatchCore training logic. Training artifacts are written
under the selected project's `models/` area and the latest run metadata is recorded in
`models/latest_training.json`.

The first Studio training controls are intentionally small: algorithm, backbone, batch size,
image size, PaDiM feature dimensions, and PatchCore coreset ratio. More advanced controls can
be added without changing the core training implementation.


### Model registry and evaluation

Studio now discovers trained `model.pt` artifacts from the project's model hierarchy
and exposes them through the **Models** workspace.

Evaluation is an adapter over the existing `anomavision.eval.run_evaluation()` pipeline.
Studio does not implement a second metric or inference path. Evaluation results are saved
with the selected model as `evaluation.json`.

The current evaluation UI is intentionally focused on the existing MVTec-style dataset
contract. Deployment formats and runtime validation remain separate concerns for the next
milestone.
