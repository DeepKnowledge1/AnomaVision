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
