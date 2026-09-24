"""FastAPI adapter for AnomaVision Studio.

The API is deliberately thin: it exposes the existing filesystem-backed
Studio services and AnomaVision engine without duplicating ML logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from apps.studio.services.catalog import ALGORITHMS, DEPLOYMENT_TARGETS
from apps.studio.services.dataset_service import inspect_dataset, save_dataset_manifest
from apps.studio.services.deployment_service import deploy_model, TARGET_DESCRIPTIONS
from apps.studio.services.monitoring_service import list_monitoring_reports, monitoring_summary
from apps.studio.services.model_registry import list_models, get_model
from apps.studio.services.project_store import ProjectStore
from apps.studio.services.training_service import train_project


STORE_ROOT = Path(
    os.getenv("ANOMAVISION_STUDIO_ROOT", "~/.anomavision/projects")
).expanduser()

store = ProjectStore(STORE_ROOT)

app = FastAPI(
    title="AnomaVision Studio API",
    version="0.1.0",
    description="Thin API layer for the AnomaVision Studio web interface.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "ANOMAVISION_STUDIO_CORS", "http://localhost:3000"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    algorithm: str = "patchcore"
    description: str = ""


class DatasetInspect(BaseModel):
    path: str = Field(min_length=1)
    recursive: bool = True


class TrainingRequest(BaseModel):
    dataset_path: str = Field(min_length=1)
    algorithm: str = "patchcore"
    class_name: str = "default"
    batch_size: int | None = Field(default=None, ge=1, le=256)
    resize: list[int] | None = Field(default=None, min_length=2, max_length=2)
    backbone: str | None = None
    feat_dim: int | None = Field(default=None, ge=1)
    coreset_ratio: float | None = Field(default=None, gt=0, le=1)


def _project_or_404(project_id: str) -> dict[str, Any]:
    project = store.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return project


def _project_dir(project_id: str) -> Path:
    return STORE_ROOT / project_id


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "anomavision-studio"}


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    return {
        "algorithms": ALGORITHMS,
        "deployment_targets": DEPLOYMENT_TARGETS,
    }


@app.get("/api/projects")
def projects() -> list[dict[str, Any]]:
    return store.list_projects()


@app.post("/api/projects", status_code=201)
def create_project(request: ProjectCreate) -> dict[str, Any]:
    algorithm = request.algorithm.lower()
    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algorithm}")

    try:
        project = store.create(request.name, algorithm)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if request.description.strip():
        project["description"] = request.description.strip()
        project_path = STORE_ROOT / project["id"] / "project.json"
        project_path.write_text(
            __import__("json").dumps(project, indent=2) + "\n",
            encoding="utf-8",
        )
    return project


@app.get("/api/projects/{project_id}")
def project(project_id: str) -> dict[str, Any]:
    return _project_or_404(project_id)


@app.get("/api/projects/{project_id}/models")
def models(project_id: str) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    return list_models(_project_dir(project_id))


@app.post("/api/projects/{project_id}/datasets/inspect")
def inspect_project_dataset(
    project_id: str, request: DatasetInspect
) -> dict[str, Any]:
    _project_or_404(project_id)
    try:
        report = inspect_dataset(request.path, recursive=request.recursive)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    manifest = _project_dir(project_id) / "datasets" / "dataset_report.json"
    save_dataset_manifest(manifest, report)
    return report


@app.post("/api/projects/{project_id}/training")
def start_training(
    project_id: str, request: TrainingRequest
) -> dict[str, Any]:
    _project_or_404(project_id)

    algorithm = request.algorithm.lower()
    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algorithm}")

    overrides: dict[str, Any] = {}
    for name in ("batch_size", "backbone", "feat_dim", "coreset_ratio"):
        value = getattr(request, name)
        if value is not None:
            overrides[name] = value
    if request.resize is not None:
        overrides["resize"] = request.resize

    try:
        return train_project(
            project_dir=_project_dir(project_id),
            dataset_source=request.dataset_path,
            algorithm=algorithm,
            class_name=request.class_name,
            **overrides,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {exc}",
        ) from exc


class DeploymentRequest(BaseModel):
    model_id: str = Field(min_length=1)
    target: str = "onnx"
    runs: int = Field(default=5, ge=1, le=100)
    warmup_runs: int = Field(default=1, ge=0, le=20)


@app.get("/api/deployments/targets")
def deployment_targets() -> dict[str, str]:
    return TARGET_DESCRIPTIONS


@app.get("/api/projects/{project_id}/deployments")
def deployments(project_id: str) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    root = _project_dir(project_id) / "deployments"
    results: list[dict[str, Any]] = []
    for path in root.glob("*/**/deployment.json"):
        try:
            results.append(__import__("json").loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(results, key=lambda item: item.get("elapsed_seconds", 0), reverse=True)


@app.post("/api/projects/{project_id}/deployments")
def start_deployment(project_id: str, request: DeploymentRequest) -> dict[str, Any]:
    _project_or_404(project_id)
    try:
        model = get_model(_project_dir(project_id), request.model_id)
        return deploy_model(
            _project_dir(project_id),
            model,
            request.target,
            runs=request.runs,
            warmup_runs=request.warmup_runs,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {exc}") from exc


@app.get("/api/projects/{project_id}/monitoring")
def monitoring(project_id: str) -> dict[str, Any]:
    _project_or_404(project_id)
    return monitoring_summary(_project_dir(project_id))

@app.get("/api/projects/{project_id}/monitoring/reports")
def monitoring_reports(project_id: str) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    return list_monitoring_reports(_project_dir(project_id))
