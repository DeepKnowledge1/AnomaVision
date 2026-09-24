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
from apps.studio.services.training_service import train_project, normalize_dataset_source
from anomavision.config import load_config


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
    class_name: str | None = None


class TrainingRequest(BaseModel):
    dataset_path: str = Field(min_length=1)
    algorithm: str = "patchcore"
    class_name: str | None = None
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


@app.get("/api/config")
def config() -> dict[str, Any]:
    """Expose the relevant canonical config.yml values to Studio."""
    config_path = Path(os.getenv("ANOMAVISION_CONFIG", "config.yml"))
    try:
        data = load_config(str(config_path)) or {}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    algorithm = str(data.get("algorithm", "patchcore")).lower()
    thresholds = {
        "padim": float(data.get("thresh_padim", 13.0)),
        "patchcore": float(data.get("thresh_patchcore", 0.25)),
        "efficientad": float(data.get("thresh_efficientad", 1.0)),
    }
    resize = data.get("resize", [224, 224])
    return {
        "algorithm": algorithm,
        "dataset_path": str(data.get("dataset_path", "") or ""),
        "class_name": str(data.get("class_name", "default") or "default"),
        "resize": resize,
        "normalize": bool(data.get("normalize", True)),
        "thresholds": thresholds,
        "stream": {
            "display_fps": bool(data.get("stream_display_fps", True)),
            "max_frames": data.get("stream_max_frames"),
        },
        "drift": {
            "enabled": bool(data.get("enable_drift_monitoring", False)),
            "window": int(data.get("drift_window", 50)),
            "min_samples": int(data.get("drift_min_samples", 5)),
            "threshold": float(data.get("drift_threshold", 0.20)),
            "evaluation_interval": int(data.get("drift_evaluation_interval", 25)),
        },
    }


def _choose_dataset_folder(initial_dir: str = "") -> str:
    """Open a native folder picker on the local Windows desktop."""
    if os.name != "nt":
        raise RuntimeError("Native folder picker is currently supported on Windows only.")

    import subprocess

    script = r"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select AnomaVision dataset folder'
$dialog.ShowNewFolderButton = $false
$initial = $env:ANOMAVISION_PICKER_INITIAL_DIR
if ($initial -and (Test-Path -LiteralPath $initial -PathType Container)) {
    $dialog.SelectedPath = $initial
}
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.Write($dialog.SelectedPath)
}
$dialog.Dispose()
"""
    env = os.environ.copy()
    env["ANOMAVISION_PICKER_INITIAL_DIR"] = initial_dir
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "PowerShell folder picker failed."
        raise RuntimeError(detail)
    return result.stdout.strip()


@app.get("/api/dataset/pick-folder")
def pick_dataset_folder() -> dict[str, str]:
    """Open the local OS folder picker and return the selected absolute path."""
    config_path = Path(os.getenv("ANOMAVISION_CONFIG", "config.yml"))
    initial_dir = ""
    try:
        data = load_config(str(config_path)) or {}
        initial_dir = str(data.get("dataset_path", "") or "")
    except (FileNotFoundError, ValueError):
        pass

    try:
        return {"path": _choose_dataset_folder(initial_dir)}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Native folder picker unavailable: {exc}",
        ) from exc


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


@app.post("/api/projects/{project_id}/datasets/resolve")
def resolve_project_dataset(
    project_id: str, request: DatasetInspect
) -> dict[str, str]:
    """Validate and normalize a dataset selection to train.py's contract."""
    _project_or_404(project_id)
    config_data = load_config(str(Path(os.getenv("ANOMAVISION_CONFIG", "config.yml")))) or {}
    class_name = str(request.class_name or config_data.get("class_name", "default") or "default")
    try:
        dataset_path, detected_class = normalize_dataset_source(request.path, class_name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "dataset_path": str(dataset_path),
        "class_name": detected_class,
        "train_good": str(dataset_path / detected_class / "train" / "good")
        if (dataset_path / detected_class / "train" / "good").is_dir()
        else str(dataset_path / "train" / "good"),
    }


@app.post("/api/projects/{project_id}/training")
def start_training(
    project_id: str, request: TrainingRequest
) -> dict[str, Any]:
    _project_or_404(project_id)

    algorithm = request.algorithm.lower()
    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algorithm}")

    # Use the canonical config class when the UI does not provide one.
    config_data = load_config(str(Path(os.getenv("ANOMAVISION_CONFIG", "config.yml")))) or {}
    class_name = str(request.class_name or config_data.get("class_name", "default") or "default")

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
            class_name=class_name,
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
