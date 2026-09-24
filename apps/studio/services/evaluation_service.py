"""Studio adapter for the existing AnomaVision evaluation pipeline."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from anomavision.eval import run_evaluation

from apps.studio.services.model_registry import get_model


def evaluate_project_model(
    project_dir: Path | str,
    model_id: str,
    dataset_path: str,
    device: str = "cpu",
    batch_size: int = 1,
    threshold: float | None = None,
) -> dict[str, Any]:
    project_dir = Path(project_dir)
    model = get_model(project_dir, model_id)

    config = Path(model["config"])
    if not config.is_file():
        raise FileNotFoundError(f"Model config not found: {config}")

    args = Namespace(
        config=str(config),
        dataset_path=str(Path(dataset_path).expanduser().resolve()),
        class_name=model["class_name"],
        model_data_path=str(project_dir / "models"),
        algorithm=model["algorithm"],
        model="model.pt",
        device=device,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=False,
        thresh=threshold,
        enable_visualization=False,
        save_visualizations=False,
        viz_output_dir=None,
        log_level="WARNING",
        detailed_timing=False,
    )
    metrics, _ = run_evaluation(args)

    evaluation = {
        "model_id": model_id,
        "model": model["path"],
        "dataset_path": args.dataset_path,
        "class_name": args.class_name,
        "metrics": metrics,
    }

    output = project_dir / "models" / model["run_name"] / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    output.write_text(json.dumps(evaluation, indent=2, default=float) + "\n", encoding="utf-8")
    return evaluation
