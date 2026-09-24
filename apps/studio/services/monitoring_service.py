"""Read-only Studio adapters for existing production monitoring artifacts."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def list_monitoring_reports(project_dir: Path | str) -> list[dict[str, Any]]:
    root = Path(project_dir) / "monitoring"
    reports = []
    if not root.exists():
        return reports
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "drift_score" in payload or "psi" in payload:
            payload["_file"] = str(path)
            reports.append(payload)
    return sorted(reports, key=lambda item: str(item.get("_file", "")), reverse=True)

def monitoring_summary(project_dir: Path | str) -> dict[str, Any]:
    reports = list_monitoring_reports(project_dir)
    latest = reports[0] if reports else None
    return {
        "report_count": len(reports),
        "latest": latest,
        "status": latest.get("status") if latest else "no_data",
        "drift_score": latest.get("drift_score") if latest else None,
        "psi": latest.get("psi") if latest else None,
        "warnings": latest.get("warnings", []) if latest else [],
    }
