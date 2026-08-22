"""Verify complete AnomaVision ONNX/HEF graphs for Hailo-8 deployment.

This tool separates three facts that are often confused:

* ONNX operator inventory: what the graph requests.
* Hailo compiler result: whether the DFC accepted the graph and produced a HEF.
* Runtime fallback: whether a separate ONNX Runtime/CPU postprocess path was used.

The standard ``ClientRunner`` compile flow does not create a CPU fallback
partition. Unsupported operations fail during translation/optimization instead.
This script therefore treats a successful HEF from that flow as device-compiled
only when no external ONNX Runtime fallback command or fallback marker is found.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REQUIRED_OUTPUTS = {"image_scores", "score_map"}
FALLBACK_PATTERNS = (
    r"onnx\s*runtime",
    r"cpu\s*(fallback|postprocess|partition|execution)",
    r"fallback\s*(op|layer|partition|execution)",
    r"unsupported\s*(op|operator|layer).*fallback",
    r"host\s*(fallback|postprocess)",
)


def inspect_onnx(path: Path) -> Dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Install onnx to inspect an ONNX graph") from exc
    model = onnx.load(str(path))
    outputs = {value.name for value in model.graph.output}
    ops = sorted({node.op_type for node in model.graph.node})
    return {
        "onnx": str(path.resolve()),
        "operators": ops,
        "operator_count": len(model.graph.node),
        "outputs": sorted(outputs),
        "complete_outputs": REQUIRED_OUTPUTS.issubset(outputs),
    }


def _read_har_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        with zipfile.ZipFile(path) as archive:
            chunks = []
            for name in archive.namelist():
                if name.endswith((".json", ".txt", ".yaml", ".yml", ".xml")):
                    chunks.append(archive.read(name).decode("utf-8", errors="ignore"))
            return "\n".join(chunks)
    except (zipfile.BadZipFile, OSError):
        return ""


def scan_for_fallback_markers(paths: Sequence[Path]) -> List[str]:
    findings = []
    for path in paths:
        if not path.is_file():
            continue
        text = (
            _read_har_text(path)
            if path.suffix.lower() == ".har"
            else path.read_text(encoding="utf-8", errors="ignore")
        )
        for pattern in FALLBACK_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                findings.append(f"{path}: {pattern}")
    return findings


def verify_graph(
    onnx_path: Path,
    hef_path: Path | None = None,
    har_path: Path | None = None,
    compiler_log: Path | None = None,
) -> Dict[str, Any]:
    report = inspect_onnx(onnx_path)
    if not report["complete_outputs"]:
        raise RuntimeError(
            f"{onnx_path} is not a complete AnomaVision graph; required outputs are "
            f"{sorted(REQUIRED_OUTPUTS)}"
        )
    if hef_path is None:
        report.update(
            {
                "compiler_verified": False,
                "fallback_verified": False,
                "status": "onnx_only_not_hardware_verified",
            }
        )
        return report
    if not hef_path.is_file() or hef_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty HEF: {hef_path}")
    scan_paths = [p for p in (har_path, compiler_log) if p is not None]
    evidence = [path for path in scan_paths if path.is_file()]
    if not evidence:
        raise RuntimeError(
            "A HEF file alone cannot prove that no fallback occurred. Provide "
            "--compiler-log or --*-har from the same Hailo compilation."
        )
    findings = scan_for_fallback_markers(evidence)
    report.update(
        {
            "hef": str(hef_path.resolve()),
            "har": str(har_path.resolve()) if har_path and har_path.exists() else None,
            "compiler_log": str(compiler_log.resolve()) if compiler_log else None,
            "compiler_verified": True,
            "fallback_evidence_files": [str(path.resolve()) for path in evidence],
            "fallback_markers": findings,
            "fallback_verified": not findings,
            "status": (
                "device_graph_no_fallback_markers"
                if not findings
                else "fallback_markers_found"
            ),
        }
    )
    if findings:
        raise RuntimeError(
            "Fallback markers found; refusing to mark the HEF as end-to-end: "
            + "; ".join(findings)
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padim-onnx", type=Path, required=True)
    parser.add_argument("--patchcore-onnx", type=Path, required=True)
    parser.add_argument("--padim-hef", type=Path)
    parser.add_argument("--patchcore-hef", type=Path)
    parser.add_argument("--padim-har", type=Path)
    parser.add_argument("--patchcore-har", type=Path)
    parser.add_argument("--compiler-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        verify_graph(
            args.padim_onnx, args.padim_hef, args.padim_har, args.compiler_log
        ),
        verify_graph(
            args.patchcore_onnx,
            args.patchcore_hef,
            args.patchcore_har,
            args.compiler_log,
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
