"""Export complete AnomaVision anomaly graphs for Hailo Dataflow Compiler.

This module produces an ONNX graph containing the entire selected algorithm. The
actual INT8 quantization and HEF generation are delegated to the Hailo SDK when
it is installed on the development host. On a machine without the Hailo SDK, the
command still creates the graph and a calibration manifest, then exits with a
clear hardware-toolchain instruction instead of silently producing a partial
CPU artifact.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image

from .hailo_graphs import (
    PadimEndToEndGraph,
    PatchCoreEndToEndGraph,
    exportable_output_names,
)


def _load_artifact(path: Path) -> Dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError(f"Expected a dictionary artifact, got {type(artifact)!r}")
    return artifact


def _build_graph(algorithm: str, artifact: Dict[str, Any], input_size: Tuple[int, int]):
    algorithm = algorithm.lower()
    if algorithm == "padim":
        required = {"backbone", "layer_indices", "channel_indices", "mean", "cov_inv"}
        missing = sorted(required.difference(artifact))
        if missing:
            raise ValueError(f"PaDiM artifact is missing keys: {', '.join(missing)}")
        return PadimEndToEndGraph(
            backbone=str(artifact["backbone"]),
            layer_indices=list(artifact["layer_indices"]),
            channel_indices=artifact["channel_indices"],
            mean=artifact["mean"],
            cov_inv=artifact["cov_inv"],
            input_size=input_size,
        )
    if algorithm == "patchcore":
        required = {"backbone", "layer_indices", "memory_bank"}
        missing = sorted(required.difference(artifact))
        if missing:
            raise ValueError(
                f"PatchCore artifact is missing keys: {', '.join(missing)}"
            )
        return PatchCoreEndToEndGraph(
            backbone=str(artifact["backbone"]),
            layer_indices=list(artifact["layer_indices"]),
            memory_bank=artifact["memory_bank"],
            patch_grid=artifact.get("patch_grid", 14),
            input_size=input_size,
        )
    raise ValueError("algorithm must be 'padim' or 'patchcore'")


def _write_calibration_manifest(
    image_dir: Path, output_dir: Path, input_size: Tuple[int, int]
) -> Path:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in suffixes)
    if not paths:
        raise ValueError(f"No calibration images found in {image_dir}")
    manifest = output_dir / "calibration_manifest.json"
    records = []
    for path in paths:
        with Image.open(path) as image:
            image.convert("RGB").resize((input_size[1], input_size[0]))
        records.append(
            {
                "path": str(path.resolve()),
                "width": input_size[1],
                "height": input_size[0],
            }
        )
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return manifest


def export_onnx(
    algorithm: str,
    artifact_path: Path,
    output_dir: Path,
    input_size: Tuple[int, int],
    opset: int = 17,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = _load_artifact(artifact_path)
    graph = _build_graph(algorithm, artifact, input_size).eval()
    sample = torch.zeros(1, 3, input_size[0], input_size[1], dtype=torch.float32)
    output_path = output_dir / f"anomavision_{algorithm.lower()}_k26_end_to_end.onnx"
    with torch.no_grad():
        torch.onnx.export(
            graph,
            sample,
            output_path,
            input_names=["images"],
            output_names=exportable_output_names(),
            dynamic_axes=None,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return output_path


def _run_hailo_command(command: str, onnx_path: Path, output_dir: Path) -> None:
    rendered = command.format(
        onnx=shlex.quote(str(onnx_path)), output=shlex.quote(str(output_dir))
    )
    subprocess.run(rendered, shell=True, check=True, cwd=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["padim", "patchcore"], required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument(
        "--hailo-command",
        help=(
            "Optional installed Hailo SDK command template. Use {onnx} and {output}; "
            "the command must perform parse, calibration/optimization, and compile."
        ),
    )
    args = parser.parse_args()
    input_size = (args.height, args.width)
    onnx_path = export_onnx(
        args.algorithm, args.artifact, args.output_dir, input_size, args.opset
    )
    manifest = _write_calibration_manifest(
        args.calibration_dir, args.output_dir, input_size
    )
    metadata = {
        "algorithm": args.algorithm,
        "quantization_scope": "end_to_end",
        "graph_outputs": exportable_output_names(),
        "input_size": list(input_size),
        "onnx": str(onnx_path),
        "calibration_manifest": str(manifest),
        "hailo_compile_invoked": bool(args.hailo_command),
    }
    (args.output_dir / "hailo_export.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if args.hailo_command:
        _run_hailo_command(args.hailo_command, onnx_path, args.output_dir)
    else:
        print("ONNX graph and calibration manifest created.")
        print(
            "No Hailo compiler was invoked; install the Hailo SDK and provide --hailo-command to create a quantized HEF."
        )


if __name__ == "__main__":
    main()
