"""Export complete AnomaVision anomaly graphs for Hailo Dataflow Compiler.

This module produces a fixed-shape, end-to-end ONNX graph for PaDiM or
PatchCore and prepares representative calibration tensors in the format
expected by the Hailo Dataflow Compiler. The Hailo SDK performs the actual
INT8 optimization and HEF compilation.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image

from .graphs import PadimEndToEndGraph, exportable_output_names
from .patchcore import PatchCoreHailoGraph


def _load_artifact(path: Path) -> Any:
    """Load an AnomaVision deployment artifact from disk."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_graph(algorithm: str, artifact: Any, input_size: Tuple[int, int]):
    """Build the fixed-shape end-to-end graph for the selected algorithm."""
    algorithm = algorithm.lower()
    if algorithm == "padim":
        if not isinstance(artifact, dict):
            raise ValueError("PaDiM Hailo export requires a statistics artifact dictionary")
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
        if isinstance(artifact, dict):
            required = {"backbone", "layer_indices", "memory_bank"}
            missing = sorted(required.difference(artifact))
            if missing:
                raise ValueError(f"PatchCore artifact is missing keys: {', '.join(missing)}")
            backbone = str(artifact["backbone"])
            layer_indices = list(artifact["layer_indices"])
            memory_bank = artifact["memory_bank"]
            patch_grid = int(artifact.get("patch_grid", 14))
        else:
            try:
                backbone = str(artifact.backbone)
                layer_indices = list(artifact.layer_indices)
                memory_bank = artifact.memory_bank
                patch_grid = int(getattr(artifact, "patch_grid", 14))
            except AttributeError as exc:
                raise ValueError(
                    "PatchCore artifact must be a PatchCore model or artifact dictionary"
                ) from exc
        return PatchCoreHailoGraph(
            backbone=backbone,
            layer_indices=layer_indices,
            memory_bank=memory_bank,
            patch_grid=patch_grid,
            input_size=input_size,
        )

    raise ValueError("algorithm must be 'padim' or 'patchcore'")


def _prepare_calibration(
    image_dir: Path, output_dir: Path, input_size: Tuple[int, int]
) -> Tuple[Path, Path]:
    """Create Hailo calibration tensors and a JSON manifest.

    Each calibration file is one resized RGB image with shape ``H x W x C``.
    Hailo DFC optimization expects this unbatched shape for the exported
    AnomaVision input. A leading ``1`` must not be stored in the ``.npy`` file.
    """
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in suffixes)
    if not paths:
        raise ValueError(f"No calibration images found in {image_dir}")

    calibration_dir = output_dir / "calibration_npy"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "calibration_manifest.json"
    records = []

    for stale in calibration_dir.glob("*.npy"):
        stale.unlink()

    for index, path in enumerate(paths[:1024]):
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (input_size[1], input_size[0]), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32)
        np.save(calibration_dir / f"sample_{index:04d}.npy", array)
        records.append(
            {
                "path": str(path.resolve()),
                "calibration": str(
                    (calibration_dir / f"sample_{index:04d}.npy").resolve()
                ),
                "shape": list(array.shape),
            }
        )

    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return calibration_dir, manifest


def export_onnx(
    algorithm: str,
    artifact_path: Path,
    output_dir: Path,
    input_size: Tuple[int, int],
    opset: int = 13,
) -> Path:
    """Export the complete anomaly detector as a fixed-shape ONNX graph."""
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
    """Run an optional user-supplied Hailo SDK command template."""
    rendered = command.format(
        onnx=shlex.quote(str(onnx_path)), output=shlex.quote(str(output_dir))
    )
    subprocess.run(rendered, shell=True, check=True, cwd=output_dir)


def main() -> None:
    """Export an end-to-end Hailo graph and prepare calibration data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["padim", "patchcore"], required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--hailo-command")
    args = parser.parse_args()

    input_size = (args.height, args.width)
    onnx_path = export_onnx(
        args.algorithm, args.artifact, args.output_dir, input_size, args.opset
    )
    calibration_dir, manifest = _prepare_calibration(
        args.calibration_dir, args.output_dir, input_size
    )
    metadata = {
        "algorithm": args.algorithm,
        "quantization_scope": "end_to_end",
        "graph_outputs": exportable_output_names(),
        "input_size": list(input_size),
        "onnx": str(onnx_path),
        "calibration_dir": str(calibration_dir),
        "calibration_manifest": str(manifest),
        "hailo_compile_invoked": bool(args.hailo_command),
    }
    (args.output_dir / "hailo_export.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if args.hailo_command:
        _run_hailo_command(args.hailo_command, onnx_path, args.output_dir)
    else:
        print("ONNX graph and Hailo calibration tensors created.")
        print(f"ONNX:        {onnx_path}")
        print(f"Calibration: {calibration_dir}")
        print(
            "No Hailo compiler was invoked; run hailo parser/optimize/compiler "
            "with the generated files to create the HEF."
        )


if __name__ == "__main__":
    main()
