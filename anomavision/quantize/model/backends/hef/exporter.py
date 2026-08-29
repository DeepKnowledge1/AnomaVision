"""Export complete AnomaVision anomaly graphs for Hailo Dataflow Compiler.

This module produces a fixed-shape, end-to-end ONNX graph for PaDiM, PatchCore,
or EfficientAD and prepares representative calibration tensors in the format
expected by the Hailo Dataflow Compiler. The Hailo SDK performs the actual INT8
optimization and HEF compilation.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image

from .efficientad import EfficientADHailoGraph
from .graphs import PadimEndToEndGraph, exportable_output_names
from .patchcore import PatchCoreHailoGraph


def _load_artifact(path: Path) -> Any:
    """Load an AnomaVision deployment artifact from disk."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_efficientad_graph(artifact: Any, input_size: Tuple[int, int]):
    """Build the fixed-shape EfficientAD Hailo graph."""
    from anomavision.algorithm.efficientad.efficientad import EfficientAD

    if isinstance(artifact, EfficientAD):
        model = artifact
    elif isinstance(artifact, dict) and artifact.get("algorithm") == "efficientad":
        model = EfficientAD(
            device=torch.device("cpu"),
            model_size=artifact.get("model_size", "s"),
            pretrained_teacher=False,
            threshold_quantile=artifact.get("threshold_quantile", 0.995),
        )
        model.load_state_dict(artifact["model_state"], strict=True)
    else:
        raise ValueError(
            "EfficientAD Hailo export requires an EfficientAD model or "
            "an EfficientAD statistics artifact"
        )

    if not bool(model.trained.item()):
        raise ValueError("EfficientAD artifact is not trained/calibrated")
    return EfficientADHailoGraph(model, input_size=input_size)


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

    if algorithm == "efficientad":
        return _build_efficientad_graph(artifact, input_size)

    raise ValueError("algorithm must be 'padim', 'patchcore' or 'efficientad'")


def _write_calibration_manifest(
    image_dir: Path, output_dir: Path, input_size: Tuple[int, int]
) -> Path:
    """Create normalized calibration tensors and a JSON manifest."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in suffixes)
    if not paths:
        raise ValueError(f"No calibration images found in {image_dir}")

    calibration_dir = output_dir / "calibration_npy"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "calibration_manifest.json"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    records = []

    for stale in calibration_dir.glob("*.npy"):
        stale.unlink()

    for index, path in enumerate(paths[:1024]):
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (input_size[1], input_size[0]), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(
            std, dtype=np.float32
        )
        np.save(calibration_dir / f"sample_{index:04d}.npy", array)
        records.append(
            {
                "path": str(path.resolve()),
                "calibration": str((calibration_dir / f"sample_{index:04d}.npy").resolve()),
                "shape": list(array.shape),
                "normalized": True,
                "normalization": {
                    "mean": list(mean),
                    "std": list(std),
                    "scale": "1/255 before mean/std",
                },
            }
        )

    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return manifest


def _prepare_calibration(
    image_dir: Path, output_dir: Path, input_size: Tuple[int, int]
) -> Tuple[Path, Path]:
    """Create Hailo calibration tensors and a JSON manifest."""
    manifest = _write_calibration_manifest(image_dir, output_dir, input_size)
    return manifest.parent / "calibration_npy", manifest


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
    parser.add_argument(
        "--algorithm", choices=["padim", "patchcore", "efficientad"], required=True
    )
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
    print(f"ONNX: {onnx_path}")
    print(f"Calibration: {calibration_dir}")
    print(f"Manifest: {manifest}")

    if args.hailo_command:
        _run_hailo_command(args.hailo_command, onnx_path, args.output_dir)


if __name__ == "__main__":
    main()
