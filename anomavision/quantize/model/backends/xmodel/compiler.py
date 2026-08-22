"""Compile an AMD Vitis AI XIR graph into an XModel for a K26 DPU.

This path is independent from Hailo-8: Hailo artifacts are HEF files, while
AMD DPU artifacts are XModel files. The compiler must be supplied by the Vitis
AI environment used for the target K26/KV260 image.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict


def compile_xmodel(
    xir_path: str | Path,
    arch_path: str | Path,
    output_dir: str | Path,
    compiler: str = "vai_c_xir",
) -> Dict[str, Any]:
    """Compile ``xir_path`` with ``vai_c_xir`` and return an artifact manifest."""
    xir_path = Path(xir_path)
    arch_path = Path(arch_path)
    output_dir = Path(output_dir)
    if not xir_path.is_file():
        raise FileNotFoundError(xir_path)
    if not arch_path.is_file():
        raise FileNotFoundError(arch_path)
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(
            f"{compiler} was not found. Activate the Vitis AI compiler environment."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        compiler_path,
        "-x",
        str(xir_path),
        "-a",
        str(arch_path),
        "-o",
        str(output_dir),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "XModel compilation failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    artifacts = sorted(output_dir.glob("*.xmodel"))
    if not artifacts:
        raise RuntimeError("Vitis AI compiler returned success but created no .xmodel")
    manifest = {
        "xir": str(xir_path.resolve()),
        "architecture": str(arch_path.resolve()),
        "compiler": compiler_path,
        "command": command,
        "xmodel": str(artifacts[0].resolve()),
    }
    (output_dir / "xmodel_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xir", type=Path, required=True)
    parser.add_argument("--arch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", default="vai_c_xir")
    args = parser.parse_args()
    manifest = compile_xmodel(args.xir, args.arch, args.output_dir, args.compiler)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
