from pathlib import Path

import pytest

from anomavision.inference.modelType import ModelType
from anomavision.quantize.model.backends.xmodel.compiler import compile_xmodel


def test_model_type_detects_hardware_artifacts(tmp_path):
    assert ModelType.from_extension(tmp_path / "model.hef") is ModelType.HEF
    assert ModelType.from_extension(tmp_path / "model.xmodel") is ModelType.XMODEL


def test_xmodel_compiler_fails_before_toolchain_lookup(tmp_path):
    with pytest.raises(FileNotFoundError):
        compile_xmodel(
            tmp_path / "missing.xir", tmp_path / "arch.json", tmp_path / "out"
        )


def test_xmodel_compiler_requires_architecture_file(tmp_path):
    xir = tmp_path / "model.xir"
    xir.write_bytes(b"xir")
    with pytest.raises(FileNotFoundError):
        compile_xmodel(xir, tmp_path / "missing.json", tmp_path / "out")
