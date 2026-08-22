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


def test_hailo_and_kv260_share_backend_protocol():
    from anomavision.inference.model.backends.base import InferenceBackend
    from anomavision.inference.model.backends.hailo_backend import HailoBackend
    from anomavision.inference.model.backends.k260_backend import KV260Backend

    assert InferenceBackend in HailoBackend.__mro__
    assert InferenceBackend in KV260Backend.__mro__
    assert callable(HailoBackend.predict)
    assert callable(KV260Backend.predict)


def test_accelerator_backends_have_public_documentation():
    from anomavision.inference.model.backends.hailo_backend import HailoBackend
    from anomavision.inference.model.backends.k260_backend import KV260Backend
    from anomavision.inference.model.backends.tensorrt_backend import TensorRTBackend

    for backend in (TensorRTBackend, HailoBackend, KV260Backend):
        assert backend.__doc__
        assert backend.predict.__doc__
        assert backend.close.__doc__
