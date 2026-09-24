from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from anomavision.deployment_validation import validate_model


def _make_model(path: Path) -> None:
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph(
        [node],
        "validation",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    model = helper.make_model(
        graph,
        producer_name="anomavision-test",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=11,
    )
    onnx.save(model, path)


def test_validate_onnx_is_observational(tmp_path):
    model_path = tmp_path / "model.onnx"
    _make_model(model_path)
    before = model_path.read_bytes()

    report = validate_model(model_path, runs=2, warmup_runs=1)

    assert report["ready_for_deployment"] is True
    assert report["inputs"][0]["shape"] == [1, 3, 8, 8]
    assert report["outputs"][0]["shape"] == [1, 3, 8, 8]
    assert report["performance"]["latency_ms"] >= 0
    assert model_path.read_bytes() == before


def test_validate_missing_model(tmp_path):
    try:
        validate_model(tmp_path / "missing.onnx")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
