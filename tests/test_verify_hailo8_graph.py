import json

import onnx
import onnx.helper
import pytest

from anomavision.quantize.model.backends.hef.verifier import inspect_onnx, verify_graph


def _write_complete_model(path):
    nodes = [
        onnx.helper.make_node("Identity", ["images"], ["image_scores"]),
        onnx.helper.make_node("Identity", ["images"], ["score_map"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "complete",
        [
            onnx.helper.make_tensor_value_info(
                "images", onnx.TensorProto.FLOAT, [1, 3, 32, 32]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "image_scores", onnx.TensorProto.FLOAT, [1, 3, 32, 32]
            ),
            onnx.helper.make_tensor_value_info(
                "score_map", onnx.TensorProto.FLOAT, [1, 3, 32, 32]
            ),
        ],
    )
    onnx.save(onnx.helper.make_model(graph), path)


def test_complete_graph_is_onnx_only_without_hef(tmp_path):
    path = tmp_path / "model.onnx"
    _write_complete_model(path)
    report = verify_graph(path)
    assert report["status"] == "onnx_only_not_hardware_verified"
    assert report["compiler_verified"] is False


def test_hef_requires_compiler_evidence(tmp_path):
    path = tmp_path / "model.onnx"
    _write_complete_model(path)
    hef = tmp_path / "model.hef"
    hef.write_bytes(b"hef")
    with pytest.raises(RuntimeError, match="cannot prove"):
        verify_graph(path, hef_path=hef)


def test_fallback_marker_is_rejected(tmp_path):
    path = tmp_path / "model.onnx"
    _write_complete_model(path)
    hef = tmp_path / "model.hef"
    hef.write_bytes(b"hef")
    log = tmp_path / "compiler.log"
    log.write_text("CPU fallback partition created", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Fallback markers"):
        verify_graph(path, hef_path=hef, compiler_log=log)


def test_clean_compiler_log_passes(tmp_path):
    path = tmp_path / "model.onnx"
    _write_complete_model(path)
    hef = tmp_path / "model.hef"
    hef.write_bytes(b"hef")
    log = tmp_path / "compiler.log"
    log.write_text("Compilation completed for Hailo-8", encoding="utf-8")
    report = verify_graph(path, hef_path=hef, compiler_log=log)
    assert report["fallback_verified"] is True
    assert report["status"] == "device_graph_no_fallback_markers"
