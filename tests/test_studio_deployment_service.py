from apps.studio.services import deployment_service

def test_target_descriptions():
    assert "Portable" in deployment_service.TARGET_DESCRIPTIONS["onnx"]

def test_unsupported_target_is_clear(tmp_path):
    model = {"path": str(tmp_path / "model.pt"), "config": str(tmp_path / "config.yml"), "run_name": "run"}
    try:
        deployment_service.deploy_model(tmp_path, model, "hailo")
    except (ValueError, FileNotFoundError):
        pass
    else:
        raise AssertionError("Expected unsupported target error")
