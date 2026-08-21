"""Regression tests for the dual-server GUI launcher imports."""

import importlib.util
from pathlib import Path

LAUNCHER_PATH = Path(__file__).resolve().parents[1] / "apps" / "ui" / "run_app.py"
spec = importlib.util.spec_from_file_location("anomavision_run_app_test", LAUNCHER_PATH)
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_launcher_loads_the_production_fastapi_module_by_path():
    module = launcher.load_module_from_path(
        "anomavision_fastapi_app_test", launcher.FASTAPI_MODULE_PATH
    )
    assert hasattr(module, "app")
    paths = {route.path for route in module.app.routes}
    assert "/synthetic/generate" in paths
    assert "/synthetic/reuse" in paths


def test_launcher_loads_the_gradio_module_by_path():
    module = launcher.load_module_from_path(
        "anomavision_gradio_app_test", launcher.GRADIO_MODULE_PATH
    )
    assert hasattr(module, "demo")
    assert module.demo is not None
