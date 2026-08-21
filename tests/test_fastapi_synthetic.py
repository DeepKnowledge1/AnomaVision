"""Regression tests for the production Synthetic Defect Studio API routes."""

import base64
import importlib.util
import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "apps" / "api" / "fastapi_app.py"
spec = importlib.util.spec_from_file_location("anomavision_fastapi_app_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def image_bytes(image: Image.Image, fmt: str = "PNG") -> io.BytesIO:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


def decode_image(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_synthetic_routes_are_registered():
    paths = {route.path for route in api.app.routes}
    assert {"/synthetic/generate", "/synthetic/reuse"} <= paths


def test_procedural_generation_endpoint_returns_image_mask_and_metadata():
    normal = Image.new("RGB", (96, 96), (180, 180, 180))
    client = TestClient(api.app)
    response = client.post(
        "/synthetic/generate?defect_type=scratch&severity=medium&seed=7",
        files={"image_file": ("normal.png", image_bytes(normal), "image/png")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    generated = decode_image(payload["generated_image_base64"])
    mask = decode_image(payload["ground_truth_mask_base64"])
    assert generated.size == normal.size
    assert mask.size == normal.size
    assert payload["metadata"]["defect_type"] == "scratch"
    assert np.asarray(mask).max() == 255


def test_real_defect_reuse_endpoint_returns_deterministic_placement_metadata():
    normal = Image.new("RGB", (96, 96), (180, 180, 180))
    defect = Image.new("RGB", (32, 32), (20, 20, 20))
    defect_mask = Image.new("L", (32, 32), 255)
    files = [
        ("normal_file", ("normal.png", image_bytes(normal), "image/png")),
        ("defect_files", ("defect.png", image_bytes(defect), "image/png")),
        ("mask_files", ("mask.png", image_bytes(defect_mask), "image/png")),
    ]
    client = TestClient(api.app)
    response = client.post(
        "/synthetic/reuse?copies_per_source=2&seed=19",
        files=files,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert decode_image(payload["generated_image_base64"]).size == normal.size
    assert decode_image(payload["ground_truth_mask_base64"]).size == normal.size
    assert payload["metadata"]["placement_count"] == 2
    assert payload["metadata"]["seed"] == 19


def test_real_defect_reuse_rejects_invalid_sensitivity():
    normal = Image.new("RGB", (96, 96), (180, 180, 180))
    defect = Image.new("RGB", (32, 32), (20, 20, 20))
    client = TestClient(api.app)
    response = client.post(
        "/synthetic/reuse?sensitivity=0",
        files=[
            ("normal_file", ("normal.png", image_bytes(normal), "image/png")),
            ("defect_files", ("defect.png", image_bytes(defect), "image/png")),
        ],
    )

    assert response.status_code == 400
