"""
ui.py
-----
Gradio frontend for AnomaVision — fully decoupled from the model.

This process never imports inference_engine and never touches ONNX
Runtime. It sends images to the API service over HTTP (POST /predict)
and renders whatever JSON comes back. That means the UI container can
ship without numpy/onnxruntime/the anomavision package at all, and the
API and UI can scale, restart, and deploy independently (e.g. separate
Kubernetes Deployments + Services).

Run standalone (API must already be reachable):

    ANOMAVISION_API_URL=http://localhost:8000 python ui.py
"""

import base64
import io
import os
import socket
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import requests
from PIL import Image

# -----------------------------------------------------------------------------
# Force IPv4-only DNS resolution
# -----------------------------------------------------------------------------
# Docker/WSL2 networking commonly answers A (IPv4) records fine but never
# answers AAAA (IPv6) ones. getaddrinfo() tries AAAA first by default and
# waits out a ~2s timeout before falling back to the working IPv4 address
# — paid on EVERY request. Since this service only ever talks IPv4 inside
# the cluster/compose network, skip the IPv6 attempt entirely.
_original_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_only_getaddrinfo

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
API_URL = os.getenv("ANOMAVISION_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("ANOMAVISION_API_TIMEOUT", "30"))
SAMPLE_DIR = os.getenv("SAMPLE_IMAGES_DIR", "sample_images")
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# A bare requests.get/post() call re-reads OS proxy settings (WinHTTP /
# registry on Windows) and opens a fresh TCP connection EVERY call. On
# Windows that proxy auto-detection lookup alone routinely costs ~2s per
# request — trust_env=False skips it, and reusing one Session keeps the
# connection to the API alive (HTTP keep-alive) instead of re-handshaking.
_session = requests.Session()
_session.trust_env = False


def _fetch_default_threshold() -> float:
    """Ask the API for its current default threshold so the slider starts
    sane. Falls back to an env var / hardcoded value if the API isn't up
    yet — the UI should still render, just call out the API is offline."""
    try:
        r = _session.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return float(r.json().get("threshold", 13.0))
    except Exception:
        return float(os.getenv("ANOMAVISION_THRESHOLD", "13.0"))


THRESHOLD_DEFAULT = _fetch_default_threshold()


# -----------------------------------------------------------------------------
# Sample image helpers
# -----------------------------------------------------------------------------
def _collect_samples() -> list:
    samples = []
    base = Path(SAMPLE_DIR)
    if not base.exists():
        return samples
    for p in sorted(base.rglob("*")):
        if p.suffix.lower() in SUPPORTED_EXT:
            rel = p.relative_to(base)
            parts = rel.parts
            if len(parts) >= 3:
                label = f"{parts[0]}/{parts[1]}"
            elif len(parts) == 2:
                label = f"{parts[0]}/{p.stem}"
            else:
                label = p.stem
            samples.append((label, str(p)))
    return samples


SAMPLES = _collect_samples()


def _sample_gallery_items() -> list:
    return [(path, label) for label, path in SAMPLES if Path(path).exists()]


def _load_sample(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB") if path and Path(path).exists() else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# base64 PNG (from the API's JSON response) → PIL, for Gradio Image outputs
# -----------------------------------------------------------------------------
def _b64_to_pil(b64_str: Optional[str]) -> Optional[Image.Image]:
    if not b64_str:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_str)))
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Inference callback — called directly by Gradio, now over HTTP
#
# Data flow:
#   PIL Image (from gr.Image)
#     → PNG bytes                   [encode once, here]
#     → POST {API_URL}/predict      [multipart upload + query params]
#     → JSON {score, base64 PNGs}   [API does all the numpy/onnx work]
#     → base64 → PIL  ×2            [decode for display]
#     → gr.Image renders
#
# One network hop each way. The API is the only process that ever
# imports inference_engine / onnxruntime.
# -----------------------------------------------------------------------------
def run_inference(
    image: Optional[Image.Image],
    threshold: float,
    resize_w: int,
    resize_h: int,
    include_viz: bool,
) -> Tuple:
    if image is None:
        return "Upload or select an image first.", None, None, None

    resize = (int(resize_w), int(resize_h))

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    params = {
        "include_visualizations": bool(include_viz),
        "threshold": float(threshold),
        "resize_width": resize[0],
        "resize_height": resize[1],
    }

    t0 = time.perf_counter()
    try:
        resp = _session.post(
            f"{API_URL}/predict",
            params=params,
            files={"file": ("image.png", buf, "image/png")},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return f"Cannot reach API at {API_URL}. Is it running?", None, None, None
    except requests.exceptions.Timeout:
        return f"API request timed out after {REQUEST_TIMEOUT:.0f}s.", None, None, None
    except requests.exceptions.HTTPError:
        detail = ""
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return f"API error ({resp.status_code}): {detail}", None, None, None
    except Exception as e:
        return f"Request failed: {e}", None, None, None
    round_trip_ms = (time.perf_counter() - t0) * 1000

    label = "ANOMALY DETECTED" if data["is_anomaly"] else "NORMAL"
    status = (
        f"{label}  |  Score: {data['anomaly_score']:.4f}  "
        f"|  Threshold: {threshold:.2f}  |  "
        f"inference {data['latency_ms']:.1f} ms  |  round-trip {round_trip_ms:.1f} ms"
    )

    original_pil = image.resize(resize, Image.BILINEAR)
    heatmap_pil = _b64_to_pil(data.get("heatmap_image_base64")) if include_viz else None
    boundary_pil = (
        _b64_to_pil(data.get("boundary_image_base64")) if include_viz else None
    )

    return status, original_pil, heatmap_pil, boundary_pil


# -----------------------------------------------------------------------------
# Gradio Blocks UI
# -----------------------------------------------------------------------------
_ACCENT = "#6366f1"
_ACCENT2 = "#ef4444"
_ACCENT3 = "#22c55e"
_BG = "#f5f6fa"
_SURFACE = "#ffffff"
_BORDER = "#e2e4f0"
_TEXT = "#1e1b4b"
_MUTED = "#7c82a8"

custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
:root {{ color-scheme: light; }}

body, .gradio-container {{
    background: {_BG} !important;
    color: {_TEXT} !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}

.app-header {{
    padding: 1.8rem 2.5rem 1.4rem;
    background: linear-gradient(135deg, #ffffff 0%, #eef0ff 50%, #f5f6ff 100%);
    border-bottom: 1px solid {_BORDER};
}}

.app-header h1 {{
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 0.3rem !important;
    color: {_TEXT} !important;
}}

.app-header .sub {{
    color: {_MUTED};
    font-size: 0.9rem;
    font-weight: 500;
}}

.status-normal  {{ color: {_ACCENT3}; font-weight: 700; }}
.status-anomaly {{ color: {_ACCENT2}; font-weight: 700; }}

.panel-label {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {_MUTED};
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

.dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {_ACCENT};
    display: inline-block;
}}

/* Result textbox */
.result-box textarea {{
    font-family: 'Plus Jakarta Sans', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}}

/* Primary button */
.btn-analyze {{
    background: linear-gradient(135deg, {_ACCENT}, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    letter-spacing: 0.02em !important;
}}

/* Sample gallery */
.sample-gallery-wrap .grid-wrap {{ border-radius: 12px; overflow: hidden; }}
"""

with gr.Blocks(
    css=custom_css,
    title="AnomaVision",
) as demo:

    # Header
    gr.HTML("""
        <div class="app-header">
          <h1>AnomaVision</h1>
          <div class="sub">Visual anomaly detection — PaDiM · ONNX · Zero-copy inference</div>
        </div>
        """)

    with gr.Tab("Inspect"):
        with gr.Row():

            # Left column — controls
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-label"><span class="dot"></span>Input</div>')

                input_img = gr.Image(label="", type="pil", height=280)

                threshold = gr.Slider(
                    0.1, 50.0, THRESHOLD_DEFAULT, step=0.1, label="Anomaly threshold"
                )

                with gr.Row():
                    resize_w = gr.Number(
                        value=224, label="Width", minimum=32, maximum=2048, precision=0
                    )
                    resize_h = gr.Number(
                        value=224, label="Height", minimum=32, maximum=2048, precision=0
                    )

                viz_check = gr.Checkbox(value=True, label="Generate visualizations")

                analyze_btn = gr.Button(
                    "Analyze image",
                    elem_classes=["btn-analyze"],
                    variant="primary",
                )

                # Sample gallery
                _gallery_items = _sample_gallery_items()
                if _gallery_items:
                    gr.HTML(
                        '<div class="panel-label" style="margin-top:1rem;"><span class="dot"></span>Samples</div>'
                    )
                    sample_gallery = gr.Gallery(
                        value=_gallery_items,
                        show_label=False,
                        columns=3,
                        rows=3,
                        height=240,
                        object_fit="cover",
                        allow_preview=False,
                        elem_classes=["sample-gallery-wrap"],
                    )
                else:
                    sample_gallery = None

            # Right column — results
            with gr.Column(scale=2):
                gr.HTML(
                    '<div class="panel-label"><span class="dot"></span>Results</div>'
                )

                result_text = gr.Textbox(
                    label="",
                    lines=2,
                    show_label=False,
                    elem_classes=["result-box"],
                    placeholder="Run inference to see results…",
                )

                with gr.Row():
                    out_original = gr.Image(label="Original", type="pil")
                    out_heatmap = gr.Image(label="Anomaly heatmap", type="pil")
                    out_boundary = gr.Image(label="Boundary", type="pil")

        # Event wiring
        analyze_btn.click(
            fn=run_inference,
            inputs=[input_img, threshold, resize_w, resize_h, viz_check],
            outputs=[result_text, out_original, out_heatmap, out_boundary],
        )

        if sample_gallery is not None:

            def _on_sample_select(evt: gr.SelectData) -> Optional[Image.Image]:
                if evt.index >= len(SAMPLES):
                    return None
                return _load_sample(SAMPLES[evt.index][1])

            sample_gallery.select(fn=_on_sample_select, outputs=[input_img])

    with gr.Tab("Draw defects"):
        with gr.Row():
            with gr.Column():
                sketch_img = gr.ImageEditor(
                    type="pil",
                    label="Draw defects here",
                    brush=gr.Brush(
                        colors=["#ff0000", "#ffff00", "#ffffff"], default_size=8
                    ),
                )
                sketch_threshold = gr.Slider(
                    0.1, 50.0, THRESHOLD_DEFAULT, step=0.1, label="Threshold"
                )
                sketch_btn = gr.Button("Analyze drawn image", variant="primary")

            with gr.Column():
                sketch_result = gr.Textbox(label="Result", lines=2)
                sketch_heatmap = gr.Image(label="Heatmap", type="pil")
                sketch_boundary = gr.Image(label="Boundary", type="pil")

        def _run_sketch(editor_val, thr):
            if editor_val is None:
                return "Draw on the image first.", None, None
            img = (
                editor_val.get("composite")
                if isinstance(editor_val, dict)
                else editor_val
            )
            if img is None:
                return "Draw on the image first.", None, None
            status, _, heat, boundary = run_inference(img, thr, 224, 224, True)
            return status, heat, boundary

        sketch_btn.click(
            fn=_run_sketch,
            inputs=[sketch_img, sketch_threshold],
            outputs=[sketch_result, sketch_heatmap, sketch_boundary],
        )

    with gr.Tab("API info"):
        gr.Markdown(f"""
### This UI talks to the API at: `{API_URL}`

Override with the `ANOMAVISION_API_URL` env var (e.g. a Kubernetes
Service DNS name like `http://anomavision-api:8000`).

### REST endpoints (separate process/container)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/model-info` | Session metadata, input/output shapes |
| POST | `/config` | Update the default threshold |
| POST | `/predict` | Inference — threshold/resize overridable per-request |
| GET | `/docs` | OpenAPI docs |

### Quick start

```bash
# Health
curl {API_URL}/health

# Predict
curl -X POST "{API_URL}/predict?include_visualizations=true&threshold=13.0" \\
     -F "file=@image.jpg"
```

### Architecture note

The UI and the API are now fully independent processes — they can run in
separate containers, scale separately, and the UI never imports onnxruntime
or the inference engine. Every analysis is one HTTP round-trip: this UI
encodes the image to PNG, POSTs it to `/predict`, and decodes the base64
PNGs that come back in the JSON response.
            """)


# -----------------------------------------------------------------------------
# Standalone entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[ui] API target: {API_URL}")
    try:
        r = _session.get(f"{API_URL}/health", timeout=5)
        print(f"[ui] API health check: {r.json()}")
    except Exception as e:
        print(
            f"[ui] WARNING: API not reachable at startup ({e}). "
            f"The UI will still launch — calls will fail until the API is up."
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7800,
        share=False,
        show_error=True,
    )
