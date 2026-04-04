"""
MVTec Anomaly Detection Demo — Gradio app (no FastAPI required).

Run:
    python gradio_app.py

Environment variables (all optional):
    ANOMAVISION_MODEL_DATA_PATH   path that contains the model file
                                  (default: "distributions/anomav_exp")
    ANOMAVISION_MODEL_FILE        model filename
                                  (default: "model.onnx")
    ANOMAVISION_DEVICE            "auto" | "cpu" | "cuda"   (default: "auto")
    ANOMAVISION_THRESHOLD         float anomaly threshold    (default: 13.0)
    ANOMAVISION_VIZ_PADDING       int, boundary-frame padding (default: 40)
    ANOMAVISION_VIZ_ALPHA         float, heatmap blend alpha  (default: 0.5)
    ANOMAVISION_VIZ_COLOR         R,G,B  highlight color      (default: 128,0,128)
    SAMPLE_IMAGES_DIR             directory with sample images (default: "sample_images")
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image

# ── lazy import so the app still starts even if anomavision isn't installed ──
try:
    import anomavision
    from anomavision.general import determine_device
    from anomavision.inference.model.wrapper import ModelWrapper
    from anomavision.inference.modelType import ModelType

    ANOMAVISION_AVAILABLE = True
except ImportError:
    ANOMAVISION_AVAILABLE = False
    print("WARNING: anomavision not found – running in DEMO mode (random scores).")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DATA_PATH = os.getenv(
    "ANOMAVISION_MODEL_DATA_PATH", "distributions/padim/bottle/anomav_exp"
)
MODEL_FILE = os.getenv("ANOMAVISION_MODEL_FILE", "model.onnx")
DEVICE_ENV = os.getenv("ANOMAVISION_DEVICE", "auto")
THRESHOLD_DEFAULT = float(os.getenv("ANOMAVISION_THRESHOLD", "13.0"))
VIZ_PADDING = int(os.getenv("ANOMAVISION_VIZ_PADDING", "40"))
VIZ_ALPHA = float(os.getenv("ANOMAVISION_VIZ_ALPHA", "0.5"))
VIZ_COLOR = tuple(map(int, os.getenv("ANOMAVISION_VIZ_COLOR", "128,0,128").split(",")))
SAMPLE_DIR = os.getenv("SAMPLE_IMAGES_DIR", "D:/01-DATA/sample_images")

# ─────────────────────────────────────────────────────────────────────────────
# Model — loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────
_model: Optional["ModelWrapper"] = None
_model_type = None
_device_str: str = "cpu"


def _load_model() -> str:
    """Load the model and return a status message."""
    global _model, _model_type, _device_str

    if not ANOMAVISION_AVAILABLE:
        return "⚠️  anomavision not installed — running in demo mode."

    model_path = os.path.realpath(os.path.join(MODEL_DATA_PATH, MODEL_FILE))
    if not os.path.exists(model_path):
        return f"⚠️  Model not found at {model_path} — running in demo mode."

    try:
        _device_str = determine_device(DEVICE_ENV)
        _model_type = ModelType.from_extension(model_path)
        _model = ModelWrapper(model_path, _device_str)

        # Optional warmup
        try:
            dummy = torch.zeros(
                (1, 3, 224, 224), dtype=torch.float32, device=_device_str
            )
            _model.warmup(batch=dummy, runs=1)
        except Exception:
            pass

        return f"✅  Model loaded: {Path(model_path).name} ({_model_type.value}) on {_device_str}"
    except Exception as e:
        return f"⚠️  Model load failed: {e} — running in demo mode."


_startup_message = _load_model()
print(_startup_message)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _demo_predict(image_np: np.ndarray):
    """Return fake results when the real model isn't available."""
    h, w = image_np.shape[:2]
    score = float(np.random.uniform(5, 25))
    heatmap_np = np.random.rand(h, w).astype(np.float32)
    return score, heatmap_np


def _real_predict(image_np: np.ndarray, threshold: float):
    """Run anomavision inference and return (score, score_map_np, boundary_np, heatmap_np, highlighted_np)."""
    device = torch.device(_device_str)
    batch = anomavision.to_batch(
        [image_np], anomavision.standard_image_transform, device
    )

    if _device_str == "cuda":
        batch = batch.half()

    with torch.no_grad():
        image_scores, score_maps = _model.predict(batch)

    score_map_cls = anomavision.classification(score_maps, threshold)
    image_cls = anomavision.classification(image_scores, threshold)

    test_images = np.array([image_np])

    boundary_images = anomavision.visualization.framed_boundary_images(
        test_images, score_map_cls, image_cls, padding=VIZ_PADDING
    )
    heatmap_images = anomavision.visualization.heatmap_images(
        test_images, score_maps, alpha=VIZ_ALPHA
    )
    highlighted_images = anomavision.visualization.highlighted_images(
        [image_np], score_map_cls, color=VIZ_COLOR
    )

    sm0 = score_maps[0]
    if isinstance(sm0, np.ndarray):
        score_map_np = sm0
    elif hasattr(sm0, "cpu"):
        score_map_np = sm0.cpu().float().numpy()
    else:
        score_map_np = np.array(sm0)

    return (
        float(image_scores[0]),
        score_map_np,
        boundary_images[0],
        heatmap_images[0],
        highlighted_images[0],
    )


def _np_to_pil(arr: np.ndarray, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    if arr is None:
        return None
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if size:
        img = img.resize(size, Image.BILINEAR)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Sample images
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_samples() -> list:
    """
    Collect sample images from SAMPLE_DIR.
    Expected layout (mirrors MVTec):
        sample_images/
            bottle/broken_large/000.png
            bottle/good/001.png
            cable/bent_wire/000.png
            …
    Falls back to any image recursively found in SAMPLE_DIR.
    Returns list of (display_label, abs_path).
    """
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


def _sample_gallery_images() -> list:
    """Return list of (path, label) tuples for gr.Gallery."""
    result = []
    for label, path in SAMPLES:
        if Path(path).exists():
            result.append((path, label))
    return result


def load_sample_image(path: str) -> Optional[Image.Image]:
    """Load a sample image from disk path."""
    if not path or not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main inference function (called by Gradio)
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    image: Optional[Image.Image],
    threshold: float,
    resize_w: int,
    resize_h: int,
    include_viz: bool,
) -> Tuple:
    if image is None:
        return "❌ Please upload or select an image.", None, None, None, None

    resize = (int(resize_w), int(resize_h))
    image_np = _pil_to_np(image)

    t0 = time.time()

    if _model is not None and ANOMAVISION_AVAILABLE:
        try:
            score, score_map_np, boundary_np, heatmap_np, highlighted_np = (
                _real_predict(image_np, threshold)
            )
            is_anomaly = score >= threshold

            original_pil = image.resize(resize, Image.BILINEAR)
            heatmap_pil = _np_to_pil(heatmap_np, resize) if include_viz else None
            boundary_pil = _np_to_pil(boundary_np, resize) if include_viz else None
            highlighted_pil = (
                _np_to_pil(highlighted_np, resize) if include_viz else None
            )

        except Exception as e:
            return f"⚠️  Inference error: {e}", None, None, None, None
    else:
        # Demo mode
        score, heatmap_raw = _demo_predict(image_np)
        is_anomaly = score >= threshold

        original_pil = image.resize(resize, Image.BILINEAR)

        if include_viz:
            import matplotlib.cm as cm

            heatmap_norm = heatmap_raw / heatmap_raw.max()
            cmap = cm.get_cmap("jet")
            heatmap_rgba = (cmap(heatmap_norm) * 255).astype(np.uint8)
            heatmap_rgb = heatmap_rgba[:, :, :3]
            blend = (0.5 * image_np + 0.5 * heatmap_rgb).astype(np.uint8)
            heatmap_pil = _np_to_pil(heatmap_rgb, resize)
            boundary_pil = _np_to_pil(blend, resize)
            highlighted_pil = _np_to_pil(image_np, resize)
        else:
            heatmap_pil = boundary_pil = highlighted_pil = None

    elapsed = time.time() - t0
    label = "🚨 ANOMALY DETECTED" if is_anomaly else "✅ NORMAL"
    status = f"Model: {Path(MODEL_FILE).stem} | Score: {score:.4f} | {label}"
    detail = f"Threshold: {threshold:.2f} | Inference time: {elapsed:.2f}s"

    return (
        f"{status}\n{detail}",
        original_pil,
        heatmap_pil,
        boundary_pil,
        highlighted_pil,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS — Clean Light Theme with Indigo/Violet Accents
# ─────────────────────────────────────────────────────────────────────────────
_ACCENT = "#6366f1"  # indigo
_ACCENT_H = "#4f46e5"  # indigo hover
_ACCENT2 = "#ef4444"  # red for anomaly alerts
_ACCENT3 = "#22c55e"  # green for normal result
_BG = "#f5f6fa"  # off-white page background
_SURFACE = "#ffffff"  # card surface
_SURFACE2 = "#f0f1f8"  # slightly tinted input background
_BORDER = "#e2e4f0"  # soft lavender border
_TEXT = "#1e1b4b"  # deep indigo text
_MUTED = "#7c82a8"  # muted blue-grey

custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── RESET & GLOBAL ─────────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}
:root {{ color-scheme: light; }}

body, .gradio-container {{
    background: {_BG} !important;
    color: {_TEXT} !important;
    font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif !important;
}}

/* ── HEADER ─────────────────────────────────────────────────────────────── */
.app-header {{
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 0;
    position: relative;
    background: linear-gradient(135deg, #ffffff 0%, #eef0ff 50%, #f5f6ff 100%);
    border-bottom: 1px solid {_BORDER};
    overflow: hidden;
}}

/* Decorative blurred orbs */
.app-header::before {{
    content: '';
    position: absolute;
    top: -40px; right: 80px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, {_ACCENT}22 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}}
.app-header::after {{
    content: '';
    position: absolute;
    bottom: -30px; right: 300px;
    width: 150px; height: 150px;
    background: radial-gradient(circle, #a5b4fc33 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}}

.app-header-inner {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1.5rem;
    position: relative;
    z-index: 1;
}}

.app-header-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {_ACCENT};
    background: {_ACCENT}12;
    border: 1px solid {_ACCENT}30;
    border-radius: 99px;
    padding: 0.22rem 0.75rem;
    margin-bottom: 0.7rem;
}}
.app-header-badge::before {{
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {_ACCENT};
    animation: blink 1.6s ease-in-out infinite;
}}

.app-header h1 {{
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 0.4rem !important;
    color: {_TEXT} !important;
    line-height: 1.15 !important;
}}
.app-header h1 .hl {{
    background: linear-gradient(90deg, {_ACCENT}, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.app-header p {{
    margin: 0.2rem 0 0 !important;
    color: {_MUTED} !important;
    font-size: 0.9rem !important;
    line-height: 1.55 !important;
}}

.app-header .startup-msg {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.71rem !important;
    color: {_ACCENT}bb !important;
    margin-top: 0.65rem !important;
    padding: 0.3rem 0.7rem;
    background: {_ACCENT}0d;
    border-left: 2px solid {_ACCENT}66;
    border-radius: 0 4px 4px 0;
    display: inline-block;
}}

/* Header stat chips */
.header-stats {{
    display: flex;
    gap: 0.8rem;
    flex-shrink: 0;
    margin-top: 0.5rem;
}}
.stat-chip {{
    text-align: center;
    padding: 0.55rem 1rem;
    background: {_SURFACE};
    border: 1px solid {_BORDER};
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(99,102,241,0.07);
    transition: box-shadow 0.2s, transform 0.2s;
}}
.stat-chip:hover {{
    box-shadow: 0 4px 16px rgba(99,102,241,0.14);
    transform: translateY(-1px);
}}
.stat-chip .val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: {_ACCENT};
    display: block;
    line-height: 1;
}}
.stat-chip .lbl {{
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {_MUTED};
    display: block;
    margin-top: 3px;
}}

/* ── STATUS PILL ─────────────────────────────────────────────────────────── */
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 0.18rem 0.7rem;
    border-radius: 99px;
    background: #dcfce7;
    color: #16a34a;
    border: 1px solid #86efac;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-left: 0.55rem;
    vertical-align: middle;
}}
.status-pill::before {{
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #16a34a;
    animation: blink 1.4s ease-in-out infinite;
}}

@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.25; }}
}}

/* ── TABS ────────────────────────────────────────────────────────────────── */
.tab-nav {{
    background: {_SURFACE} !important;
    border-bottom: 1px solid {_BORDER} !important;
    padding: 0 1.5rem !important;
}}
.tab-nav button {{
    background: transparent !important;
    color: {_MUTED} !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1.1rem !important;
    transition: color .18s, border-color .18s !important;
    margin-bottom: -1px !important;
}}
.tab-nav button:hover {{
    color: {_TEXT} !important;
}}
.tab-nav button.selected {{
    color: {_ACCENT} !important;
    border-bottom-color: {_ACCENT} !important;
}}

/* ── PANEL LABELS ────────────────────────────────────────────────────────── */
.panel-label {{
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: {_MUTED} !important;
    margin-bottom: 0.6rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
}}
.panel-label span.dot {{
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {_ACCENT};
    display: inline-block;
    flex-shrink: 0;
}}

/* ── RESULT TEXTBOX ──────────────────────────────────────────────────────── */
.result-header textarea, .result-header {{
    background: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    border-left: 3px solid {_ACCENT} !important;
    border-radius: 8px !important;
    padding: 0.85rem 1rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    white-space: pre-wrap !important;
    color: {_TEXT} !important;
    min-height: 3.8rem !important;
    line-height: 1.65 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}}

/* ── ANALYZE BUTTON ──────────────────────────────────────────────────────── */
.btn-analyze, .btn-analyze button {{
    background: linear-gradient(135deg, {_ACCENT}, {_ACCENT_H}) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.78rem 1rem !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px {_ACCENT}40 !important;
}}
.btn-analyze:hover, .btn-analyze button:hover {{
    box-shadow: 0 6px 22px {_ACCENT}55 !important;
    transform: translateY(-1px) !important;
    filter: brightness(1.06) !important;
}}
.btn-analyze:active, .btn-analyze button:active {{
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px {_ACCENT}33 !important;
}}

/* ── FORM CONTROLS ───────────────────────────────────────────────────────── */
input[type=range] {{
    accent-color: {_ACCENT} !important;
}}
.gr-input, .gr-box, .gr-form, textarea,
input[type=number], input[type=text],
.gr-textbox textarea, .gr-number input {{
    background: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    color: {_TEXT} !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}
.gr-input:focus, textarea:focus, input:focus {{
    border-color: {_ACCENT}88 !important;
    box-shadow: 0 0 0 3px {_ACCENT}15 !important;
    outline: none !important;
}}
label span, .gr-form label span {{
    color: {_MUTED} !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}}
select, .gr-dropdown {{
    background: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    color: {_TEXT} !important;
    border-radius: 8px !important;
}}

/* ── IMAGE PANELS ────────────────────────────────────────────────────────── */
.image-output-wrapper, .gr-image {{
    background: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}}
.image-output-wrapper:hover, .gr-image:hover {{
    border-color: {_ACCENT}66 !important;
    box-shadow: 0 4px 18px {_ACCENT}18 !important;
}}

/* ── SAMPLE GALLERY ──────────────────────────────────────────────────────── */
.sample-gallery-wrap .thumbnails {{
    gap: 6px !important;
    background: {_SURFACE2} !important;
    padding: 7px !important;
    border-radius: 10px !important;
    border: 1px solid {_BORDER} !important;
}}
.sample-gallery-wrap img {{
    border-radius: 7px !important;
    object-fit: cover !important;
    border: 1.5px solid {_BORDER} !important;
    transition: border-color 0.18s, transform 0.15s, box-shadow 0.18s !important;
}}
.sample-gallery-wrap img:hover {{
    border-color: {_ACCENT}88 !important;
    transform: scale(1.04) !important;
    box-shadow: 0 4px 12px {_ACCENT}22 !important;
}}

/* ── CHECKBOX ────────────────────────────────────────────────────────────── */
input[type=checkbox] {{
    accent-color: {_ACCENT} !important;
    width: 15px !important; height: 15px !important;
}}

/* ── SKETCH EDITOR ───────────────────────────────────────────────────────── */
.gr-image-editor {{
    background: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 10px !important;
}}

/* ── MARKDOWN ────────────────────────────────────────────────────────────── */
.gr-markdown h2 {{
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    color: {_TEXT} !important;
    margin-bottom: 0.8rem !important;
    letter-spacing: -0.02em !important;
}}
.gr-markdown p, .gr-markdown li {{
    color: {_MUTED} !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}}
.gr-markdown strong {{ color: {_TEXT} !important; }}
.gr-markdown code {{
    background: {_SURFACE2} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 4px !important;
    padding: 0.1em 0.4em !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em !important;
    color: {_ACCENT} !important;
}}

/* ── SCROLLBAR ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {_BG}; }}
::-webkit-scrollbar-thumb {{
    background: #c7cbdf;
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {_ACCENT}88; }}

/* ── MISC ────────────────────────────────────────────────────────────────── */
footer, .footer {{ display: none !important; }}

.gr-padded, .gr-panel, .gr-block {{
    background: transparent !important;
    border: none !important;
}}
.gr-box {{
    background: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}}
.gr-row {{ gap: 14px !important; }}

/* ── IMAGE GRID ──────────────────────────────────────────────────────────── */
.image-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}}
.img-card {{
    background: {_SURFACE};
    border: 1px solid {_BORDER};
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}}
.img-card-title {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {_MUTED};
    text-align: center;
    padding: 6px 8px 0;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio app
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="AnomaVision — Industrial Anomaly Detection") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="app-header">
      <div class="app-header-inner">
        <div>
          <div class="app-header-badge">AnomaVision &nbsp;·&nbsp; Industrial Inspection AI</div>
          <h1>
            <span class="hl">ANOMALY</span> DETECTION
            <span class="status-pill">ONLINE</span>
          </h1>
          <p>Upload an image or pick a sample — get the heatmap, overlay &amp; predicted mask in milliseconds.</p>
          <p class="startup-msg">{_startup_message}</p>
        </div>
        <div class="header-stats">
          <div class="stat-chip">
            <span class="val">PaDiM</span>
            <span class="lbl">Model</span>
          </div>
          <div class="stat-chip">
            <span class="val">15</span>
            <span class="lbl">Categories</span>
          </div>
          <div class="stat-chip">
            <span class="val">224²</span>
            <span class="lbl">Resolution</span>
          </div>
        </div>
      </div>
    </div>
    """)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Upload Image ───────────────────────────────────────────────
        with gr.Tab("📤 Upload Image"):

            with gr.Row(equal_height=False):

                # ── Left column: controls ────────────────────────────────────
                with gr.Column(scale=1, min_width=300):

                    gr.HTML(
                        '<div class="panel-label"><span class="dot"></span>Input</div>'
                    )

                    input_img = gr.Image(
                        type="pil",
                        label="Upload Image",
                        show_label=False,
                        height=280,
                    )

                    with gr.Row():
                        model_dd = gr.Dropdown(
                            choices=[Path(MODEL_FILE).stem],
                            value=Path(MODEL_FILE).stem,
                            label="Model",
                            scale=1,
                        )
                        category_dd = gr.Dropdown(
                            choices=[
                                "bottle",
                                "cable",
                                "carpet",
                                "grid",
                                "hazelnut",
                                "leather",
                                "metal_nut",
                                "pill",
                                "screw",
                                "tile",
                                "toothbrush",
                                "transistor",
                                "wood",
                                "zipper",
                                "other",
                            ],
                            value="bottle",
                            label="Category",
                            scale=1,
                        )

                    threshold = gr.Slider(
                        0.1, 50.0, THRESHOLD_DEFAULT, step=0.1, label="Threshold"
                    )

                    with gr.Row():
                        resize_w = gr.Number(
                            value=224,
                            label="Width",
                            minimum=32,
                            maximum=2048,
                            precision=0,
                        )
                        resize_h = gr.Number(
                            value=224,
                            label="Height",
                            minimum=32,
                            maximum=2048,
                            precision=0,
                        )

                    viz_check = gr.Checkbox(value=True, label="Generate Visualizations")

                    analyze_btn = gr.Button(
                        "🔍  Analyze Image",
                        elem_classes=["btn-analyze"],
                        variant="primary",
                    )

                    # ── Sample gallery (native gr.Gallery) ───────────────────
                    gr.HTML(
                        '<div class="panel-label" style="margin-top:1.2rem;">'
                        '<span class="dot"></span>Sample Images '
                        '<span style="font-weight:400;text-transform:none;letter-spacing:0;">'
                        "(click to select)</span></div>"
                    )

                    _gallery_items = _sample_gallery_images()

                    if _gallery_items:
                        sample_gallery = gr.Gallery(
                            value=_gallery_items,
                            label="",
                            show_label=False,
                            columns=3,
                            rows=3,
                            height=280,
                            object_fit="cover",
                            allow_preview=False,
                            elem_classes=["sample-gallery-wrap"],
                        )
                    else:
                        gr.HTML(
                            f"<div style='color:{_MUTED};padding:0.75rem;font-size:.85rem;"
                            f"background:{_SURFACE};border:1px solid {_BORDER};"
                            f"border-radius:8px;'>"
                            f"No sample images found in <code>{SAMPLE_DIR}</code>.<br>"
                            f"Place images there and restart.</div>"
                        )
                        sample_gallery = None

                # ── Right column: results ─────────────────────────────────────
                with gr.Column(scale=2):

                    gr.HTML(
                        '<div class="panel-label"><span class="dot"></span>Results</div>'
                    )

                    result_text = gr.Textbox(
                        label="",
                        lines=2,
                        show_label=False,
                        elem_classes=["result-header"],
                        placeholder="Run inference to see results…",
                    )

                    with gr.Row():
                        out_original = gr.Image(label="Original", type="pil")
                        out_heatmap = gr.Image(label="Anomaly Heatmap", type="pil")
                        out_overlay = gr.Image(label="Overlay", type="pil")
                        out_mask = gr.Image(label="Predicted Mask", type="pil")

            # ── Event wiring ─────────────────────────────────────────────────

            # Analyze button
            analyze_btn.click(
                fn=run_inference,
                inputs=[input_img, threshold, resize_w, resize_h, viz_check],
                outputs=[result_text, out_original, out_heatmap, out_overlay, out_mask],
            )

            # Sample gallery click → load image into input_img
            if sample_gallery is not None:

                def on_sample_select(evt: gr.SelectData) -> Image.Image:
                    """Load the clicked sample image into the input component."""
                    if evt.index >= len(SAMPLES):
                        return None
                    _label, path = SAMPLES[evt.index]
                    return load_sample_image(path)

                sample_gallery.select(
                    fn=on_sample_select,
                    inputs=None,
                    outputs=[input_img],
                )

        # ── Tab 2: Draw Defects ───────────────────────────────────────────────
        with gr.Tab("🎨 Draw Defects"):
            gr.HTML("""
            <div style="padding:1.2rem 0 0.4rem;">
              <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
                          color:#7c82a8;margin-bottom:0.5rem;">Synthetic Defect Testing</div>
              <div style="font-size:1.4rem;font-weight:800;color:#1e1b4b;letter-spacing:-0.02em;margin-bottom:0.6rem;">
                Draw Artificial Defects
              </div>
              <ol style="color:#7c82a8;font-size:0.88rem;line-height:2;padding-left:1.2rem;margin:0;">
                <li>Upload a <strong style="color:#1e1b4b;">GOOD (normal)</strong> reference image</li>
                <li>Use the brush tool to paint artificial defects anywhere</li>
                <li>Click Analyze — watch the model catch what you drew</li>
              </ol>
              <p style="font-size:0.75rem;color:#7c82a8;margin-top:0.5rem;font-style:italic;">
                ✦ Requires Gradio ≥ 4.x for the sketch editor
              </p>
            </div>
            """)
            with gr.Row():
                with gr.Column():
                    sketch_img = gr.ImageEditor(
                        type="pil",
                        label="Draw Defects Here",
                        brush=gr.Brush(
                            colors=["#ff0000", "#ffff00", "#ffffff"], default_size=8
                        ),
                    )
                    sketch_threshold = gr.Slider(
                        0.1, 50.0, THRESHOLD_DEFAULT, step=0.1, label="Threshold"
                    )
                    sketch_btn = gr.Button("🔍  Analyze Drawn Image", variant="primary")
                with gr.Column():
                    sketch_result = gr.Textbox(label="Result", lines=2)
                    sketch_heat = gr.Image(label="Heatmap", type="pil")
                    sketch_overlay = gr.Image(label="Overlay", type="pil")

            def run_sketch(editor_val, thr):
                if editor_val is None:
                    return "Please draw on the image first.", None, None
                img = (
                    editor_val.get("composite")
                    if isinstance(editor_val, dict)
                    else editor_val
                )
                if img is None:
                    return "Please draw on the image first.", None, None
                status, orig, heat, boundary, _ = run_inference(
                    img, thr, 224, 224, True
                )
                return status, heat, boundary

            sketch_btn.click(
                fn=run_sketch,
                inputs=[sketch_img, sketch_threshold],
                outputs=[sketch_result, sketch_heat, sketch_overlay],
            )

        # ── Tab 3: Compare Models ─────────────────────────────────────────────
        with gr.Tab("⚖️ Compare Models"):
            gr.HTML("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        padding:4rem 2rem;text-align:center;">
              <div style="font-size:2.8rem;margin-bottom:1rem;opacity:0.2;">⚖️</div>
              <div style="font-size:1.5rem;font-weight:800;color:#1e1b4b;letter-spacing:-0.02em;margin-bottom:0.5rem;">
                Side-by-Side Model Comparison
              </div>
              <div style="color:#7c82a8;font-size:0.88rem;max-width:380px;line-height:1.7;">
                Run two models simultaneously on the same image and compare their anomaly scores,
                heatmaps, and inference times.
              </div>
              <div style="margin-top:1.4rem;padding:0.4rem 1.1rem;background:#eef0ff;
                          border:1px solid #c7cbff;border-radius:99px;
                          font-size:0.72rem;font-weight:700;color:#6366f1;letter-spacing:0.1em;
                          text-transform:uppercase;">Coming Soon</div>
            </div>
            """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.violet,
            neutral_hue=gr.themes.colors.slate,
        ),
        css=custom_css,
    )
