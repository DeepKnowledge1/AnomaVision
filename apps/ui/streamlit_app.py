import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path
from typing import List, Optional

import requests
import streamlit as st
from PIL import Image

# streamlit run apps/ui/streamlit_app.py -- --port 8080


# -----------------------------
# CLI args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=8000, help="Port of the FastAPI server"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host of the FastAPI server"
    )
    return parser.parse_known_args()[0]


args = parse_args()

FASTAPI_URL = f"http://{args.host}:{args.port}"
PREDICT_ENDPOINT = f"{FASTAPI_URL}/predict"
HEALTH_ENDPOINT = f"{FASTAPI_URL}/health"
CONFIG_ENDPOINT = f"{FASTAPI_URL}/config"
MODEL_INFO_ENDPOINT = f"{FASTAPI_URL}/model-info"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# -----------------------------
# Helpers
# -----------------------------
def safe_get(url: str, timeout: int = 5):
    try:
        return requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException:
        return None


def safe_post(
    url: str, *, json_payload=None, files=None, params=None, timeout: int = 30
):
    try:
        return requests.post(
            url, json=json_payload, files=files, params=params, timeout=timeout
        )
    except requests.exceptions.RequestException:
        return None


def decode_base64_image(base64_str: str):
    if not base64_str:
        return None
    try:
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes))
    except Exception:
        return None


def percent_diff(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_image_files_from_folder(uploaded_files) -> List:
    """Extract and sort image files from uploaded folder."""
    if not uploaded_files:
        return []

    image_files = []
    for file in uploaded_files:
        ext = Path(file.name).suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            image_files.append(file)

    # Sort by filename
    image_files.sort(key=lambda x: x.name)
    return image_files


# -----------------------------
# Initialize session state
# -----------------------------
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "image_files" not in st.session_state:
    st.session_state.image_files = []
if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AnomaVision – Visual Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Ultra-fancy CSS
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Subtle animated background */
.stApp {
  background:
    radial-gradient(1200px 800px at 10% 10%, rgba(102,126,234,0.20), transparent 60%),
    radial-gradient(1000px 700px at 90% 20%, rgba(118,75,162,0.18), transparent 55%),
    radial-gradient(900px 650px at 50% 90%, rgba(72,187,120,0.12), transparent 55%),
    linear-gradient(180deg, rgba(255,255,255,0.92), rgba(250,250,252,0.92));
}

/* Hero */
.hero {
  padding: 1.25rem 1.35rem;
  border-radius: 20px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(255,255,255,0.55);
  backdrop-filter: blur(10px);
  box-shadow: 0 20px 60px rgba(0,0,0,0.08);
  position: relative;
  overflow: hidden;
}
.hero:before {
  content:"";
  position:absolute;
  inset:-2px;
  background: linear-gradient(135deg, rgba(102,126,234,0.55), rgba(118,75,162,0.50), rgba(72,187,120,0.45));
  filter: blur(20px);
  opacity: 0.35;
  z-index:0;
}
.hero-inner { position: relative; z-index: 1; }

.hero-title {
  margin: 0;
  font-size: 2.65rem;
  font-weight: 800;
  letter-spacing: -1.2px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 45%, #48bb78 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
}
.hero-sub {
  margin: 0.45rem 0 0;
  color: rgba(0,0,0,0.62);
  font-size: 1.05rem;
  font-weight: 500;
}
.badges { margin-top: 0.8rem; display:flex; gap:0.55rem; flex-wrap: wrap; }
.badge {
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  color: rgba(0,0,0,0.68);
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(255,255,255,0.70);
  backdrop-filter: blur(8px);
}

/* Glass cards */
.glass {
  border-radius: 18px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(255,255,255,0.55);
  backdrop-filter: blur(10px);
  box-shadow: 0 16px 45px rgba(0,0,0,0.07);
  padding: 1.1rem 1.1rem;
}
.card-title {
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: -0.2px;
  margin-bottom: 0.35rem;
}
.card-sub {
  color: rgba(0,0,0,0.60);
  font-size: 0.95rem;
  margin-top: -0.05rem;
}

/* Status badges */
.status-pill {
  display:inline-flex;
  align-items:center;
  gap: 0.45rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
  color: white;
}
.ok { background: linear-gradient(135deg, #48bb78, #38a169); }
.warn { background: linear-gradient(135deg, #f6ad55, #ed8936); }
.bad { background: linear-gradient(135deg, #fc8181, #f56565); }

/* Result panels */
.result-ok {
  border-left: 6px solid rgba(72,187,120,0.95);
  background: linear-gradient(135deg, rgba(240,255,244,0.92), rgba(198,246,213,0.78));
}
.result-bad {
  border-left: 6px solid rgba(245,101,101,0.95);
  background: linear-gradient(135deg, rgba(255,245,245,0.92), rgba(254,215,215,0.80));
  animation: glow 2.4s ease-in-out infinite;
}
@keyframes glow {
  0%,100% { box-shadow: 0 16px 45px rgba(245,101,101,0.10); }
  50% { box-shadow: 0 16px 55px rgba(245,101,101,0.22); }
}

/* Navigation buttons */
.nav-container {
  display: flex;
  gap: 0.8rem;
  margin: 1rem 0;
  align-items: center;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 0.65rem 1.1rem;
  font-weight: 750;
  transition: all 0.25s;
  box-shadow: 0 10px 22px rgba(102,126,234,0.30);
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 26px rgba(102,126,234,0.36);
}

/* Images */
.stImage > div {
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 18px;
  padding: 10px;
  background: rgba(255,255,255,0.72);
  box-shadow: 0 20px 55px rgba(0,0,0,0.08);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(247,250,252,0.95) 0%, rgba(237,242,247,0.95) 100%);
}
section[data-testid="stSidebar"] > div { padding-top: 1.4rem; }

/* Footer */
.footer {
  margin-top: 2rem;
  padding: 1.6rem;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(102,126,234,0.92), rgba(118,75,162,0.92));
  color: white;
  text-align: center;
  box-shadow: 0 18px 55px rgba(102,126,234,0.24);
}
.footer p { margin: 0.25rem 0; color: rgba(255,255,255,0.92); }

/* Progress indicator */
.progress-info {
  background: rgba(102,126,234,0.12);
  border-radius: 12px;
  padding: 0.6rem 1rem;
  font-weight: 600;
  color: rgba(0,0,0,0.75);
  text-align: center;
  margin: 0.8rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Hero header
# -----------------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-inner">
    <div class="hero-title">🔍 AnomaVision</div>
    <div class="hero-sub">Industrial-grade visual anomaly detection with explainable heatmaps and fast inference.</div>
    <div class="badges">
      <div class="badge">⚡ FastAPI Backend</div>
      <div class="badge">🖥️ Streamlit Demo</div>
      <div class="badge">🧠 PaDiM-style Features</div>
      <div class="badge">🎯 Threshold Control</div>
      <div class="badge">🗺️ Explainability</div>
      <div class="badge">📁 Batch Processing</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------
# Sidebar: Backend + config
# -----------------------------
with st.sidebar:
    st.header("⚙️ Inference Control")

    health_response = safe_get(HEALTH_ENDPOINT, timeout=5)
    api_online = False
    model_type = "unknown"
    current_threshold = 13.0
    status_label = "Offline"

    if health_response and health_response.status_code == 200:
        try:
            health_data = health_response.json()
            status = str(health_data.get("status", "unknown")).lower()
            model_type = str(health_data.get("model_type", "unknown"))
            current_threshold = float(health_data.get("threshold", current_threshold))

            if status == "healthy":
                api_online = True
                status_label = "Online"
                st.markdown(
                    '<span class="status-pill ok">✅ Backend Online</span>',
                    unsafe_allow_html=True,
                )
            else:
                status_label = "Degraded"
                st.markdown(
                    '<span class="status-pill warn">⚠️ Backend Degraded</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"**Model:** {model_type.upper()}")
            st.caption(f"Endpoint: {FASTAPI_URL}")
        except Exception:
            st.markdown(
                '<span class="status-pill bad">❌ Invalid Health Response</span>',
                unsafe_allow_html=True,
            )
            st.caption("Backend responded but JSON parsing failed.")
    else:
        st.markdown(
            '<span class="status-pill bad">❌ Backend Offline</span>',
            unsafe_allow_html=True,
        )
        st.caption("Start FastAPI, then refresh this page.")

    st.divider()

    st.subheader("🎨 Explainability")
    include_visualizations = st.checkbox(
        "Enable heatmap & boundary",
        value=True,
        help="If enabled, backend may return base64 heatmap/boundary images.",
    )

    st.subheader("🎯 Detection Threshold")
    new_threshold = st.slider(
        "Anomaly threshold",
        min_value=0.1,
        max_value=50.0,
        value=float(current_threshold),
        step=0.1,
        help="Higher = stricter (fewer anomalies). Lower = more sensitive.",
    )
    st.caption("Higher → stricter. Lower → more sensitive.")

    st.subheader("🔧 Preprocessing")
    c1, c2 = st.columns(2)
    with c1:
        resize_width = st.number_input(
            "Width", min_value=32, max_value=2048, value=900, step=8
        )
    with c2:
        resize_height = st.number_input(
            "Height", min_value=32, max_value=2048, value=900, step=8
        )

    if st.button(
        "💾 Apply Configuration", use_container_width=True, disabled=not api_online
    ):
        resp = safe_post(
            CONFIG_ENDPOINT,
            json_payload={
                "threshold": float(new_threshold),
                "resize_width": int(resize_width),
                "resize_height": int(resize_height),
            },
            timeout=5,
        )
        if resp and resp.status_code == 200:
            st.success("✅ Updated.")
            # Clear cache when config changes
            st.session_state.results_cache = {}
            st.rerun()
        else:
            st.error("❌ Failed.")
            if resp is not None:
                st.caption(f"HTTP {resp.status_code}: {resp.text}")
            else:
                st.caption("No response from backend.")

    st.divider()

    # Batch processing controls
    if len(st.session_state.image_files) > 0:
        st.subheader("📊 Batch Stats")
        total_images = len(st.session_state.image_files)
        processed = len(st.session_state.results_cache)
        anomalies = sum(
            1
            for r in st.session_state.results_cache.values()
            if r.get("is_anomaly", False)
        )

        st.metric("Total Images", total_images)
        st.metric("Processed", f"{processed}/{total_images}")
        st.metric("Anomalies Found", anomalies)

        if st.button("🔄 Clear Cache", use_container_width=True):
            st.session_state.results_cache = {}
            st.rerun()

    with st.expander("📌 Quick Start (API)"):
        st.code(
            f"""# Health
curl -s {HEALTH_ENDPOINT}

# Predict
curl -s -X POST "{PREDICT_ENDPOINT}?include_visualizations=true" \\
  -F "file=@/path/to/image.jpg"

# Update config
curl -s -X POST "{CONFIG_ENDPOINT}" \\
  -H "Content-Type: application/json" \\
  -d '{{"threshold": {new_threshold:.1f}, "resize_width": {int(resize_width)}, "resize_height": {int(resize_height)}}}'
""",
            language="bash",
        )

    with st.expander("📊 Model Info"):
        if st.button(
            "Fetch model details", use_container_width=True, disabled=not api_online
        ):
            r = safe_get(MODEL_INFO_ENDPOINT, timeout=6)
            if r and r.status_code == 200:
                st.json(r.json())
            else:
                st.error("Failed to fetch model info.")

    st.divider()
    st.markdown(
        """
<div class="glass" style="padding:0.9rem 1rem;">
  <div class="card-title">🧠 How it works</div>
  <div class="card-sub">
    Learns "normal" patterns, scores deviations, and visualizes contributing regions.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# -----------------------------
# Main layout
# -----------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Upload</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-sub">Upload single image or multiple images from a folder.</div>',
        unsafe_allow_html=True,
    )

    # File uploader supporting multiple files
    uploaded_files = st.file_uploader(
        "Upload image(s)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
        help="Supported: JPG, PNG, BMP, TIFF. You can select multiple files.",
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Process uploaded files
    if uploaded_files:
        new_files = get_image_files_from_folder(uploaded_files)
        if new_files:
            # Check if files changed
            new_filenames = [f.name for f in new_files]
            old_filenames = [f.name for f in st.session_state.image_files]

            if new_filenames != old_filenames:
                st.session_state.image_files = new_files
                st.session_state.current_index = 0
                # Don't clear cache - keep results for files that haven't changed

    # Navigation controls
    if st.session_state.image_files:
        total = len(st.session_state.image_files)
        current_idx = st.session_state.current_index

        st.markdown(
            f'<div class="progress-info">Image {current_idx + 1} of {total}</div>',
            unsafe_allow_html=True,
        )

        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 1])

        with nav_col1:
            if st.button(
                "⏮️ First", use_container_width=True, disabled=(current_idx == 0)
            ):
                st.session_state.current_index = 0
                st.rerun()

        with nav_col2:
            if st.button(
                "◀️ Previous", use_container_width=True, disabled=(current_idx == 0)
            ):
                st.session_state.current_index = max(0, current_idx - 1)
                st.rerun()

        with nav_col3:
            if st.button(
                "Next ▶️", use_container_width=True, disabled=(current_idx >= total - 1)
            ):
                st.session_state.current_index = min(total - 1, current_idx + 1)
                st.rerun()

        with nav_col4:
            if st.button(
                "Last ⏭️", use_container_width=True, disabled=(current_idx >= total - 1)
            ):
                st.session_state.current_index = total - 1
                st.rerun()

        # Jump to specific image
        jump_to = st.number_input(
            "Jump to image #",
            min_value=1,
            max_value=total,
            value=current_idx + 1,
            step=1,
            key="jump_to_input",
        )
        if st.button("Go", use_container_width=True):
            st.session_state.current_index = jump_to - 1
            st.rerun()

        # Display current image
        current_file = st.session_state.image_files[current_idx]
        st.image(
            current_file, caption=f"🔸 {current_file.name}", use_container_width=True
        )

    elif uploaded_files is None or len(uploaded_files) == 0:
        st.write("")
        st.markdown(
            """
<div class="glass" style="text-align:center;">
  <div style="font-size:1.2rem; font-weight:800; margin-bottom:0.35rem;">✨ Ready when you are</div>
  <div style="color:rgba(0,0,0,0.62);">
    Upload one or more images to instantly generate anomaly scores and visual heatmaps.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


# -----------------------------
# Auto-inference with caching
# -----------------------------
def run_inference_on_current():
    """Run inference on the current image with intelligent caching."""
    if not st.session_state.image_files or not api_online:
        return

    current_idx = st.session_state.current_index
    current_file = st.session_state.image_files[current_idx]

    file_bytes = current_file.getvalue()
    file_hash = sha256_bytes(file_bytes)

    # Create cache key
    cache_key = f"{file_hash}_{new_threshold}_{resize_width}_{resize_height}_{include_visualizations}"

    # Check cache
    if cache_key in st.session_state.results_cache:
        return

    # Run inference
    with st.spinner(f"🔮 Analyzing {current_file.name}..."):
        files = {
            "file": (current_file.name, file_bytes, current_file.type or "image/jpeg")
        }
        params = {"include_visualizations": include_visualizations}

        start_time = time.time()
        resp = safe_post(PREDICT_ENDPOINT, files=files, params=params, timeout=60)
        elapsed = time.time() - start_time

        if resp and resp.status_code == 200:
            try:
                result = resp.json()
                result["_elapsed"] = elapsed
                result["_filename"] = current_file.name
                st.session_state.results_cache[cache_key] = result
            except Exception as e:
                st.session_state.results_cache[cache_key] = {
                    "_error": f"Invalid JSON response: {str(e)}",
                    "_filename": current_file.name,
                }
        else:
            error_msg = (
                "Connection error" if resp is None else f"HTTP {resp.status_code}"
            )
            st.session_state.results_cache[cache_key] = {
                "_error": error_msg,
                "_filename": current_file.name,
            }


# Run inference on current image
if st.session_state.image_files and api_online:
    run_inference_on_current()


# -----------------------------
# Results panel
# -----------------------------
with right_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-sub">Anomaly score, verdict, latency, and explainability.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if not st.session_state.image_files:
        st.info("Upload images to see results.")
        st.markdown("</div>", unsafe_allow_html=True)
    elif not api_online:
        st.error("Backend is offline – start FastAPI then refresh.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        current_idx = st.session_state.current_index
        current_file = st.session_state.image_files[current_idx]
        file_bytes = current_file.getvalue()
        file_hash = sha256_bytes(file_bytes)
        cache_key = f"{file_hash}_{new_threshold}_{resize_width}_{resize_height}_{include_visualizations}"

        result = st.session_state.results_cache.get(cache_key)

        if result is None:
            st.info("Processing image...")
            st.markdown("</div>", unsafe_allow_html=True)
        elif "_error" in result:
            st.error(f"Inference failed: {result['_error']}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            analysis_time = float(result.get("_elapsed", 0.0))
            anomaly_score = float(result.get("anomaly_score", 0.0))
            is_anomaly = bool(result.get("is_anomaly", False))
            filename = result.get("_filename", current_file.name)

            m1, m2, m3 = st.columns(3)
            m1.metric("🎯 Score", f"{anomaly_score:.3f}")
            m2.metric("🔌 Verdict", "🚨 ANOMALY" if is_anomaly else "✅ NORMAL")
            m3.metric("⚡ Latency", f"{analysis_time:.2f}s")

            st.write("")

            if is_anomaly:
                sev = percent_diff(anomaly_score, float(new_threshold))
                st.markdown(
                    f"""
<div class="glass result-bad">
  <div style="font-size:1.15rem; font-weight:900;">🚨 Anomaly Detected</div>
  <div style="color:rgba(0,0,0,0.66); margin-top:0.2rem;">
    <b>{filename}</b> scored <b>{anomaly_score:.4f}</b> (≥ <b>{float(new_threshold):.4f}</b>).
    Severity: <b>{sev:.1f}%</b> above threshold.
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                margin = abs(percent_diff(anomaly_score, float(new_threshold)))
                st.markdown(
                    f"""
<div class="glass result-ok">
  <div style="font-size:1.15rem; font-weight:900;">✅ Normal Image</div>
  <div style="color:rgba(0,0,0,0.66); margin-top:0.2rem;">
    <b>{filename}</b> scored <b>{anomaly_score:.4f}</b> (&lt; <b>{float(new_threshold):.4f}</b>).
    Margin: <b>{margin:.1f}%</b> below threshold.
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.write("")
            tabs = st.tabs(["🎨 Explainability", "🧾 Raw JSON", "⬇️ Export"])

            with tabs[0]:
                if not include_visualizations:
                    st.info("Explainability is disabled in the sidebar.")
                else:
                    c1, c2 = st.columns(2)
                    heatmap = decode_base64_image(
                        result.get("heatmap_image_base64", "")
                    )
                    boundary = decode_base64_image(
                        result.get("boundary_image_base64", "")
                    )

                    with c1:
                        if heatmap:
                            st.image(
                                heatmap,
                                caption="🌡️ Heatmap Overlay",
                                use_container_width=True,
                            )
                        else:
                            st.warning("No heatmap returned.")

                    with c2:
                        if boundary:
                            st.image(
                                boundary,
                                caption="🎯 Boundary Visualization",
                                use_container_width=True,
                            )
                        else:
                            st.warning("No boundary visualization returned.")

            with tabs[1]:
                # Remove internal keys before displaying
                display_result = {
                    k: v for k, v in result.items() if not k.startswith("_")
                }
                st.json(display_result)

            with tabs[2]:
                display_result = {
                    k: v for k, v in result.items() if not k.startswith("_")
                }
                st.download_button(
                    "⬇️ Download result JSON",
                    data=json.dumps(display_result, indent=2).encode("utf-8"),
                    file_name=f"anomavision_{filename}.json",
                    mime="application/json",
                    use_container_width=True,
                )

                # Export all results
                if len(st.session_state.results_cache) > 1:
                    all_results = []
                    for k, v in st.session_state.results_cache.items():
                        clean_result = {
                            key: val
                            for key, val in v.items()
                            if not key.startswith("_")
                        }
                        clean_result["filename"] = v.get("_filename", "unknown")
                        all_results.append(clean_result)

                    st.download_button(
                        "⬇️ Download all results (JSON)",
                        data=json.dumps(all_results, indent=2).encode("utf-8"),
                        file_name="anomavision_batch_results.json",
                        mime="application/json",
                        use_container_width=True,
                    )
            st.markdown(
                """
            <div class="footer">
            <div style="font-weight:900; font-size:1.15rem;">AnomaVision — Visual Anomaly Detection</div>
            <p>FastAPI • Streamlit • Explainable Inference • Industrial Inspection Ready</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
