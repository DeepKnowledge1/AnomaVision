"""Live, customer-facing dashboard for production data-drift monitoring.

The dashboard is read-only. It consumes the JSON status emitted by
``ProductionDriftMonitor`` and, when available, shows recent production input
images from the configured ``img_path``. It never changes anomaly inference.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

STATUS_FILE = Path(os.getenv("ANOMAVISION_DRIFT_STATUS_FILE", "./drift/drift_status.json"))
CONFIG_FILE = Path(os.getenv("ANOMAVISION_DRIFT_CONFIG", "./config.yml"))
REFRESH_SECONDS = max(1, int(os.getenv("ANOMAVISION_DRIFT_REFRESH_SECONDS", "2")))
SAMPLE_COUNT = 8

_BG = "#080b14"
_CARD = "#101522"
_CARD2 = "#151b2b"
_BORDER = "#263149"
_TEXT = "#f4f7ff"
_MUTED = "#8d98b2"
_PURPLE = "#8b6cff"
_RED = "#ff5c7a"
_GREEN = "#2bd576"
_AMBER = "#ffbd59"

_DEFAULT: dict[str, Any] = {
    "ready": False,
    "status": "unavailable",
    "samples_seen": 0,
    "window_size": 500,
    "window_fill": 0,
    "drift_score": None,
    "psi": None,
    "mean_shift": None,
    "std_shift": None,
    "cosine_shift": None,
    "threshold": 0.2,
    "warnings": [],
}


_cache: dict[str, Any] = {"mtime": None, "data": dict(_DEFAULT)}


def _read_status() -> dict[str, Any]:
    try:
        mtime = STATUS_FILE.stat().st_mtime_ns
    except OSError:
        _cache.update(mtime=None, data=dict(_DEFAULT))
        return _cache["data"]

    if mtime == _cache["mtime"]:
        return _cache["data"]

    try:
        data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("status must be a JSON object")
    except (OSError, ValueError, json.JSONDecodeError):
        data = dict(_DEFAULT)

    _cache.update(mtime=mtime, data=data)
    return data


def _load_image_path() -> Path | None:
    configured = os.getenv("ANOMAVISION_DRIFT_IMAGE_PATH")
    if configured:
        path = Path(configured)
        return path if path.exists() else None

    if not CONFIG_FILE.exists():
        return None

    try:
        from anomavision.config import load_config

        cfg = load_config(str(CONFIG_FILE))
        value = cfg.get("img_path") if hasattr(cfg, "get") else None
        if value:
            path = Path(str(value))
            if not path.is_absolute():
                path = CONFIG_FILE.parent / path
            return path if path.exists() else None
    except Exception:
        return None
    return None


def _image_files() -> list[Path]:
    root = _load_image_path()
    if root is None:
        return []
    if root.is_file():
        return [root]
    return sorted(
        (p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}),
        key=lambda p: p.stat().st_mtime_ns,
    )


def _recent_samples(samples_seen: int) -> list[tuple[str, str]]:
    files = _image_files()
    if not files or samples_seen <= 0:
        return []

    # AnodetDataset processes the directory in its normal file order. The
    # dashboard uses the same count to show the most recently processed inputs.
    ordered = sorted(files, key=lambda p: str(p).lower())
    end = min(samples_seen, len(ordered))
    start = max(0, end - SAMPLE_COUNT)
    selected = ordered[start:end]
    return [
        (str(path), f"Production sample #{start + index + 1}\n{path.name}")
        for index, path in enumerate(selected)
    ]


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _percent(value: Any) -> int:
    try:
        return max(0, min(100, round(float(value) * 100)))
    except (TypeError, ValueError):
        return 0


def _status_copy(status: str, ready: bool) -> tuple[str, str, str]:
    if not ready or status == "warming_up":
        return (
            "COLLECTING DATA",
            "We are learning what normal production data looks like.",
            "warming",
        )
    if status == "drift":
        return (
            "DATA CHANGE DETECTED",
            "Recent production images look different from the approved normal pattern.",
            "danger",
        )
    if status == "stable":
        return (
            "PRODUCTION LOOKS STABLE",
            "Recent production data is consistent with the approved normal pattern.",
            "good",
        )
    return ("MONITORING UNAVAILABLE", "The monitoring service is not reporting yet.", "neutral")


def render() -> tuple[str, list[tuple[str, str]]]:
    data = _read_status()
    status = str(data.get("status", "unavailable")).lower()
    ready = bool(data.get("ready", False))
    label, message, tone = _status_copy(status, ready)

    samples = int(data.get("samples_seen", 0) or 0)
    fill = int(data.get("window_fill", 0) or 0)
    window = int(data.get("window_size", 0) or 0)
    fill_pct = round((fill / window) * 100) if window else 0
    score = _percent(data.get("drift_score"))
    psi = data.get("psi")
    threshold = data.get("threshold", 0.2)
    mean_shift = data.get("mean_shift")
    std_shift = data.get("std_shift")
    cosine_shift = data.get("cosine_shift")
    warnings = data.get("warnings") or []
    updated = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    if tone == "danger":
        accent = _RED
    elif tone == "good":
        accent = _GREEN
    elif tone == "warming":
        accent = _AMBER
    else:
        accent = _PURPLE

    warning_text = ", ".join(str(item).replace("_", " ") for item in warnings) or "No active warnings"
    why = (
        "A change in the incoming images does not automatically mean the model is wrong. "
        "It means production has changed enough that the team should review recent images and quality."
        if status == "drift"
        else "The monitor continuously compares recent production data with the approved normal reference."
    )

    html = f"""
    <div class="av-shell">
      <div class="av-top">
        <div>
          <div class="av-eyebrow">ANOMAVISION / PRODUCTION MONITORING</div>
          <h1>Production <span>Health Center</span></h1>
          <p>One simple view of whether your production images still look like the data your model was built for.</p>
        </div>
        <div class="av-live"><i></i> LIVE · {updated}</div>
      </div>

      <section class="av-hero {tone}">
        <div class="av-hero-copy">
          <div class="av-pill">{label}</div>
          <h2>{message}</h2>
          <p>{why}</p>
        </div>
        <div class="av-ring" style="--score:{score}%;--accent:{accent};">
          <strong>{score}%</strong><small>DATA CHANGE</small>
        </div>
      </section>

      <div class="av-cards">
        <div class="av-card"><small>IMAGES CHECKED</small><strong>{samples:,}</strong><span>production images seen</span></div>
        <div class="av-card"><small>RECENT WINDOW</small><strong>{fill:,}<em> / {window:,}</em></strong><span>{fill_pct}% of the monitoring window</span><div class="av-bar"><b style="width:{fill_pct}%"></b></div></div>
        <div class="av-card"><small>CHANGE LEVEL</small><strong>{score}%</strong><span>0% = no meaningful change</span></div>
        <div class="av-card"><small>MONITORING</small><strong class="{tone}">{'READY' if ready else 'STARTING'}</strong><span>live comparison</span></div>
      </div>

      <div class="av-grid">
        <section class="av-panel">
          <div class="av-head"><div><h3>Recent production images</h3><p>These are the latest images processed by the model.</p></div><span>LIVE</span></div>
          <div class="av-note">Review these images when the status changes. They help answer the most important question: <b>“What changed?”</b></div>
        </section>
      </div>

      <div class="av-grid two">
        <section class="av-panel">
          <div class="av-head"><div><h3>What changed?</h3><p>Plain-language explanation of the live signals.</p></div></div>
          <div class="av-signal"><b>Overall change</b><strong>{score}%</strong><p>How different the recent production data is from the approved reference.</p></div>
          <div class="av-signal"><b>Typical appearance</b><strong>{_number(mean_shift, 2)}</strong><p>How much the typical model representation has moved.</p></div>
          <div class="av-signal"><b>Variation</b><strong>{_number(std_shift, 2)}</strong><p>Whether production images are becoming more or less varied.</p></div>
        </section>

        <section class="av-panel">
          <div class="av-head"><div><h3>What should I do?</h3><p>Simple operational guidance.</p></div></div>
          <ol class="av-steps">
            <li><b>Look at the recent images.</b><span>Check whether the product, camera, lighting, or process changed.</span></li>
            <li><b>Check model quality.</b><span>Compare recent anomaly results with your normal quality measurements.</span></li>
            <li><b>Decide whether the new data is expected.</b><span>If the production change is intentional, the reference can be reviewed later.</span></li>
          </ol>
          <div class="av-warning">● {warning_text}</div>
        </section>
      </div>

      <details class="av-advanced">
        <summary>Technical details</summary>
        <div class="av-tech">
          <div><span>PSI</span><b>{_number(psi, 3)}</b><small>threshold {_number(threshold, 2)}</small></div>
          <div><span>Mean shift</span><b>{_number(mean_shift, 4)}</b></div>
          <div><span>Std shift</span><b>{_number(std_shift, 4)}</b></div>
          <div><span>Cosine shift</span><b>{_number(cosine_shift, 4)}</b></div>
        </div>
      </details>

      <div class="av-footer">ANOMAVISION · Data drift is an early-warning signal. It does not by itself prove that model accuracy has degraded.</div>
    </div>
    """
    return html, _recent_samples(samples)


CSS = f"""
* {{ box-sizing:border-box; }}
body,.gradio-container {{ background:{_BG}!important; color:{_TEXT}!important; font-family:Inter,ui-sans-serif,system-ui,sans-serif!important; }}
footer {{ display:none!important; }}
.av-shell {{ max-width:1280px; margin:auto; padding:28px 28px 46px; }}
.av-top {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; margin-bottom:22px; }}
.av-eyebrow {{ color:{_PURPLE}; font-size:11px; font-weight:800; letter-spacing:.16em; }}
h1 {{ font-size:36px; line-height:1.05; margin:8px 0 8px; letter-spacing:-.045em; }}
h1 span {{ color:#a995ff; }}
.av-top p {{ margin:0; color:{_MUTED}; font-size:14px; line-height:1.55; max-width:760px; }}
.av-live {{ white-space:nowrap; border:1px solid {_BORDER}; background:{_CARD}; color:{_MUTED}; border-radius:999px; padding:9px 14px; font-size:11px; }}
.av-live i {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:{_GREEN}; box-shadow:0 0 12px {_GREEN}; }}
.av-hero {{ min-height:210px; padding:30px 34px; border:1px solid {_BORDER}; border-radius:24px; display:flex; align-items:center; justify-content:space-between; overflow:hidden; position:relative; background:linear-gradient(135deg,{_CARD},#171e31); }}
.av-hero.danger {{ border-color:{_RED}66; }} .av-hero.good {{ border-color:{_GREEN}66; }} .av-hero.warming {{ border-color:{_AMBER}66; }}
.av-hero-copy {{ max-width:790px; position:relative; z-index:1; }}
.av-pill {{ display:inline-block; color:#fff; background:{_PURPLE}22; border:1px solid {_PURPLE}44; border-radius:999px; padding:6px 11px; font-size:10px; font-weight:800; letter-spacing:.09em; }}
.danger .av-pill {{ background:{_RED}20; border-color:{_RED}55; }} .good .av-pill {{ background:{_GREEN}20; border-color:{_GREEN}55; }} .warming .av-pill {{ background:{_AMBER}20; border-color:{_AMBER}55; }}
.av-hero h2 {{ margin:14px 0 8px; font-size:27px; letter-spacing:-.03em; }}
.av-hero p {{ color:{_MUTED}; font-size:13px; line-height:1.65; margin:0; }}
.av-ring {{ width:138px; height:138px; flex:0 0 138px; border-radius:50%; display:grid; place-content:center; text-align:center; background:radial-gradient(circle at center,{_CARD2} 58%,transparent 59%),conic-gradient(var(--accent) 0 var(--score),#283148 var(--score) 100%); box-shadow:0 0 45px color-mix(in srgb,var(--accent) 20%,transparent); position:relative; z-index:1; }}
.av-ring strong {{ font-size:29px; }} .av-ring small {{ color:{_MUTED}; font-size:8px; font-weight:800; letter-spacing:.12em; }}
.av-cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0; }}
.av-card,.av-panel,.av-advanced {{ background:{_CARD}; border:1px solid {_BORDER}; border-radius:17px; }}
.av-card {{ padding:18px; min-height:124px; }} .av-card small {{ display:block; color:{_MUTED}; font-size:9px; font-weight:800; letter-spacing:.12em; }}
.av-card strong {{ display:block; margin-top:11px; font-size:28px; letter-spacing:-.03em; }} .av-card strong.good {{ color:{_GREEN}; }} .av-card strong.warming {{ color:{_AMBER}; }} .av-card strong.danger {{ color:{_RED}; }}
.av-card em {{ color:{_MUTED}; font-size:14px; font-style:normal; }} .av-card span {{ color:{_MUTED}; font-size:11px; display:block; margin-top:5px; }}
.av-bar {{ height:5px; margin-top:12px; background:#252d40; border-radius:8px; overflow:hidden; }} .av-bar b {{ display:block; height:100%; background:linear-gradient(90deg,{_PURPLE},#b39cff); }}
.av-grid {{ margin-top:14px; }} .av-grid.two {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.av-panel {{ padding:21px; }} .av-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }} .av-head h3 {{ margin:0; font-size:16px; }} .av-head p {{ margin:5px 0 0; color:{_MUTED}; font-size:11px; }} .av-head > span {{ color:{_GREEN}; font-size:9px; font-weight:800; border:1px solid {_GREEN}55; border-radius:999px; padding:5px 8px; }}
.av-note {{ margin-top:16px; padding:13px 14px; border-radius:12px; background:{_CARD2}; color:{_MUTED}; font-size:12px; line-height:1.6; }} .av-note b {{ color:{_TEXT}; }}
.av-signal {{ padding:14px 0; border-bottom:1px solid {_BORDER}; }} .av-signal:last-child {{ border-bottom:0; }} .av-signal b {{ font-size:12px; }} .av-signal strong {{ float:right; font-size:15px; }} .av-signal p {{ clear:both; color:{_MUTED}; font-size:11px; margin:6px 0 0; line-height:1.5; }}
.av-steps {{ margin:15px 0 0; padding-left:20px; }} .av-steps li {{ margin:0 0 14px; padding-left:4px; }} .av-steps b {{ font-size:12px; }} .av-steps span {{ display:block; color:{_MUTED}; font-size:11px; margin-top:3px; line-height:1.5; }}
.av-warning {{ margin-top:14px; color:#ff9aae; background:{_RED}12; border:1px solid {_RED}33; border-radius:10px; padding:9px 11px; font-size:10px; }}
.av-advanced {{ margin-top:14px; padding:0; overflow:hidden; }} .av-advanced summary {{ cursor:pointer; padding:15px 18px; color:{_MUTED}; font-size:11px; font-weight:700; }} .av-tech {{ display:grid; grid-template-columns:repeat(4,1fr); border-top:1px solid {_BORDER}; }} .av-tech div {{ padding:15px 18px; border-right:1px solid {_BORDER}; }} .av-tech span,.av-tech small {{ display:block; color:{_MUTED}; font-size:10px; }} .av-tech b {{ display:block; margin:6px 0; font-size:15px; }}
.av-footer {{ color:{_MUTED}; text-align:center; font-size:10px; margin-top:18px; }}
@media(max-width:900px) {{ .av-cards,.av-grid.two {{ grid-template-columns:1fr 1fr; }} .av-ring {{ width:110px;height:110px;flex-basis:110px; }} }}
@media(max-width:650px) {{ .av-shell {{ padding:18px 12px 30px; }} .av-top,.av-hero {{ flex-direction:column; }} .av-cards,.av-grid.two,.av-tech {{ grid-template-columns:1fr; }} .av-ring {{ align-self:center; }} }}
"""


def create_app() -> gr.Blocks:
    with gr.Blocks(title="AnomaVision Production Health") as app:
        dashboard = gr.HTML()
        gallery = gr.Gallery(
            label="Recent production images",
            columns=4,
            rows=2,
            height="auto",
            object_fit="cover",
            show_label=False,
            allow_preview=True,
        )
        timer = gr.Timer(value=REFRESH_SECONDS, active=True)
        timer.tick(render, outputs=[dashboard, gallery])
        app.load(render, outputs=[dashboard, gallery])
    return app


def launch(*, status_file: str | Path | None = None, image_path: str | Path | None = None, port: int = 7860, inbrowser: bool = True) -> None:
    """Launch the dashboard in a background thread for the detect CLI."""
    global STATUS_FILE
    if status_file is not None:
        STATUS_FILE = Path(status_file)
    if image_path is not None:
        os.environ["ANOMAVISION_DRIFT_IMAGE_PATH"] = str(image_path)

    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=port, inbrowser=inbrowser, prevent_thread_lock=True, show_error=True)


def main() -> None:
    port = int(os.getenv("ANOMAVISION_DRIFT_DASHBOARD_PORT", "7860"))
    launch(port=port, inbrowser=True)
    # Keep the standalone process alive.
    threading.Event().wait()


if __name__ == "__main__":
    main()
