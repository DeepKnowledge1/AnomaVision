"""
Customer-facing AnomaVision data drift dashboard.

This UI is intentionally read-only. It never touches the anomaly model or
inference path. A ProductionDriftMonitor can persist its status JSON to the
path configured by ANOMAVISION_DRIFT_STATUS_FILE.

Run:
    ANOMAVISION_DRIFT_STATUS_FILE=monitoring/drift_status.json \
        python apps/ui/drift_dashboard.py
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

STATUS_FILE = Path(
    os.getenv("ANOMAVISION_DRIFT_STATUS_FILE", "monitoring/drift_status.json")
)
REFRESH_SECONDS = int(os.getenv("ANOMAVISION_DRIFT_REFRESH_SECONDS", "15"))

_ACCENT = "#7c5cff"
_ACCENT2 = "#ff5c7a"
_GREEN = "#2bd576"
_BG = "#080b14"
_CARD = "#101522"
_CARD2 = "#141a2a"
_BORDER = "#20283a"
_TEXT = "#f4f7ff"
_MUTED = "#8c96ad"


def _read_status() -> dict[str, Any]:
    try:
        with STATUS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {
            "ready": False,
            "status": "unavailable",
            "samples_seen": 0,
            "window_size": 0,
            "window_fill": 0,
            "drift_score": 0.0,
            "psi": 0.0,
            "mean_shift": 0.0,
            "std_shift": 0.0,
            "cosine_shift": 0.0,
            "threshold": 0.2,
            "warnings": ["monitoring_status_unavailable"],
        }


def _pct(value: float) -> str:
    return f"{max(0.0, min(1.0, float(value))) * 100:.0f}%"


def _num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _bar(value: float, maximum: float = 1.0) -> str:
    try:
        ratio = max(0.0, min(1.0, float(value) / maximum))
    except (TypeError, ValueError, ZeroDivisionError):
        ratio = 0.0
    return f"<div class='metric-bar'><span style='width:{ratio * 100:.1f}%'></span></div>"


def _history_html(history: list[Any]) -> str:
    if not history:
        return "<div class='empty-history'>Live monitoring · historical samples will appear here when available.</div>"

    cells = []
    for item in history[-24:]:
        if not isinstance(item, dict):
            continue
        state = str(item.get("status", "unknown")).lower()
        cls = "drift" if state == "drift" else "normal" if state == "stable" else "unknown"
        cells.append(f"<span class='history-cell {cls}' title='{state}'></span>")
    return "<div class='history-row'>" + "".join(cells) + "</div>"


def render_dashboard() -> tuple[str, dict]:
    data = _read_status()
    status = str(data.get("status", "unavailable")).lower()
    ready = bool(data.get("ready", False))
    is_drift = status == "drift"

    if not ready and status == "unavailable":
        headline = "Monitoring is waiting for data"
        subtitle = "Connect ProductionDriftMonitor to this dashboard to see live production health."
        status_label = "NOT CONNECTED"
        status_class = "neutral"
    elif is_drift:
        headline = "Data drift detected"
        subtitle = "Production embeddings are materially different from the reference distribution."
        status_label = "DRIFT DETECTED"
        status_class = "danger"
    else:
        headline = "Production data is stable"
        subtitle = "The current production window is within the configured drift threshold."
        status_label = "STABLE"
        status_class = "good"

    score = float(data.get("drift_score", 0.0) or 0.0)
    samples = int(data.get("samples_seen", 0) or 0)
    fill = int(data.get("window_fill", 0) or 0)
    window = int(data.get("window_size", 0) or 0)
    psi = float(data.get("psi", 0.0) or 0.0)
    mean_shift = float(data.get("mean_shift", 0.0) or 0.0)
    std_shift = float(data.get("std_shift", 0.0) or 0.0)
    cosine_shift = float(data.get("cosine_shift", 0.0) or 0.0)
    threshold = float(data.get("threshold", 0.2) or 0.2)
    warnings = data.get("warnings", []) or []
    history = data.get("history", []) or []

    fill_ratio = (fill / window) if window else 0.0
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    warning_html = "".join(
        f"<span class='warning-pill'>⚠ {str(w).replace('_', ' ')}</span>" for w in warnings
    ) or "<span class='warning-pill muted'>No active warnings</span>"

    html = f"""
    <div class='dashboard'>
      <div class='topline'>
        <div>
          <div class='eyebrow'>ANOMAVISION / PRODUCTION MONITORING</div>
          <h1>Data Drift <span>Command Center</span></h1>
          <p class='subtitle'>{subtitle}</p>
        </div>
        <div class='live'><i></i> LIVE&nbsp;&nbsp; {updated}</div>
      </div>

      <div class='hero {status_class}'>
        <div class='hero-copy'>
          <div class='status-badge'>{status_label}</div>
          <h2>{headline}</h2>
          <p>Monitoring compares incoming model representations with the approved reference distribution.</p>
        </div>
        <div class='score-ring'>
          <div class='score'>{_pct(score)}</div>
          <div class='score-label'>DRIFT SCORE</div>
        </div>
      </div>

      <div class='cards'>
        <div class='card'><div class='card-label'>SAMPLES OBSERVED</div><div class='card-value'>{samples:,}</div><div class='card-meta'>production samples</div></div>
        <div class='card'><div class='card-label'>CURRENT WINDOW</div><div class='card-value'>{fill:,}<span> / {window:,}</span></div><div class='card-meta'>{_pct(fill_ratio)} filled</div>{_bar(fill_ratio)}</div>
        <div class='card'><div class='card-label'>PSI</div><div class='card-value'>{_num(psi, 2)}</div><div class='card-meta'>threshold {_num(threshold, 2)}</div></div>
        <div class='card'><div class='card-label'>MODEL DIRECTION</div><div class='card-value'>{_num(cosine_shift, 3)}</div><div class='card-meta'>cosine shift</div></div>
      </div>

      <div class='grid-2'>
        <div class='panel'>
          <div class='panel-head'><div><b>Distribution signals</b><small>What changed in production?</small></div><span class='mini-badge'>LIVE</span></div>
          <div class='signal'><div><b>Mean shift</b><small>Location of the embedding distribution</small></div><strong>{_num(mean_shift, 3)}</strong></div>
          {_bar(mean_shift, 1.0)}
          <div class='signal'><div><b>Std shift</b><small>Spread / variance change</small></div><strong>{_num(std_shift, 3)}</strong></div>
          {_bar(std_shift, 1.0)}
          <div class='signal'><div><b>Cosine shift</b><small>Change in embedding direction</small></div><strong>{_num(cosine_shift, 3)}</strong></div>
          {_bar(cosine_shift, 1.0)}
        </div>

        <div class='panel'>
          <div class='panel-head'><div><b>Monitoring health</b><small>Reference → production window</small></div></div>
          <div class='health-row'><span>Reference comparison</span><strong class='ok'>READY</strong></div>
          <div class='health-row'><span>Window coverage</span><strong>{_pct(fill_ratio)}</strong></div>
          <div class='health-row'><span>Drift threshold</span><strong>{_num(threshold, 2)}</strong></div>
          <div class='health-row'><span>Evaluation status</span><strong class='{status_class}'>{status.upper()}</strong></div>
          <div class='warnings'>{warning_html}</div>
        </div>
      </div>

      <div class='panel history'>
        <div class='panel-head'><div><b>Drift timeline</b><small>Recent monitoring evaluations</small></div></div>
        {_history_html(history)}
      </div>

      <div class='footer-note'>
        <span>ANOMAVISION</span> · Drift is an early-warning signal, not proof of model degradation. Review recent samples and downstream quality metrics before taking corrective action.
      </div>
    </div>
    """
    return html, data


custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
* {{ box-sizing: border-box; }}
body, .gradio-container {{ background:{_BG} !important; color:{_TEXT} !important; font-family:'Plus Jakarta Sans',sans-serif !important; }}
footer {{ display:none !important; }}
.dashboard {{ max-width:1320px; margin:0 auto; padding:28px 28px 50px; }}
.topline {{ display:flex; justify-content:space-between; align-items:flex-start; gap:20px; margin-bottom:22px; }}
.eyebrow {{ color:{_ACCENT}; font-size:11px; font-weight:800; letter-spacing:.16em; }}
h1 {{ font-size:34px; letter-spacing:-.04em; margin:7px 0 5px; font-weight:800; }}
h1 span {{ color:#9d8cff; }}
.subtitle {{ color:{_MUTED}; margin:0; font-size:14px; max-width:760px; line-height:1.6; }}
.live {{ border:1px solid {_BORDER}; background:{_CARD}; border-radius:999px; padding:9px 14px; color:{_MUTED}; font-size:11px; white-space:nowrap; }}
.live i {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:{_GREEN}; box-shadow:0 0 12px { _GREEN }; }}
.hero {{ min-height:190px; border:1px solid {_BORDER}; border-radius:22px; padding:30px 34px; display:flex; justify-content:space-between; align-items:center; background:linear-gradient(135deg,{_CARD},#151c2e); box-shadow:0 20px 55px rgba(0,0,0,.22); position:relative; overflow:hidden; }}
.hero::after {{ content:''; position:absolute; right:-70px; top:-100px; width:360px; height:360px; border-radius:50%; background:radial-gradient(circle,{_ACCENT}32,transparent 68%); pointer-events:none; }}
.hero.danger {{ border-color:{_ACCENT2}55; }}
.hero.good {{ border-color:{_GREEN}55; }}
.hero.neutral {{ border-color:#64748b55; }}
.status-badge {{ display:inline-block; padding:5px 10px; border-radius:999px; font-size:10px; font-weight:800; letter-spacing:.1em; background:{_ACCENT}18; color:#b5a9ff; }}
.danger .status-badge {{ background:{_ACCENT2}18; color:#ff8fa3; }}
.good .status-badge {{ background:{_GREEN}18; color:#65e9a0; }}
.hero h2 {{ font-size:28px; margin:13px 0 7px; letter-spacing:-.03em; }}
.hero p {{ color:{_MUTED}; max-width:680px; font-size:13px; line-height:1.6; margin:0; }}
.score-ring {{ width:126px; height:126px; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center; background:radial-gradient(circle at center,{_CARD2} 58%,transparent 59%), conic-gradient({_ACCENT} 0 79%,#252d40 79% 100%); box-shadow:0 0 35px {_ACCENT}22; position:relative; z-index:1; flex-shrink:0; }}
.danger .score-ring {{ background:radial-gradient(circle at center,{_CARD2} 58%,transparent 59%), conic-gradient({_ACCENT2} 0 79%,#252d40 79% 100%); }}
.score {{ font-size:26px; font-weight:800; }}
.score-label {{ color:{_MUTED}; font-size:8px; font-weight:800; letter-spacing:.12em; margin-top:2px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0; }}
.card,.panel {{ background:{_CARD}; border:1px solid {_BORDER}; border-radius:16px; }}
.card {{ padding:18px; min-height:118px; }}
.card-label {{ color:{_MUTED}; font-size:9px; font-weight:800; letter-spacing:.13em; }}
.card-value {{ font-size:27px; font-weight:800; margin-top:12px; letter-spacing:-.03em; }}
.card-value span {{ color:{_MUTED}; font-size:15px; font-weight:600; }}
.card-meta {{ color:{_MUTED}; font-size:11px; margin-top:5px; }}
.metric-bar {{ height:5px; border-radius:10px; background:#20283a; overflow:hidden; margin-top:12px; }}
.metric-bar span {{ display:block; height:100%; background:linear-gradient(90deg,{_ACCENT},#a78bfa); border-radius:10px; }}
.grid-2 {{ display:grid; grid-template-columns:1.35fr 1fr; gap:14px; }}
.panel {{ padding:20px; margin-bottom:14px; }}
.panel-head {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:18px; }}
.panel-head b {{ font-size:14px; }}
.panel-head small {{ display:block; color:{_MUTED}; font-size:11px; margin-top:4px; }}
.mini-badge {{ font-size:9px; color:{_GREEN}; border:1px solid {_GREEN}44; padding:4px 7px; border-radius:999px; font-weight:800; }}
.signal {{ display:flex; justify-content:space-between; align-items:center; margin-top:15px; }}
.signal b {{ font-size:12px; }}
.signal small {{ display:block; color:{_MUTED}; font-size:10px; margin-top:3px; }}
.signal strong {{ font-family:monospace; font-size:13px; }}
.health-row {{ display:flex; justify-content:space-between; padding:13px 0; border-bottom:1px solid {_BORDER}; color:{_MUTED}; font-size:12px; }}
.health-row strong {{ color:{_TEXT}; font-size:11px; letter-spacing:.05em; }}
.health-row strong.ok,.health-row strong.good {{ color:{_GREEN}; }}
.health-row strong.danger {{ color:{_ACCENT2}; }}
.warnings {{ padding-top:15px; display:flex; flex-wrap:wrap; gap:7px; }}
.warning-pill {{ background:{_ACCENT2}12; border:1px solid {_ACCENT2}30; color:#ff91a4; border-radius:999px; padding:5px 8px; font-size:10px; font-weight:700; }}
.warning-pill.muted {{ color:{_MUTED}; background:{_MUTED}0d; border-color:{_BORDER}; }}
.history-row {{ display:flex; gap:6px; height:34px; align-items:center; }}
.history-cell {{ height:26px; flex:1; max-width:48px; border-radius:6px; background:#31394c; }}
.history-cell.drift {{ background:linear-gradient(180deg,#ff7690,{_ACCENT2}); box-shadow:0 0 10px {_ACCENT2}22; }}
.history-cell.normal {{ background:linear-gradient(180deg,#4ce69a,{_GREEN}); }}
.empty-history {{ color:{_MUTED}; font-size:11px; padding:10px 0; }}
.footer-note {{ color:#667188; font-size:10px; line-height:1.6; padding:4px 4px; }}
.footer-note span {{ color:#9d8cff; font-weight:800; letter-spacing:.1em; }}
@media(max-width:900px) {{ .cards {{grid-template-columns:repeat(2,1fr);}} .grid-2 {{grid-template-columns:1fr;}} .hero {{align-items:flex-start;}} .score-ring {{width:100px;height:100px;}} }}
@media(max-width:600px) {{ .dashboard {{padding:16px;}} .topline {{flex-direction:column;}} .cards {{grid-template-columns:1fr;}} .hero {{padding:22px; flex-direction:column; gap:20px;}} h1 {{font-size:28px;}} }}
"""


def refresh() -> tuple[str, dict]:
    return render_dashboard()


with gr.Blocks(title="AnomaVision — Data Drift Command Center", css=custom_css) as demo:
    gr.HTML("<div style='display:none'>AnomaVision Data Drift Dashboard</div>")
    dashboard = gr.HTML()
    raw = gr.JSON(label="Live monitoring payload", open=False)
    refresh_btn = gr.Button("↻  Refresh monitoring", variant="secondary")

    demo.load(fn=refresh, outputs=[dashboard, raw])
    refresh_btn.click(fn=refresh, outputs=[dashboard, raw])


if __name__ == "__main__":
    print(f"[drift-dashboard] status file: {STATUS_FILE}")
    print(f"[drift-dashboard] refresh: manual / every {REFRESH_SECONDS}s configured")
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("DRIFT_DASHBOARD_PORT", "7861")), share=False, show_error=True)
