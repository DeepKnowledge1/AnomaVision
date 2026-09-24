"use client";

import { useState } from "react";
import {
  Activity, Box, BrainCircuit, ChevronDown, CircleGauge, Database,
  FlaskConical, LayoutDashboard, MonitorCog, Play, Rocket, Settings2,
  ShieldCheck, SlidersHorizontal, Sparkles, UploadCloud, Wifi
} from "lucide-react";

type Page = "Overview" | "Projects" | "Datasets" | "Training" | "Models" | "Deployments" | "Live" | "Monitoring";

const nav: { label: Page; icon: React.ElementType }[] = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Projects", icon: Box },
  { label: "Datasets", icon: Database },
  { label: "Training", icon: BrainCircuit },
  { label: "Models", icon: FlaskConical },
  { label: "Deployments", icon: Rocket },
  { label: "Live", icon: Wifi },
  { label: "Monitoring", icon: Activity },
];

const workflows = [
  ["01", "Dataset", "Inspect image quality before training."],
  ["02", "Train", "Run PaDiM or PatchCore with the existing engine."],
  ["03", "Validate", "Check artifact integrity and runtime behavior."],
  ["04", "Deploy", "Target CPU, ONNX, OpenVINO, TensorRT or Hailo."],
  ["05", "Monitor", "Watch latency, health and production drift."],
];

export default function StudioPage() {
  const [page, setPage] = useState<Page>("Overview");

  return (
    <div className="studio-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">AV</div>
          <div>
            <div className="brand-name">AnomaVision</div>
            <div className="brand-sub">Studio · 0.1</div>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        {nav.map(({ label, icon: Icon }) => (
          <button key={label} className={`nav-item ${page === label ? "active" : ""}`} onClick={() => setPage(label)}>
            <Icon size={15} strokeWidth={1.8} /><span>{label}</span>
          </button>
        ))}

        <div className="nav-label">SYSTEM</div>
        <button className="nav-item"><Settings2 size={15} strokeWidth={1.8} /><span>Settings</span></button>

        <div className="sidebar-bottom">
          <div className="storage"><strong>Local workspace</strong>~/.anomavision/projects</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="crumb">Studio / {page}</div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <button className="project-switcher"><CircleGauge size={13} /> Bottle Inspection <ChevronDown size={13} /></button>
            <div className="avatar">AR</div>
          </div>
        </header>

        <div className="content">
          {page === "Overview" ? <Overview onNavigate={setPage} /> : <Placeholder page={page} />}
        </div>
      </main>
    </div>
  );
}

function Overview({ onNavigate }: { onNavigate: (page: Page) => void }) {
  return (
    <>
      <div className="page-head">
        <div>
          <div className="eyebrow">Project overview</div>
          <h1>Bottle Inspection</h1>
          <p className="subtitle">Industrial anomaly detection · PatchCore · local workspace</p>
        </div>
        <button className="primary" onClick={() => onNavigate("Training")}><Play size={13} style={{ marginRight: 7, verticalAlign: -2 }} /> New training run</button>
      </div>

      <div className="grid-4">
        <Stat icon={<Database size={15} />} label="Dataset" value="1,248" meta="normal images · ready" />
        <Stat icon={<BrainCircuit size={15} />} label="Latest model" value="PatchCore" meta="trained · 18 min ago" />
        <Stat icon={<ShieldCheck size={15} />} label="Validation" value="Passed" meta="CPU · ONNX · 11.0 ms" green />
        <Stat icon={<MonitorCog size={15} />} label="Deployment" value="Not deployed" meta="choose a target" />
      </div>

      <section className="section">
        <div className="section-head">
          <div className="section-title">Studio workflow</div>
          <div className="section-link">Data → Train → Validate → Deploy → Monitor</div>
        </div>
        <div className="workflow">
          {workflows.map(([step, title, text]) => (
            <div className="card workflow-card" key={step}>
              <div className="step">{step}</div>
              <div className="workflow-title">{title}</div>
              <div className="workflow-text">{text}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="section activity">
        <div className="card">
          <div className="section-head"><div className="section-title">Recent activity</div><div className="section-link">View all</div></div>
          <ActivityRow icon={<UploadCloud size={14} />} title="Dataset analyzed" sub="bottle / train / good · 1,248 images" badge="Ready" />
          <ActivityRow icon={<BrainCircuit size={14} />} title="PatchCore training completed" sub="18 minutes ago · 224 × 224 · CPU" badge="Ready" />
          <ActivityRow icon={<ShieldCheck size={14} />} title="ONNX validation completed" sub="11.0 ms latency · consistency checked" badge="Passed" />
        </div>

        <div className="card">
          <div className="section-head"><div className="section-title">Runtime health</div><div className="section-link">Monitoring</div></div>
          <div className="row"><div className="row-main"><div className="icon-box"><CircleGauge size={14} /></div><div><div className="row-title">Inference engine</div><div className="row-sub">AnomaVision core</div></div></div><div className="badge"><span className="status-dot" />Healthy</div></div>
          <div className="row"><div className="row-main"><div className="icon-box"><SlidersHorizontal size={14} /></div><div><div className="row-title">Data drift</div><div className="row-sub">Reference window · 500</div></div></div><div className="badge">No alert</div></div>
          <div className="row"><div className="row-main"><div className="icon-box"><Sparkles size={14} /></div><div><div className="row-title">Studio</div><div className="row-sub">Local API connection</div></div></div><div className="badge">Ready</div></div>
        </div>
      </section>
    </>
  );
}

function Stat({ icon, label, value, meta, green }: { icon: React.ReactNode; label: string; value: string; meta: string; green?: boolean }) {
  return <div className="card"><div className="stat-label">{icon} <span style={{ marginLeft: 5 }}>{label}</span></div><div className="stat-value">{value}</div><div className="stat-meta">{green && <span className="status-dot" />}{meta}</div></div>;
}

function ActivityRow({ icon, title, sub, badge }: { icon: React.ReactNode; title: string; sub: string; badge: string }) {
  return <div className="row"><div className="row-main"><div className="icon-box">{icon}</div><div><div className="row-title">{title}</div><div className="row-sub">{sub}</div></div></div><div className="badge">{badge}</div></div>;
}

function Placeholder({ page }: { page: Page }) {
  const descriptions: Record<Page, string> = {
    Overview: "",
    Projects: "Manage local anomaly-detection projects and their datasets, experiments and deployments.",
    Datasets: "Inspect image quality, duplicates, resolutions and training readiness.",
    Training: "Launch reproducible training runs through the existing AnomaVision training engine.",
    Models: "Browse trained artifacts, metrics and evaluation results.",
    Deployments: "Export and validate models for production targets without changing the algorithm core.",
    Live: "Inspect camera or stream inference using the existing AnomaVision runtime.",
    Monitoring: "Track runtime health, latency and production data drift.",
  };
  return (
    <>
      <div className="page-head">
        <div><div className="eyebrow">Workspace</div><h1>{page}</h1><p className="subtitle">{descriptions[page]}</p></div>
        <button className="secondary"><SlidersHorizontal size={13} style={{ marginRight: 7, verticalAlign: -2 }} /> Configure</button>
      </div>
      <div className="card" style={{ minHeight: 260, display: "grid", placeItems: "center" }}>
        <div style={{ textAlign: "center", maxWidth: 420 }}>
          <div className="icon-box" style={{ margin: "0 auto 14px", width: 42, height: 42 }}><Sparkles size={18} /></div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>This workspace is ready for the API</div>
          <div className="subtitle">The React/Next.js shell is now separate from the Python ML engine. The next step is wiring these views to the existing Studio services.</div>
        </div>
      </div>
    </>
  );
}
