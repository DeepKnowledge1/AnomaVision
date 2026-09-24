"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity, Box, BrainCircuit, ChevronDown, CircleGauge, Database,
  FlaskConical, LayoutDashboard, MonitorCog, Play, Rocket, Settings2,
  ShieldCheck, SlidersHorizontal, Sparkles, UploadCloud, Wifi
} from "lucide-react";

type Page = "Overview" | "Projects" | "Datasets" | "Training" | "Models" | "Deployments" | "Live" | "Monitoring";

type Project = {
  id: string;
  name: string;
  description?: string;
  algorithm: string;
  status: string;
};

type Model = {
  id: string;
  algorithm: string;
  class_name: string;
  run_name: string;
  status: string;
};

const API_BASE = process.env.NEXT_PUBLIC_STUDIO_API_URL ?? "http://localhost:8000";

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
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string>("");
  const [models, setModels] = useState<Model[]>([]);
  const [apiHealthy, setApiHealthy] = useState(false);

  useEffect(() => {
    void loadProjects();
  }, []);

  useEffect(() => {
    if (!selectedProject) {
      setModels([]);
      return;
    }
    void loadModels(selectedProject);
  }, [selectedProject]);

  async function loadProjects() {
    try {
      const [healthResponse, projectResponse] = await Promise.all([
        fetch(`${API_BASE}/api/health`, { cache: "no-store" }),
        fetch(`${API_BASE}/api/projects`, { cache: "no-store" }),
      ]);
      if (!healthResponse.ok || !projectResponse.ok) throw new Error("Studio API unavailable");
      const data = (await projectResponse.json()) as Project[];
      setApiHealthy(true);
      setProjects(data);
      setSelectedProject((current) => current || data[0]?.id || "");
    } catch {
      setApiHealthy(false);
    }
  }

  async function loadModels(projectId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/projects/${projectId}/models`, { cache: "no-store" });
      if (!response.ok) throw new Error("Models unavailable");
      setModels((await response.json()) as Model[]);
    } catch {
      setModels([]);
    }
  }

  const project = useMemo(
    () => projects.find((item) => item.id === selectedProject),
    [projects, selectedProject],
  );

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
          <div className="storage">
            <strong>Studio API</strong>
            <span className={`status-dot ${apiHealthy ? "" : "offline-dot"}`} />
            {apiHealthy ? "Connected" : "Offline"}
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="crumb">Studio / {page}</div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <select
              className="project-switcher"
              value={selectedProject}
              onChange={(event) => setSelectedProject(event.target.value)}
              aria-label="Select project"
            >
              {projects.length === 0 && <option value="">No projects</option>}
              {projects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
            <div className="avatar">AR</div>
          </div>
        </header>

        <div className="content">
          {page === "Overview" ? (
            <Overview project={project} models={models} apiHealthy={apiHealthy} onNavigate={setPage} />
          ) : (
            <Placeholder page={page} />
          )}
        </div>
      </main>
    </div>
  );
}

function Overview({
  project,
  models,
  apiHealthy,
  onNavigate,
}: {
  project?: Project;
  models: Model[];
  apiHealthy: boolean;
  onNavigate: (page: Page) => void;
}) {
  const latestModel = models[0];
  const projectName = project?.name ?? "No project selected";
  const algorithm = latestModel?.algorithm ?? project?.algorithm ?? "—";

  return (
    <>
      <div className="page-head">
        <div>
          <div className="eyebrow">Project overview</div>
          <h1>{projectName}</h1>
          <p className="subtitle">
            {project?.description || `Industrial anomaly detection · ${algorithm.toUpperCase()} · local workspace`}
          </p>
        </div>
        <button className="primary" onClick={() => onNavigate("Training")}><Play size={13} style={{ marginRight: 7, verticalAlign: -2 }} /> New training run</button>
      </div>

      {!apiHealthy && (
        <div className="api-warning">
          <CircleGauge size={14} />
          <span>Studio API is offline. Start the FastAPI service to load your real projects and models.</span>
        </div>
      )}

      <div className="grid-4">
        <Stat icon={<Database size={15} />} label="Projects" value={String(project ? 1 : 0)} meta={project ? "active project" : "create a project"} />
        <Stat icon={<BrainCircuit size={15} />} label="Models" value={String(models.length)} meta={latestModel ? `${latestModel.algorithm} · ${latestModel.status}` : "no trained models"} />
        <Stat icon={<ShieldCheck size={15} />} label="Validation" value="Ready" meta="deployment validation available" green />
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
          <div className="section-head"><div className="section-title">Recent model activity</div><div className="section-link">{models.length} model{models.length === 1 ? "" : "s"}</div></div>
          {models.length === 0 ? (
            <div className="empty">No trained models registered for this project yet.</div>
          ) : (
            models.slice(0, 4).map((model) => (
              <ActivityRow
                key={model.id}
                icon={<BrainCircuit size={14} />}
                title={`${model.algorithm.toUpperCase()} · ${model.class_name}`}
                sub={model.run_name}
                badge={model.status}
              />
            ))
          )}
        </div>

        <div className="card">
          <div className="section-head"><div className="section-title">Runtime health</div><div className="section-link">Monitoring</div></div>
          <div className="row"><div className="row-main"><div className="icon-box"><CircleGauge size={14} /></div><div><div className="row-title">Studio API</div><div className="row-sub">Python / FastAPI adapter</div></div></div><div className="badge"><span className={`status-dot ${apiHealthy ? "" : "offline-dot"}`} />{apiHealthy ? "Healthy" : "Offline"}</div></div>
          <div className="row"><div className="row-main"><div className="icon-box"><SlidersHorizontal size={14} /></div><div><div className="row-title">Data drift</div><div className="row-sub">Existing observer-only monitor</div></div></div><div className="badge">Available</div></div>
          <div className="row"><div className="row-main"><div className="icon-box"><Sparkles size={14} /></div><div><div className="row-title">AnomaVision core</div><div className="row-sub">Training and inference engine</div></div></div><div className="badge">Ready</div></div>
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
          <div style={{ fontSize: 15, fontWeight: 700 }}>Connected to the Studio architecture</div>
          <div className="subtitle">This view is ready to consume the same Python services through the Studio API. No ML logic is duplicated in the frontend.</div>
        </div>
      </div>
    </>
  );
}
