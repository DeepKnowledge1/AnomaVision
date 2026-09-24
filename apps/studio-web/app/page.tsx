"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity, Box, BrainCircuit, CircleGauge, Database, FlaskConical,
  LayoutDashboard, MonitorCog, Play, Rocket, Settings2, ShieldCheck,
  SlidersHorizontal, Sparkles, Wifi
} from "lucide-react";

type Page = "Overview" | "Projects" | "Datasets" | "Training" | "Models" | "Deployments" | "Live" | "Monitoring";
type Project = { id: string; name: string; description?: string; algorithm: string; status: string; };
type Model = { id: string; algorithm: string; class_name: string; run_name: string; status: string; path?: string; };
type DatasetReport = {
  image_count: number; valid_count: number; failed_count: number; duplicate_count: number;
  resolutions: Record<string, number>; failures?: { path: string; error: string }[];
};
type Catalog = { algorithms: Record<string, unknown>; deployment_targets: Record<string, unknown> };

const API_BASE = process.env.NEXT_PUBLIC_STUDIO_API_URL ?? "http://localhost:8000";
const nav: { label: Page; icon: React.ElementType }[] = [
  { label: "Overview", icon: LayoutDashboard }, { label: "Projects", icon: Box },
  { label: "Datasets", icon: Database }, { label: "Training", icon: BrainCircuit },
  { label: "Models", icon: FlaskConical }, { label: "Deployments", icon: Rocket },
  { label: "Live", icon: Wifi }, { label: "Monitoring", icon: Activity },
];
const workflows = [
  ["01", "Dataset", "Inspect image quality before training."], ["02", "Train", "Run PaDiM or PatchCore with the existing engine."],
  ["03", "Validate", "Check artifact integrity and runtime behavior."], ["04", "Deploy", "Target CPU, ONNX, OpenVINO, TensorRT or Hailo."],
  ["05", "Monitor", "Watch latency, health and production drift."],
];

export default function StudioPage() {
  const [page, setPage] = useState<Page>("Overview");
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState("");
  const [models, setModels] = useState<Model[]>([]);
  const [apiHealthy, setApiHealthy] = useState(false);

  async function loadProjects() {
    try {
      const [health, response] = await Promise.all([
        fetch(`${API_BASE}/api/health`, { cache: "no-store" }),
        fetch(`${API_BASE}/api/projects`, { cache: "no-store" }),
      ]);
      if (!health.ok || !response.ok) throw new Error();
      const data = (await response.json()) as Project[];
      setProjects(data); setSelectedProject((current) => current || data[0]?.id || ""); setApiHealthy(true);
    } catch { setApiHealthy(false); }
  }
  async function loadModels(projectId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/projects/${projectId}/models`, { cache: "no-store" });
      setModels(response.ok ? ((await response.json()) as Model[]) : []);
    } catch { setModels([]); }
  }
  useEffect(() => { void loadProjects(); }, []);
  useEffect(() => { if (selectedProject) void loadModels(selectedProject); else setModels([]); }, [selectedProject]);

  const project = useMemo(() => projects.find((item) => item.id === selectedProject), [projects, selectedProject]);

  return <div className="studio-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">AV</div><div><div className="brand-name">AnomaVision</div><div className="brand-sub">Studio · 0.1</div></div></div>
      <div className="nav-label">WORKSPACE</div>
      {nav.map(({ label, icon: Icon }) => <button key={label} className={`nav-item ${page === label ? "active" : ""}`} onClick={() => setPage(label)}><Icon size={15} strokeWidth={1.8}/><span>{label}</span></button>)}
      <div className="nav-label">SYSTEM</div>
      <button className="nav-item"><Settings2 size={15} strokeWidth={1.8}/><span>Settings</span></button>
      <div className="sidebar-bottom"><div className="storage"><strong>Studio API</strong><span className={`status-dot ${apiHealthy ? "" : "offline-dot"}`}/>{apiHealthy ? "Connected" : "Offline"}</div></div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div className="crumb">Studio / {page}</div>
        <div className="topbar-actions">
          <select className="project-switcher" value={selectedProject} onChange={(e) => setSelectedProject(e.target.value)} aria-label="Select project">
            {projects.length === 0 && <option value="">No projects</option>}
            {projects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select><div className="avatar">AR</div>
        </div>
      </header>
      <div className="content">
        {page === "Overview" && <Overview project={project} projects={projects} models={models} apiHealthy={apiHealthy} onNavigate={setPage}/>}
        {page === "Projects" && <ProjectsPage projects={projects} selectedProject={selectedProject} onSelect={setSelectedProject} onCreated={loadProjects}/>}
        {page === "Datasets" && <DatasetsPage project={project}/>}
        {page === "Training" && <TrainingPage project={project} onFinished={() => loadModels(selectedProject)}/>}
        {page === "Models" && <ModelsPage models={models} onRefresh={() => loadModels(selectedProject)}/>}
        {page === "Deployments" && <DeploymentsPage project={project} models={models}/>} {["Live", "Monitoring"].includes(page) && <Placeholder page={page}/>}
      </div>
    </main>
  </div>;
}

function Overview({ project, projects, models, apiHealthy, onNavigate }: { project?: Project; projects: Project[]; models: Model[]; apiHealthy: boolean; onNavigate: (p: Page) => void }) {
  const latest = models[0];
  return <>
    <div className="page-head"><div><div className="eyebrow">Project overview</div><h1>{project?.name ?? "No project selected"}</h1><p className="subtitle">{project?.description || "Industrial anomaly detection · local workspace"}</p></div><button className="primary" onClick={() => onNavigate("Training")}><Play size={13}/> New training run</button></div>
    {!apiHealthy && <div className="api-warning"><CircleGauge size={14}/> Studio API is offline. Start FastAPI on port 8000.</div>}
    <div className="grid-4">
      <Stat icon={<Database size={15}/>} label="Projects" value={String(projects.length)} meta={projects.length ? "workspace projects" : "create a project"}/>
      <Stat icon={<BrainCircuit size={15}/>} label="Models" value={String(models.length)} meta={latest ? `${latest.algorithm} · ${latest.status}` : "no trained models"}/>
      <Stat icon={<ShieldCheck size={15}/>} label="Validation" value="Ready" meta="deployment validation available" green/>
      <Stat icon={<MonitorCog size={15}/>} label="Deployment" value="Not deployed" meta="choose a target"/>
    </div>
    <section className="section"><div className="section-head"><div className="section-title">Studio workflow</div><div className="section-link">Data → Train → Validate → Deploy → Monitor</div></div><div className="workflow">{workflows.map(([step,title,text]) => <div className="card workflow-card" key={step}><div className="step">{step}</div><div className="workflow-title">{title}</div><div className="workflow-text">{text}</div></div>)}</div></section>
    <section className="section activity"><div className="card"><div className="section-head"><div className="section-title">Recent model activity</div><div className="section-link">{models.length} models</div></div>{models.length ? models.slice(0,4).map((m) => <ActivityRow key={m.id} icon={<BrainCircuit size={14}/>} title={`${m.algorithm.toUpperCase()} · ${m.class_name}`} sub={m.run_name} badge={m.status}/>) : <div className="empty">No trained models registered for this project yet.</div>}</div>
      <div className="card"><div className="section-head"><div className="section-title">Runtime health</div><div className="section-link">Monitoring</div></div><HealthRow title="Studio API" sub="Python / FastAPI adapter" ok={apiHealthy}/><HealthRow title="Data drift" sub="Existing observer-only monitor" ok/><HealthRow title="AnomaVision core" sub="Training and inference engine" ok/></div></section>
  </>;
}

function ProjectsPage({ projects, selectedProject, onSelect, onCreated }: { projects: Project[]; selectedProject: string; onSelect: (id:string)=>void; onCreated:()=>Promise<void> }) {
  const [name,setName]=useState(""); const [description,setDescription]=useState(""); const [algorithm,setAlgorithm]=useState("patchcore"); const [error,setError]=useState(""); const [busy,setBusy]=useState(false);
  async function create() {
    setError(""); if (!name.trim()) { setError("Project name is required."); return; }
    setBusy(true); try {
      const r=await fetch(`${API_BASE}/api/projects`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name,description,algorithm})});
      const data=await r.json(); if(!r.ok) throw new Error(data.detail || "Could not create project");
      setName("");setDescription("");await onCreated();onSelect(data.id);
    } catch(e){setError(e instanceof Error?e.message:"Could not create project");} finally{setBusy(false);}
  }
  return <><div className="page-head"><div><div className="eyebrow">Workspace</div><h1>Projects</h1><p className="subtitle">Create and manage isolated anomaly-detection workspaces.</p></div></div>
    <div className="two-column"><section className="card"><div className="section-title">Create project</div><p className="form-help">A project keeps datasets, models, experiments and deployments together.</p>
      <label>Project name<input value={name} onChange={e=>setName(e.target.value)} placeholder="e.g. Bottle Inspection"/></label>
      <label>Algorithm<select value={algorithm} onChange={e=>setAlgorithm(e.target.value)}><option value="patchcore">PatchCore</option><option value="padim">PaDiM</option><option value="efficientad">EfficientAD</option></select></label>
      <label>Description<input value={description} onChange={e=>setDescription(e.target.value)} placeholder="Optional project description"/></label>
      {error && <div className="form-error">{error}</div>}<button className="primary full" onClick={create} disabled={busy}>{busy?"Creating…":"Create project"}</button>
    </section><section><div className="section-head"><div className="section-title">Your projects</div><div className="section-link">{projects.length} total</div></div>
      {projects.length===0?<div className="card empty">No projects yet. Create your first workspace.</div>:projects.map(p=><button key={p.id} className={`project-card ${p.id===selectedProject?"selected":""}`} onClick={()=>onSelect(p.id)}><div><strong>{p.name}</strong><span>{p.description || "No description"}</span></div><div className="project-meta"><b>{p.algorithm}</b><small>{p.status}</small></div></button>)}</section></div></>;
}

function DatasetsPage({ project }: { project?: Project }) {
  const [path,setPath]=useState(""); const [report,setReport]=useState<DatasetReport|null>(null); const [error,setError]=useState(""); const [busy,setBusy]=useState(false);
  async function inspect(){if(!project){setError("Select a project first.");return;}if(!path.trim()){setError("Dataset path is required.");return;}setBusy(true);setError("");try{const r=await fetch(`${API_BASE}/api/projects/${project.id}/datasets/inspect`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path,recursive:true})});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Dataset inspection failed");setReport(d);}catch(e){setError(e instanceof Error?e.message:"Dataset inspection failed");}finally{setBusy(false);}}
  return <><div className="page-head"><div><div className="eyebrow">Data</div><h1>Datasets</h1><p className="subtitle">Inspect an existing local image folder before training. Images are not copied.</p></div></div>
    {!project?<div className="card empty">Select or create a project first.</div>:<><div className="card dataset-form"><div><label>Local dataset path<input value={path} onChange={e=>setPath(e.target.value)} placeholder="C:\data\bottle"/></label></div><button className="primary" onClick={inspect} disabled={busy}>{busy?"Inspecting…":"Inspect dataset"}</button></div>
    {error&&<div className="form-error">{error}</div>}{report&&<><div className="grid-4 section"><Stat icon={<Database size={15}/>} label="Images" value={String(report.image_count)} meta={`${report.valid_count} valid`}/><Stat icon={<ShieldCheck size={15}/>} label="Failed" value={String(report.failed_count)} meta="image read failures"/><Stat icon={<Sparkles size={15}/>} label="Duplicates" value={String(report.duplicate_count)} meta="duplicate images"/><Stat icon={<Box size={15}/>} label="Resolutions" value={String(Object.keys(report.resolutions).length)} meta="unique sizes"/></div><div className="card section"><div className="section-title">Resolution distribution</div>{Object.keys(report.resolutions).length?Object.entries(report.resolutions).map(([resolution,count])=><div className="resolution-row" key={resolution}><span>{resolution}</span><b>{count}</b></div>):<div className="empty">No valid images found.</div>}</div></>}</>}</>;
}

function TrainingPage({ project, onFinished }: { project?: Project; onFinished:()=>Promise<void> }) {
  const [dataset,setDataset]=useState("");const [algorithm,setAlgorithm]=useState(project?.algorithm||"patchcore");const [className,setClassName]=useState("default");const [batch,setBatch]=useState("");const [backbone,setBackbone]=useState("");const [resize,setResize]=useState("224,224");const [result,setResult]=useState<Record<string,unknown>|null>(null);const [error,setError]=useState("");const [busy,setBusy]=useState(false);
  useEffect(()=>{if(project)setAlgorithm(project.algorithm)},[project]);
  async function train(){if(!project){setError("Select a project first.");return;}setBusy(true);setError("");setResult(null);try{const body:{dataset_path:string;algorithm:string;class_name:string;batch_size?:number;resize?:number[];backbone?:string}={dataset_path:dataset,algorithm,class_name};if(batch)body.batch_size=Number(batch);if(backbone)body.backbone=backbone;const dims=resize.split(",").map(Number);if(dims.length===2&&dims.every(Number.isFinite))body.resize=dims;const r=await fetch(`${API_BASE}/api/projects/${project.id}/training`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Training failed");setResult(d);await onFinished();}catch(e){setError(e instanceof Error?e.message:"Training failed")}finally{setBusy(false)}}
  return <><div className="page-head"><div><div className="eyebrow">Experiments</div><h1>Training</h1><p className="subtitle">Launch the existing AnomaVision training engine with a reproducible Studio configuration.</p></div></div>{!project?<div className="card empty">Select a project first.</div>:<div className="card form-grid"><label>Dataset path<input value={dataset} onChange={e=>setDataset(e.target.value)} placeholder="C:\data\bottle"/></label><label>Algorithm<select value={algorithm} onChange={e=>setAlgorithm(e.target.value)}><option>patchcore</option><option>padim</option><option>efficientad</option></select></label><label>Class name<input value={className} onChange={e=>setClassName(e.target.value)}/></label><label>Image size<input value={resize} onChange={e=>setResize(e.target.value)} placeholder="224,224"/></label><label>Batch size<input value={batch} onChange={e=>setBatch(e.target.value)} placeholder="Use config default"/></label><label>Backbone<input value={backbone} onChange={e=>setBackbone(e.target.value)} placeholder="Use config default"/></label><div className="form-actions"><button className="primary" onClick={train} disabled={busy}>{busy?"Training…":"Start training"}</button></div>{error&&<div className="form-error">{error}</div>}{result&&<pre className="result-box">{JSON.stringify(result,null,2)}</pre>}</div>}</>;
}

function ModelsPage({models,onRefresh}:{models:Model[];onRefresh:()=>Promise<void>}) {
  return <><div className="page-head"><div><div className="eyebrow">Artifacts</div><h1>Models</h1><p className="subtitle">Browse models produced by the existing AnomaVision training pipeline.</p></div><button className="secondary" onClick={onRefresh}><Activity size={13}/> Refresh</button></div><div className="card">{models.length?models.map(m=><div className="row" key={m.id}><div className="row-main"><div className="icon-box"><BrainCircuit size={14}/></div><div><div className="row-title">{m.algorithm.toUpperCase()} · {m.class_name}</div><div className="row-sub">{m.run_name}</div></div></div><div className="badge">{m.status}</div></div>):<div className="empty">No models for the selected project yet. Run training first.</div>}</div></>;
}

function DeploymentsPage({project,models}:{project?:Project;models:Model[]}) {
  const [target,setTarget]=useState("onnx"); const [modelId,setModelId]=useState(models[0]?.id||"");
  const [result,setResult]=useState<Record<string,any>|null>(null); const [error,setError]=useState(""); const [busy,setBusy]=useState(false);
  useEffect(()=>{setModelId(models[0]?.id||"")},[models]);
  async function deploy(){
    if(!project){setError("Select a project first.");return} if(!modelId){setError("Select a model first.");return}
    setBusy(true);setError("");setResult(null);
    try{const r=await fetch(`${API_BASE}/api/projects/${project.id}/deployments`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_id:modelId,target,runs:5,warmup_runs:1})});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Deployment failed");setResult(d)}
    catch(e){setError(e instanceof Error?e.message:"Deployment failed")}finally{setBusy(false)}
  }
  const checks=result?.validation?.checks as Record<string,boolean>|undefined;
  return <><div className="page-head"><div><div className="eyebrow">Production</div><h1>Deployments</h1><p className="subtitle">Export and validate a trained model using AnomaVision's existing deployment pipeline.</p></div></div>
    {!project?<div className="card empty">Select a project first.</div>:<div className="two-column">
      <section className="card"><div className="section-title">Create deployment</div><p className="form-help">Studio only orchestrates export and validation. The model and algorithm implementation stay unchanged.</p>
        <label>Model<select value={modelId} onChange={e=>setModelId(e.target.value)}>{models.length?models.map(m=><option key={m.id} value={m.id}>{m.algorithm} · {m.class_name} · {m.run_name}</option>):<option value="">No trained models</option>}</select></label>
        <label>Target<select value={target} onChange={e=>setTarget(e.target.value)}><option value="cpu">CPU validation</option><option value="onnx">ONNX</option><option value="openvino">OpenVINO</option><option value="tensorrt">TensorRT</option><option value="torchscript">TorchScript</option><option value="hailo">Hailo — manual flow</option><option value="kv260">KV260 — manual flow</option></select></label>
        {error&&<div className="form-error">{error}</div>}
        <button className="primary full" onClick={deploy} disabled={busy||!models.length}>{busy?"Exporting and validating…":"Export & validate"}</button>
      </section>
      <section><div className="card"><div className="section-head"><div className="section-title">Validation result</div>{result&&<div className="badge">{result.status}</div>}</div>
        {!result?<div className="empty">Choose a model and target, then run validation.</div>:<>
          <div className="deployment-result"><strong>{result.ready_for_deployment?"Ready for deployment":"Validation failed"}</strong><span>{result.artifact}</span></div>
          <div className="check-list">{checks&&Object.entries(checks).map(([name,ok])=><div className="check-row" key={name}><span>{ok?"✓":"✕"} {name.replaceAll("_"," ")}</span><b>{ok?"PASS":"FAIL"}</b></div>)}</div>
          {result.validation?.performance?.latency_ms!=null&&<div className="perf-grid"><Stat icon={<CircleGauge size={15}/>} label="Latency" value={`${result.validation.performance.latency_ms} ms`} meta={`${result.validation.performance.fps} FPS`}/><Stat icon={<ShieldCheck size={15}/>} label="Artifact" value={String(result.validation.format).toUpperCase()} meta="validated format"/></div>}
        </>}
      </div></section>
    </div>}
  </>;
}

function Placeholder({page}:{page:Page}) { const descriptions:Record<Page,string>={Overview:"",Projects:"",Datasets:"",Training:"",Models:"",Deployments:"Export and validate models for production targets without changing the algorithm core.",Live:"Inspect camera or stream inference using the existing AnomaVision runtime.",Monitoring:"Track runtime health, latency and production data drift."};return <><div className="page-head"><div><div className="eyebrow">Workspace</div><h1>{page}</h1><p className="subtitle">{descriptions[page]}</p></div><button className="secondary"><SlidersHorizontal size={13}/> Configure</button></div><div className="card placeholder"><div className="icon-box"><Sparkles size={18}/></div><div><b>Connected to the Studio architecture</b><p className="subtitle">This view is ready to consume the same Python services through the Studio API. No ML logic is duplicated in the frontend.</p></div></div></>; }

function Stat({icon,label,value,meta,green}:{icon:React.ReactNode;label:string;value:string;meta:string;green?:boolean}){return <div className="card"><div className="stat-label">{icon}<span>{label}</span></div><div className="stat-value">{value}</div><div className="stat-meta">{green&&<span className="status-dot"/>}{meta}</div></div>}
function ActivityRow({icon,title,sub,badge}:{icon:React.ReactNode;title:string;sub:string;badge:string}){return <div className="row"><div className="row-main"><div className="icon-box">{icon}</div><div><div className="row-title">{title}</div><div className="row-sub">{sub}</div></div></div><div className="badge">{badge}</div></div>}
function HealthRow({title,sub,ok}:{title:string;sub:string;ok:boolean}){return <div className="row"><div className="row-main"><div className="icon-box"><CircleGauge size={14}/></div><div><div className="row-title">{title}</div><div className="row-sub">{sub}</div></div></div><div className="badge"><span className={`status-dot ${ok?"":"offline-dot"}`}/>{ok?"Healthy":"Offline"}</div></div>}
