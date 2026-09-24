"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, Box, BrainCircuit, CircleGauge, Database, FlaskConical,
  LayoutDashboard, MonitorCog, Play, Rocket, Settings2, ShieldCheck,
  SlidersHorizontal, Sparkles, Wifi, FolderOpen, Menu, X, ChevronDown,
  HelpCircle
} from "lucide-react";

type Page = "Overview" | "Projects" | "Datasets" | "Training" | "Models" | "Deployments" | "Live" | "Monitoring" | "Settings";
type Project = { id: string; name: string; description?: string; algorithm: string; status: string; };
type Model = { id: string; algorithm: string; class_name: string; run_name: string; status: string; path?: string; };
type DatasetReport = {
  image_count: number; valid_count: number; failed_count: number; duplicate_count: number;
  resolutions: Record<string, number>; failures?: { path: string; error: string }[];
};
type Catalog = { algorithms: Record<string, unknown>; deployment_targets: Record<string, unknown> };

const API_BASE = process.env.NEXT_PUBLIC_STUDIO_API_URL ?? "http://localhost:8000";
const navGroups: { label: string; items: { label: Page; icon: React.ElementType }[] }[] = [
  { label: "Workspace", items: [
    { label: "Overview", icon: LayoutDashboard },
    { label: "Projects", icon: Box },
  ]},
  { label: "Build & test", items: [
    { label: "Datasets", icon: Database },
    { label: "Training", icon: BrainCircuit },
    { label: "Models", icon: FlaskConical },
    { label: "Deployments", icon: Rocket },
    { label: "Live", icon: Wifi },
  ]},
  { label: "Operate", items: [
    { label: "Monitoring", icon: Activity },
  ]},
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
  const [deploymentModelId, setDeploymentModelId] = useState("");
  const [apiHealthy, setApiHealthy] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  async function loadProjects() {
    try {
      const [health, response] = await Promise.all([
        fetch(`${API_BASE}/api/health`, { cache: "no-store" }),
        fetch(`${API_BASE}/api/projects`, { cache: "no-store" }),
      ]);
      if (!health.ok || !response.ok) throw new Error();
      const data = (await response.json()) as Project[];
      setProjects(data);
      setSelectedProject((current) =>
        data.some((item) => item.id === current) ? current : (data[0]?.id || "")
      );
      setApiHealthy(true);
    } catch {
      setApiHealthy(false);
    }
  }

  async function loadModels(projectId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/projects/${projectId}/models`, { cache: "no-store" });
      setModels(response.ok ? ((await response.json()) as Model[]) : []);
    } catch {
      setModels([]);
    }
  }

  useEffect(() => { void loadProjects(); }, []);
  useEffect(() => {
    if (selectedProject) void loadModels(selectedProject);
    else setModels([]);
  }, [selectedProject]);

  useEffect(() => {
    setMobileNavOpen(false);
  }, [page]);

  const project = useMemo(
    () => projects.find((item) => item.id === selectedProject),
    [projects, selectedProject]
  );

  function navigate(next: Page) {
    setPage(next);
    setMobileNavOpen(false);
  }

  return (
    <div className="studio-shell">
      {mobileNavOpen && <button className="mobile-nav-backdrop" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)} />}

      <aside className={`sidebar ${mobileNavOpen ? "mobile-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">AV</div>
          <div>
            <div className="brand-name">AnomaVision</div>
            <div className="brand-sub">Studio · workspace</div>
          </div>
          <button className="mobile-close" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)}>
            <X size={17}/>
          </button>
        </div>

        <div className="workspace-context">
          <span>ACTIVE PROJECT</span>
          <strong title={project?.name}>{project?.name || "No project selected"}</strong>
          <small>{project ? "Ready to work" : "Create a project to begin"}</small>
        </div>

        <nav aria-label="Studio navigation">
          {navGroups.map((group) => (
            <div key={group.label}>
              <div className="nav-label">{group.label}</div>
              {group.items.map(({ label, icon: Icon }) => (
                <button
                  key={label}
                  className={`nav-item ${page === label ? "active" : ""}`}
                  onClick={() => navigate(label)}
                  aria-current={page === label ? "page" : undefined}
                >
                  <Icon size={15} strokeWidth={1.8}/>
                  <span>{label}</span>
                </button>
              ))}
            </div>
          ))}
          <div className="nav-label">System</div>
          <button className={`nav-item ${page === "Settings" ? "active" : ""}`} onClick={() => navigate("Settings")} aria-current={page === "Settings" ? "page" : undefined}>
            <Settings2 size={15} strokeWidth={1.8}/><span>Settings</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="storage">
            <strong><span className={`status-dot ${apiHealthy ? "" : "offline-dot"}`}/>{apiHealthy ? "Studio connected" : "Studio offline"}</strong>
            <span>{apiHealthy ? "Workspace services are available." : "Start the Studio API to continue."}</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar-left">
            <button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileNavOpen(true)}>
              <Menu size={18}/>
            </button>
            <div className="breadcrumbs">
              <span>AnomaVision</span><b>/</b><strong>{page}</strong>
            </div>
          </div>

          <div className="topbar-actions">
            <label className="project-select-wrap">
              <span className="sr-only">Active project</span>
              <select className="project-switcher" value={selectedProject} onChange={(e) => setSelectedProject(e.target.value)} aria-label="Select active project">
                {projects.length === 0 && <option value="">No projects</option>}
                {projects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
              <ChevronDown size={13} aria-hidden="true"/>
            </label>
            <div className={`connection-pill ${apiHealthy ? "online" : "offline"}`}>
              <span className="status-dot"/>{apiHealthy ? "Connected" : "Offline"}
            </div>
            <button className="help-button" aria-label="Studio help" title="Studio help"><HelpCircle size={16}/></button>
          </div>
        </header>

        <div className="content">
          <div className="workspace-strip">
            <div>
              <span>ACTIVE WORKSPACE</span>
              <strong>{project?.name || "No project selected"}</strong>
            </div>
            <div className="workspace-strip-status">
              <span className={`status-dot ${apiHealthy ? "" : "offline-dot"}`}/>
              {apiHealthy ? "Studio ready" : "API unavailable"}
            </div>
          </div>

          {page === "Overview" && <Overview project={project} projects={projects} models={models} apiHealthy={apiHealthy} onNavigate={navigate}/>}
          {page === "Projects" && <ProjectsPage projects={projects} selectedProject={selectedProject} onSelect={setSelectedProject} onCreated={loadProjects}/>}
          {page === "Datasets" && <DatasetsPage project={project}/>}
          {page === "Training" && <TrainingPage project={project} onFinished={() => loadModels(selectedProject)}/>}
          {page === "Models" && <ModelsPage models={models} onRefresh={() => loadModels(selectedProject)} onNavigate={navigate} onDeploy={(id) => { setDeploymentModelId(id); navigate("Deployments"); }}/>}
          {page === "Deployments" && <DeploymentsPage project={project} models={models} initialModelId={deploymentModelId}/>}
          {page === "Monitoring" && <MonitoringPage project={project}/>}
          {page === "Live" && <LivePage apiHealthy={apiHealthy}/>}
          {page === "Settings" && <SettingsPage apiHealthy={apiHealthy}/>}
        </div>
      </main>
    </div>
  );
}

function Overview({ project, projects, models, apiHealthy, onNavigate }: { project?: Project; projects: Project[]; models: Model[]; apiHealthy: boolean; onNavigate: (p: Page) => void }) {
  const latest = models[0];
  const workflowPages: Record<string, Page | undefined> = {
    Dataset: "Datasets",
    Train: "Training",
    Deploy: "Deployments",
    Monitor: "Monitoring",
  };
  return <>
    <div className="page-head">
      <div>
        <div className="eyebrow">Workspace overview</div>
        <h1>{project?.name ?? "Welcome to AnomaVision Studio"}</h1>
        <p className="subtitle">{project?.description || "Build, validate and run industrial anomaly detection from one place."}</p>
      </div>
      <button className="primary" onClick={() => onNavigate(project ? "Training" : "Projects")}>
        <Play size={13}/>{project ? "Start training" : "Create project"}
      </button>
    </div>

    {!project && (
      <div className="welcome-card">
        <div className="welcome-icon"><Sparkles size={19}/></div>
        <div>
          <strong>Start with a project</strong>
          <p>Create a workspace, inspect your images, then train and validate your first anomaly detector.</p>
        </div>
        <button className="secondary" onClick={() => onNavigate("Projects")}>Create project</button>
      </div>
    )}

    {!apiHealthy && <div className="api-warning"><CircleGauge size={14}/> Studio API is offline. Start FastAPI on port 8000.</div>}

    <div className="grid-4">
      <Stat icon={<Database size={15}/>} label="Projects" value={String(projects.length)} meta={projects.length ? "workspaces available" : "create your first workspace"}/>
      <Stat icon={<BrainCircuit size={15}/>} label="Models" value={String(models.length)} meta={latest ? `${latest.algorithm} · ${latest.status}` : "train a model to get started"}/>
      <Stat icon={<ShieldCheck size={15}/>} label="Validation" value={latest ? "Available" : "—"} meta={latest ? "validate before deployment" : "needs a trained model"} green={Boolean(latest)}/>
      <Stat icon={<MonitorCog size={15}/>} label="Deployment" value="Ready" meta="choose a target when validated"/>
    </div>

    <section className="section">
      <div className="section-head">
        <div>
          <div className="section-title">Your workflow</div>
          <div className="subtitle">A simple path from images to production</div>
        </div>
        <div className="section-link">Data → Train → Validate → Deploy → Monitor</div>
      </div>
      <div className="workflow">
        {workflows.map(([step,title,text]) => {
          const target = workflowPages[title];
          return target
            ? <button className="card workflow-card workflow-action" key={step} onClick={() => onNavigate(target)}>
                <div className="step">{step}</div>
                <div className="workflow-title">{title}<span className="workflow-arrow">→</span></div>
                <div className="workflow-text">{text}</div>
              </button>
            : <div className="card workflow-card" key={step}>
                <div className="step">{step}</div>
                <div className="workflow-title">{title}</div>
                <div className="workflow-text">{text}</div>
              </div>;
        })}
      </div>
    </section>

    <section className="section activity">
      <div className="card">
        <div className="section-head">
          <div><div className="section-title">Recent models</div><div className="subtitle">Your latest trained artifacts</div></div>
          <button className="text-button" onClick={() => onNavigate("Models")}>View all →</button>
        </div>
        {models.length
          ? models.slice(0,4).map((m) => <ActivityRow key={m.id} icon={<BrainCircuit size={14}/>} title={`${m.algorithm.toUpperCase()} · ${m.class_name}`} sub={m.run_name} badge={m.status}/>)
          : <div className="empty-state"><strong>No models yet</strong><span>Train your first model to see it here.</span><button className="secondary" onClick={() => onNavigate("Training")}>Start training</button></div>}
      </div>

      <div className="card">
        <div className="section-head">
          <div><div className="section-title">System status</div><div className="subtitle">Studio services at a glance</div></div>
          <button className="text-button" onClick={() => onNavigate("Monitoring")}>Monitoring →</button>
        </div>
        <HealthRow title="Studio API" sub="FastAPI workspace service" ok={apiHealthy}/>
        <HealthRow title="AnomaVision core" sub="Training and inference engine" ok/>
        <HealthRow title="Drift monitoring" sub="Production observer" ok/>
      </div>
    </section>
  </>;
}

function ProjectsPage({ projects, selectedProject, onSelect, onCreated }: { projects: Project[]; selectedProject: string; onSelect: (id:string)=>void; onCreated:()=>Promise<void> }) {
  const [name,setName]=useState(""); const [description,setDescription]=useState(""); const [algorithm,setAlgorithm]=useState("patchcore"); const [error,setError]=useState(""); const [busy,setBusy]=useState(false);
  async function create() {
    setError(""); if (!name.trim()) { setError("Give your project a name first."); return; }
    setBusy(true); try {
      const r=await fetch(`${API_BASE}/api/projects`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:name.trim(),description:description.trim(),algorithm})});
      const data=await r.json(); if(!r.ok) throw new Error(data.detail || "Could not create project");
      setName("");setDescription("");await onCreated();onSelect(data.id);
    } catch(e){setError(e instanceof Error?e.message:"Could not create project");} finally{setBusy(false);}
  }
  return <>
    <div className="page-head"><div><div className="eyebrow">Workspace</div><h1>Projects</h1><p className="subtitle">Each project keeps its data, models and deployments together.</p></div></div>
    <div className="project-intro"><div><strong>{projects.length ? "Choose a workspace or create a new one." : "Your first project starts here."}</strong><span>{projects.length ? "Select a project to make it the active workspace." : "Set the method now — you can continue with the existing AnomaVision workflow."}</span></div><div className="project-count"><b>{projects.length}</b><span>{projects.length===1?"project":"projects"}</span></div></div>
    <div className="two-column project-layout">
      <section className="card project-create-card">
        <div className="project-create-head"><div className="icon-box"><Sparkles size={16}/></div><div><div className="section-title">Create project</div><p className="form-help">Start a clean workspace for one inspection task or product.</p></div></div>
        <label>Project name<input autoFocus value={name} onChange={e=>setName(e.target.value)} placeholder="e.g. Bottle Inspection" onKeyDown={e=>{if(e.key==="Enter"&&!busy)void create()}}/></label>
        <label>Algorithm<select value={algorithm} onChange={e=>setAlgorithm(e.target.value)}><option value="patchcore">PatchCore · fast industrial baseline</option><option value="padim">PaDiM · feature distribution</option><option value="efficientad">EfficientAD · efficient detection</option></select></label>
        <label>Description<input value={description} onChange={e=>setDescription(e.target.value)} placeholder="Optional · e.g. bottle surface inspection"/></label>
        {error && <div className="form-error">{error}</div>}
        <button className="primary full" onClick={create} disabled={busy}><Play size={13}/>{busy?"Creating…":"Create project"}</button>
      </section>
      <section>
        <div className="section-head"><div><div className="section-title">Your projects</div><div className="subtitle">Select a workspace to continue.</div></div><div className="section-link">{projects.length} total</div></div>
        {projects.length===0
          ? <div className="card project-empty"><div className="dataset-empty-icon"><Box size={19}/></div><strong>No projects yet</strong><span>Create your first workspace and then move to Dataset.</span></div>
          : <div className="project-list">{projects.map(p=><button key={p.id} className={`project-card ${p.id===selectedProject?"selected":""}`} onClick={()=>onSelect(p.id)}>
              <div className="project-card-main"><div className="project-symbol"><Box size={15}/></div><div><strong>{p.name}</strong><span>{p.description || "No description yet"}</span></div></div>
              <div className="project-meta"><b>{p.algorithm}</b><small><span className="status-dot"/> {p.id===selectedProject?"Active":"Ready"}</small></div>
            </button>)}</div>}
      </section>
    </div>
    <div className="card project-next"><div><strong>Next step</strong><span>{selectedProject ? "Your workspace is ready. Check the dataset before training." : "Create or select a project to unlock the workflow."}</span></div><div className="project-flow"><span className={selectedProject?"done":""}>1 · Project</span><span>→</span><span>2 · Dataset</span><span>→</span><span>3 · Train</span></div></div>
  </>;
}
function DatasetsPage({ project }: { project?: Project }) {
  const [path,setPath]=useState("");
  const [configPath,setConfigPath]=useState("");
  const [configClass,setConfigClass]=useState("default");
  const [configLoaded,setConfigLoaded]=useState(false);
  const [report,setReport]=useState<DatasetReport|null>(null);
  useEffect(()=>{
    fetch(API_BASE+"/api/config",{cache:"no-store"})
      .then(r=>r.ok?r.json():null)
      .then(d=>{
        if(d?.dataset_path){setConfigPath(String(d.dataset_path));setPath(current=>current||String(d.dataset_path));}
        if(d?.class_name)setConfigClass(String(d.class_name));
      })
      .catch(()=>{})
      .finally(()=>setConfigLoaded(true));
  },[]);
  const [error,setError]=useState("");
  const [busy,setBusy]=useState(false);
  const [pickerBusy,setPickerBusy]=useState(false);

  async function chooseFolder(){
    if(!project){setError("Select a project first.");return;}
    setPickerBusy(true);setError("");
    try{
      const r=await fetch(API_BASE+"/api/dataset/pick-folder",{cache:"no-store"});
      const d=await r.json().catch(()=>({}));
      if(!r.ok)throw new Error(d.detail||"Could not open the folder dialog");
      if(!d.path){
        setError("No folder was selected. Choose a dataset folder and try again.");
        return;
      }
      setPath(String(d.path));setReport(null);
    }catch(e){
      setError(e instanceof Error?e.message:"Could not open the folder dialog");
    }finally{setPickerBusy(false);}
  }

  async function inspect(){
    if(!project){setError("Select a project first.");return;}
    if(!path.trim()){setError("Choose a local image folder first.");return;}
    setBusy(true);setError("");
    try{
      const r=await fetch(`${API_BASE}/api/projects/${project.id}/datasets/inspect`,{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({path,recursive:true})
      });
      const d=await r.json();
      if(!r.ok)throw new Error(d.detail||"Dataset inspection failed");
      setReport(d);
    }catch(e){
      setError(e instanceof Error?e.message:"Dataset inspection failed");
    }finally{setBusy(false);}
  }

  const issueCount=(report?.failed_count||0)+(report?.duplicate_count||0);
  const ready=Boolean(report && report.image_count>0 && report.failed_count===0);
  const resolutionEntries=report?Object.entries(report.resolutions).sort((a,b)=>b[1]-a[1]):[];
  const commonResolution=resolutionEntries[0]?.[0]||"—";

  return <>
    <div className="page-head">
      <div>
        <div className="eyebrow">Data readiness</div>
        <h1>Dataset</h1>
        <p className="subtitle">Check your images before training. Studio only inspects the folder — your images stay where they are.</p>
      </div>
    </div>

    {!project
      ? <div className="card empty-state dataset-empty">
          <div className="dataset-empty-icon"><Database size={20}/></div>
          <strong>Create or select a project first</strong>
          <span>Your dataset will be linked to the selected project.</span>
        </div>
      : <>
        <section className="card dataset-hero">
          <div className="dataset-hero-copy">
            <div className="dataset-icon"><Database size={18}/></div>
            <div>
              <div className="section-title">Check a local image folder</div>
              <p>Studio looks at image count, readable files, duplicates and image sizes before you start training.</p>
            </div>
          </div>
          <div className="dataset-config-card">
            <div>
              <span className="dataset-config-label">CONFIGURED DATASET</span>
              <strong>{configPath || (configLoaded ? "No dataset_path configured" : "Loading config…")}</strong>
              {configClass && <small>Class: <b>{configClass}</b> · source: config.yml</small>}
            </div>
            <div className="dataset-config-actions">
              {configPath && path!==configPath && <button className="secondary" onClick={()=>{setPath(configPath);setReport(null);}}>Use config folder</button>}
              <button className="secondary" onClick={chooseFolder} disabled={pickerBusy}>
                <FolderOpen size={13}/>{pickerBusy?"Opening…":"Choose another folder"}
              </button>
            </div>
          </div>
          <div className="dataset-input-row">
            <label>
              Dataset folder
              <div className="dataset-path-control">
                <input value={path} onChange={e=>{setPath(e.target.value);setReport(null)}} placeholder={configLoaded?"Enter or select a local dataset folder":"Loading config…"} />
              </div>
            </label>
            <button className="primary" onClick={inspect} disabled={busy||!path.trim()}>
              <ShieldCheck size={13}/>{busy?"Checking…":"Check dataset"}
            </button>
          </div>
          <div className="dataset-source">
            <span className="badge">{configPath && path===configPath ? "From config.yml" : "Custom folder"}</span>
            <span>{path ? "Selected: "+path : "Choose a folder to continue."}</span>
            {configClass && <span>Class: <strong>{configClass}</strong></span>}
          </div>
          <div className="dataset-tip">Studio uses the canonical dataset_path from config.yml by default. Choose another folder only when you intentionally want to override it for this check.</div>
        </section>

        {error&&<div className="form-error">{error}</div>}

        {!report && !error && <div className="card dataset-guide">
          <div className="dataset-guide-step"><span>1</span><div><strong>Choose your image folder</strong><p>Use a local path such as <code>C:\data\bottle</code>.</p></div></div>
          <div className="dataset-guide-step"><span>2</span><div><strong>Check readiness</strong><p>Studio will scan the images without copying or modifying them.</p></div></div>
          <div className="dataset-guide-step"><span>3</span><div><strong>Start training</strong><p>Once the data looks good, continue to the Training step.</p></div></div>
        </div>}

        {report&&<>
          <div className={`dataset-readiness ${ready?"ready":"review"}`}>
            <div className="dataset-readiness-icon">{ready?<ShieldCheck size={19}/>:<CircleGauge size={19}/>}</div>
            <div>
              <strong>{ready?"Ready for training":"Review your dataset first"}</strong>
              <span>{ready
                ? `${report.valid_count} readable images found. You can continue to training.`
                : report.image_count===0
                  ? "No readable images were found in this folder."
                  : `${report.failed_count} image(s) could not be read. Fix those files and check again.`}</span>
            </div>
          </div>

          <div className="grid-4 section">
            <Stat icon={<Database size={15}/>} label="Images" value={String(report.image_count)} meta={`${report.valid_count} readable`}/>
            <Stat icon={<ShieldCheck size={15}/>} label="Issues" value={String(report.failed_count)} meta={report.failed_count?"files need attention":"no read failures"} green={!report.failed_count}/>
            <Stat icon={<Sparkles size={15}/>} label="Duplicates" value={String(report.duplicate_count)} meta={report.duplicate_count?"review before training":"none found"} green={!report.duplicate_count}/>
            <Stat icon={<Box size={15}/>} label="Main size" value={commonResolution} meta={`${resolutionEntries.length} unique sizes`}/>
          </div>

          <div className="dataset-details section">
            <div className="section-head">
              <div><div className="section-title">Dataset details</div><div className="subtitle">A quick look at what Studio found.</div></div>
              <div className="section-link">{issueCount ? `${issueCount} item(s) to review` : "Looks clean"}</div>
            </div>
            <div className="card">
              {resolutionEntries.length
                ? <div className="resolution-list">
                    {resolutionEntries.map(([resolution,count])=><div className="resolution-row" key={resolution}><span>{resolution}</span><b>{count}</b></div>)}
                  </div>
                : <div className="empty">No valid images found.</div>}
              {report.failures&&report.failures.length>0&&<div className="dataset-failures">
                <div className="dataset-subtitle">Files that could not be read</div>
                {report.failures.slice(0,5).map((failure)=><div className="failure-row" key={failure.path}><span title={failure.path}>{failure.path}</span><small>{failure.error}</small></div>)}
                {report.failures.length>5&&<div className="dataset-more">+ {report.failures.length-5} more</div>}
              </div>}
            </div>
          </div>
        </>}
      </>}
  </>;
}

function TrainingPage({ project, onFinished }: { project?: Project; onFinished:()=>Promise<void> }) {
  const [dataset,setDataset]=useState("");
  const [algorithm,setAlgorithm]=useState(project?.algorithm||"patchcore");
  const [className,setClassName]=useState("");
  const [batch,setBatch]=useState("");
  const [backbone,setBackbone]=useState("");
  const [resize,setResize]=useState("");
  const [configLoaded,setConfigLoaded]=useState(false);
  const [result,setResult]=useState<Record<string,unknown>|null>(null);
  const [error,setError]=useState("");
  const [busy,setBusy]=useState(false);

  useEffect(()=>{if(project)setAlgorithm(project.algorithm)},[project]);

  useEffect(()=>{
    fetch(`${API_BASE}/api/config`,{cache:"no-store"})
      .then(r=>r.ok?r.json():null)
      .then(d=>{
        if(!d)return;
        if(d.dataset_path)setDataset(String(d.dataset_path));
        if(d.class_name)setClassName(String(d.class_name));
        if(Array.isArray(d.resize)&&d.resize.length===2)setResize(d.resize.join(","));
      })
      .catch(()=>{})
      .finally(()=>setConfigLoaded(true));
  },[]);

  async function train(){
    if(!project){setError("Select a project first.");return;}
    if(!dataset.trim()){setError("Add your dataset path first.");return;}
    setBusy(true);setError("");setResult(null);
    try{
      const body:{dataset_path:string;algorithm:string;class_name?:string;batch_size?:number;resize?:number[];backbone?:string}={dataset_path:dataset,algorithm};
      const selectedClass=className.trim();
      if(selectedClass) body.class_name=selectedClass;
      if(batch)body.batch_size=Number(batch);
      if(backbone)body.backbone=backbone;
      const dims=resize.split(",").map(Number);
      if(dims.length===2&&dims.every(Number.isFinite))body.resize=dims;
      const r=await fetch(`${API_BASE}/api/projects/${project.id}/training`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
      const d=await r.json();
      if(!r.ok)throw new Error(d.detail||"Training failed");
      setResult(d);await onFinished();
    }catch(e){setError(e instanceof Error?e.message:"Training failed")}
    finally{setBusy(false)}
  }

  return <>
    <div className="page-head">
      <div>
        <div className="eyebrow">Build a model</div>
        <h1>Training</h1>
        <p className="subtitle">Set up your experiment, then let the existing AnomaVision engine do the training.</p>
      </div>
    </div>
    {!project
      ? <div className="card empty-state"><strong>Select a project first</strong><span>Your training run will be saved inside the selected project.</span></div>
      : <>
        <div className="training-steps">
          <div className="training-step active"><span>1</span><div><strong>Data</strong><small>Choose images</small></div></div>
          <div className="training-line"/>
          <div className="training-step active"><span>2</span><div><strong>Method</strong><small>Choose detector</small></div></div>
          <div className="training-line"/>
          <div className="training-step"><span>3</span><div><strong>Run</strong><small>Start training</small></div></div>
        </div>

        <div className="two-column training-layout">
          <section className="card">
            <div className="section-title">1. Training data</div>
            <p className="form-help">Point Studio to the same local image folder you checked in Data Readiness.</p>
            <label>Dataset path<input value={dataset} onChange={e=>setDataset(e.target.value)} placeholder={configLoaded?"From config.yml":"Loading config…"}/></label>
            <label>Class name<input value={className} onChange={e=>setClassName(e.target.value)} placeholder={configLoaded?"From config.yml":"Loading config…"}/></label>
            <div className="config-field-note">{className ? <>Using <strong>{className}</strong> from {configLoaded ? "config.yml" : "the current setup"}.</> : <>Class name will be taken from <strong>config.yml</strong> if you leave it empty.</>}</div>
          </section>

          <section className="card">
            <div className="section-title">2. Detection method</div>
            <p className="form-help">Choose the anomaly detector for this experiment.</p>
            <div className="algorithm-options">
              {[
                ["patchcore","PatchCore","Strong local-feature baseline"],
                ["padim","PaDiM","Fast statistical baseline"],
                ["efficientad","EfficientAD","Lightweight industrial detector"]
              ].map(([value,title,desc])=>
                <button key={value} type="button" className={`algorithm-option ${algorithm===value?"selected":""}`} onClick={()=>setAlgorithm(value)}>
                  <span className="algorithm-radio">{algorithm===value?"✓":""}</span>
                  <span><strong>{title}</strong><small>{desc}</small></span>
                </button>
              )}
            </div>
          </section>
        </div>

        <section className="card training-advanced">
          <div className="section-head">
            <div><div className="section-title">Optional settings</div><div className="subtitle">Leave these empty to use the canonical config.yml values.</div></div>
            <div className="section-link">Config-aware</div>
          </div>
          <div className="form-grid">
            <label>Image size<input value={resize} onChange={e=>setResize(e.target.value)} placeholder="From config.yml"/></label>
            <label>Batch size<input value={batch} onChange={e=>setBatch(e.target.value)} placeholder="Use config default"/></label>
            <label>Backbone<input value={backbone} onChange={e=>setBackbone(e.target.value)} placeholder="Use config default"/></label>
          </div>
        </section>

        <div className="training-action">
          <div><strong>Ready to train?</strong><span>{algorithm.toUpperCase()} · {resize || "config.yml"} · {className || "config.yml class"}</span></div>
          <button className="primary" onClick={train} disabled={busy}><Play size={13}/>{busy?"Training…":"Start training"}</button>
        </div>

        {error&&<div className="form-error">{error}</div>}
        {result&&<section className="card training-result">
          <div className="section-head">
            <div><div className="section-title">Training completed</div><div className="subtitle">The model is now available in Models.</div></div>
            <div className="badge success-badge">Completed</div>
          </div>
          <div className="training-success">
            <div className="training-success-icon"><ShieldCheck size={18}/></div>
            <div><strong>Your model was created successfully.</strong><span>Open Models to review the artifact or continue to validation.</span></div>
            <button className="secondary" onClick={()=>window.scrollTo({top:0,behavior:"smooth"})}>Continue</button>
          </div>
          <div className="training-result-grid">
            <div><span>Algorithm</span><b>{String(result.algorithm ?? algorithm).toUpperCase()}</b></div>
            <div><span>Class</span><b>{String(result.class_name ?? className ?? "From config")}</b></div>
            <div><span>Run</span><b>{String(result.run_name ?? result.model_id ?? "Created")}</b></div>
            <div><span>Status</span><b>{String(result.status ?? "trained")}</b></div>
          </div>
          {(result.model_path || result.path) && <div className="artifact-box"><span>Model artifact</span><code>{String(result.model_path ?? result.path)}</code></div>}
          <details className="technical-details">
            <summary>Technical details</summary>
            <pre className="result-box">{JSON.stringify(result,null,2)}</pre>
          </details>
        </section>
      </>}
  </>;
}

function ModelsPage({models,onRefresh,onNavigate,onDeploy}:{models:Model[];onRefresh:()=>Promise<void>;onNavigate:(p:Page)=>void;onDeploy:(id:string)=>void}) {
  const trainedCount=models.filter(m=>m.status==="trained").length;
  const latest=models[0];

  return <>
    <div className="page-head">
      <div>
        <div className="eyebrow">Model registry</div>
        <h1>Models</h1>
        <p className="subtitle">Your trained models, ready to validate and move toward deployment.</p>
      </div>
      <button className="secondary" onClick={onRefresh}><Activity size={13}/> Refresh</button>
    </div>

    <div className="grid-4 section">
      <Stat icon={<BrainCircuit size={15}/>} label="Models" value={String(models.length)} meta={models.length ? "trained artifacts" : "train your first model"}/>
      <Stat icon={<ShieldCheck size={15}/>} label="Trained" value={String(trainedCount)} meta="available for validation" green={trainedCount > 0}/>
      <Stat icon={<FlaskConical size={15}/>} label="Latest method" value={latest?.algorithm ? latest.algorithm.toUpperCase() : "—"} meta={latest?.class_name || "no model yet"}/>
      <Stat icon={<Rocket size={15}/>} label="Next step" value={models.length ? "Deploy" : "Train"} meta={models.length ? "validate a model first" : "build your first artifact"}/>
    </div>

    <section className="section">
      <div className="section-head">
        <div>
          <div className="section-title">Trained artifacts</div>
          <div className="subtitle">Each model keeps the original AnomaVision training configuration and artifact.</div>
        </div>
      </div>

      {models.length ? <div className="model-list">
        {models.map((m,index) => {
          const artifact = m.path ? m.path.split(/[\\/]/).pop() : "model.pt";
          return <div className="card model-card" key={m.id}>
            <div className="model-main">
              <div className="model-icon"><BrainCircuit size={17}/></div>
              <div className="model-info">
                <div className="model-title">
                  <span>{m.algorithm.toUpperCase()}</span>
                  <span className="model-separator">·</span>
                  <span>{m.class_name}</span>
                  {index === 0 && <span className="latest-chip">Latest</span>}
                </div>
                <div className="model-run">{m.run_name}</div>
                <div className="model-meta"><span>Artifact</span><code>{artifact}</code><span>Status</span><b>{m.status}</b></div>
              </div>
            </div>
            <div className="model-actions">
              <span className="badge"><span className="status-dot"/>{m.status}</span>
              <button className="secondary" onClick={() => onDeploy(m.id)}><Rocket size={13}/> Validate & deploy</button>
            </div>
          </div>;
        })}
      </div> : <div className="card empty-state">
        <div className="dataset-empty-icon"><BrainCircuit size={20}/></div>
        <strong>No trained models yet</strong>
        <span>Train a model first. Once training finishes, its artifact will appear here with a direct deployment path.</span>
        <button className="primary" onClick={() => onNavigate("Training")}><Play size={13}/> Start training</button>
      </div>}
    </section>
  </>;
}

function DeploymentsPage({project,models,initialModelId}:{project?:Project;models:Model[];initialModelId?:string}) {
  const [target,setTarget]=useState("onnx");
  const [modelId,setModelId]=useState(models[0]?.id||"");
  const [result,setResult]=useState<Record<string,any>|null>(null);
  const [error,setError]=useState("");
  const [busy,setBusy]=useState(false);

  useEffect(()=>{setModelId(initialModelId || models[0]?.id || "")},[models,initialModelId]);

  const selected=models.find(m=>m.id===modelId);

  async function deploy(){
    if(!project){setError("Select a project first.");return}
    if(!modelId){setError("Select a model first.");return}
    setBusy(true);setError("");setResult(null);
    try{
      const r=await fetch(`${API_BASE}/api/projects/${project.id}/deployments`,{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({model_id:modelId,target,runs:5,warmup_runs:1})
      });
      const d=await r.json();
      if(!r.ok)throw new Error(d.detail||"Deployment failed");
      setResult(d);
    }catch(e){setError(e instanceof Error?e.message:"Deployment failed")}
    finally{setBusy(false)}
  }

  const checks=result?.validation?.checks as Record<string,boolean>|undefined;
  const passed=checks?Object.values(checks).filter(Boolean).length:0;
  const total=checks?Object.keys(checks).length:0;
  const performance=result?.validation?.performance;

  return <>
    <div className="page-head">
      <div>
        <div className="eyebrow">Production readiness</div>
        <h1>Deployments</h1>
        <p className="subtitle">Choose a target, export the trained artifact, then validate it with the existing AnomaVision pipeline.</p>
      </div>
    </div>

    {!project ? <div className="card empty-state">
      <div className="dataset-empty-icon"><Rocket size={20}/></div>
      <strong>Select a project first</strong>
      <span>Your project contains the models and deployment artifacts for this workflow.</span>
    </div> : <>
      <div className="deployment-steps">
        <div className="deployment-step active"><span>1</span><div><b>Select model</b><small>Choose a trained artifact</small></div></div>
        <div className="deployment-step"><span>2</span><div><b>Choose target</b><small>Pick the runtime format</small></div></div>
        <div className="deployment-step"><span>3</span><div><b>Validate</b><small>Check integrity and performance</small></div></div>
      </div>

      <div className="two-column">
        <section className="card deployment-config">
          <div className="section-title">Deployment setup</div>
          <p className="form-help">Studio orchestrates export and validation. Your model and algorithm implementation remain unchanged.</p>

          <label>Model
            <select value={modelId} onChange={e=>{setModelId(e.target.value);setResult(null)}} disabled={busy}>
              {models.length?models.map(m=><option key={m.id} value={m.id}>{m.algorithm.toUpperCase()} · {m.class_name} · {m.run_name}</option>):<option value="">No trained models</option>}
            </select>
          </label>

          {selected && <div className="selected-model">
            <div className="icon-box"><BrainCircuit size={14}/></div>
            <div><b>{selected.algorithm.toUpperCase()} · {selected.class_name}</b><span>{selected.run_name}</span></div>
            <span className="badge">{selected.status}</span>
          </div>}

          <div className="target-label">Deployment target</div>
          <div className="target-grid">
            {[
              ["cpu","CPU","Validate existing model"],
              ["onnx","ONNX","Portable runtime artifact"],
              ["openvino","OpenVINO","Intel / CPU deployment"],
              ["tensorrt","TensorRT","NVIDIA deployment"],
              ["torchscript","TorchScript","PyTorch runtime"],
              ["hailo","Hailo","Manual deployment flow"],
              ["kv260","KV260","Manual deployment flow"],
            ].map(([id,name,desc])=><button key={id} className={`target-card ${target===id?"selected":""}`} onClick={()=>{setTarget(id);setResult(null)}} disabled={busy}>
              <strong>{name}</strong><span>{desc}</span>
            </button>)}
          </div>

          {error&&<div className="form-error">{error}</div>}
          <button className="primary full" onClick={deploy} disabled={busy||!models.length}>
            <Rocket size={13}/>{busy?"Exporting and validating…":"Export & validate"}
          </button>
        </section>

        <section className="card deployment-result-card">
          <div className="section-head">
            <div><div className="section-title">Validation result</div><div className="subtitle">{result?"Latest run":"Results will appear here after validation."}</div></div>
            {result&&<div className={`badge ${result.ready_for_deployment?"success-badge":"failure-badge"}`}>{result.ready_for_deployment?"READY":"FAILED"}</div>}
          </div>

          {!result ? <div className="deployment-empty">
            <div className="deployment-empty-icon"><ShieldCheck size={22}/></div>
            <strong>Validate before deployment</strong>
            <span>Studio will export the selected target, run the existing validation checks, and report the result here.</span>
          </div> : <>
            <div className={`deployment-outcome ${result.ready_for_deployment?"ready":"failed"}`}>
              <div className="outcome-icon">{result.ready_for_deployment?"✓":"!"}</div>
              <div><strong>{result.ready_for_deployment?"Ready for deployment":"Validation failed"}</strong><span>{result.ready_for_deployment?"All required checks passed.":"Review the failed checks before using this artifact."}</span></div>
            </div>

            {checks&&<div className="validation-summary"><b>{passed}/{total}</b><span>validation checks passed</span></div>}

            <div className="check-list">{checks&&Object.entries(checks).map(([name,ok])=><div className="check-row" key={name}><span>{ok?"✓":"✕"} {name.replaceAll("_"," ")}</span><b>{ok?"PASS":"FAIL"}</b></div>)}</div>

            {performance?.latency_ms!=null&&<div className="perf-grid">
              <Stat icon={<CircleGauge size={15}/>} label="Latency" value={`${performance.latency_ms} ms`} meta={`${performance.fps} FPS`}/>
              <Stat icon={<ShieldCheck size={15}/>} label="Format" value={String(result.validation.format).toUpperCase()} meta="validated artifact"/>
            </div>}

            <div className="artifact-box"><span>Artifact</span><code>{result.artifact}</code></div>
          </>}
        </section>
      </div>
    </>}
  </>;
}

function MonitoringPage({project}:{project?:Project}) {
  const [summary,setSummary]=useState<Record<string,any>|null>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");

  async function load(){
    if(!project){setSummary(null);return}
    setBusy(true);setError("");
    try{
      const r=await fetch(`${API_BASE}/api/projects/${project.id}/monitoring`,{cache:"no-store"});
      const d=await r.json();
      if(!r.ok)throw new Error(d.detail||"Could not load monitoring");
      setSummary(d);
    }catch(e){setError(e instanceof Error?e.message:"Could not load monitoring")}
    finally{setBusy(false)}
  }

  useEffect(()=>{void load()},[project]);

  const latest=summary?.latest;
  const status=String(summary?.status||"no_data");
  const warnings=(latest?.warnings||[]) as string[];
  const driftScore=latest?.drift_score!=null?Number(latest.drift_score):null;
  const psi=latest?.psi!=null?Number(latest.psi):null;
  const healthy=status==="ok" || status==="healthy" || status==="no_data";
  const statusLabel=status==="no_data"?"No data":status.replaceAll("_"," ");

  return <>
    <div className="page-head">
      <div>
        <div className="eyebrow">Production health</div>
        <h1>Monitoring</h1>
        <p className="subtitle">Understand how the data seen by your model is changing over time.</p>
      </div>
      <button className="secondary" onClick={load} disabled={busy}><Activity size={13}/>{busy?"Refreshing…":"Refresh"}</button>
    </div>

    {!project ? <div className="card empty-state">
      <div className="dataset-empty-icon"><Activity size={20}/></div>
      <strong>Select a project first</strong>
      <span>Monitoring reports are stored inside each Studio project.</span>
    </div> : <>
      <div className={`monitor-status ${healthy?"healthy":"attention"}`}>
        <div className="monitor-status-icon"><span className="status-dot"/></div>
        <div>
          <strong>{latest ? (healthy?"Monitoring is healthy":"Review drift signals") : "Monitoring is ready"}</strong>
          <span>{latest ? `Latest report · ${warnings.length} warning${warnings.length===1?"":"s"} · ${summary?.report_count||0} stored reports` : "No stored drift report yet. Reports will appear here when monitoring produces them."}</span>
        </div>
        <div className="monitor-status-value">{statusLabel}</div>
      </div>

      {error&&<div className="form-error">{error}</div>}

      <div className="grid-4 section">
        <Stat icon={<Activity size={15}/>} label="Drift score" value={driftScore!=null?driftScore.toFixed(3):"—"} meta="combined drift signal"/>
        <Stat icon={<Database size={15}/>} label="PSI" value={psi!=null?psi.toFixed(3):"—"} meta="distribution shift"/>
        <Stat icon={<ShieldCheck size={15}/>} label="Warnings" value={String(warnings.length)} meta={warnings.length?"review latest signals":"no warnings reported"} green={!warnings.length}/>
        <Stat icon={<CircleGauge size={15}/>} label="Reports" value={String(summary?.report_count||0)} meta="stored monitoring reports"/>
      </div>

      <div className="two-column">
        <section className="card">
          <div className="section-head">
            <div><div className="section-title">Latest drift report</div><div className="subtitle">The most recent report produced by the existing monitoring engine.</div></div>
          </div>
          {latest ? <div className="check-list">
            <div className="check-row"><span>Reference samples</span><b>{latest.reference_samples}</b></div>
            <div className="check-row"><span>Current samples</span><b>{latest.current_samples}</b></div>
            <div className="check-row"><span>Feature dimensions</span><b>{latest.feature_dimensions}</b></div>
            <div className="check-row"><span>Mean shift</span><b>{Number(latest.mean_shift).toFixed(4)}</b></div>
            <div className="check-row"><span>Std shift</span><b>{Number(latest.std_shift).toFixed(4)}</b></div>
            <div className="check-row"><span>Cosine shift</span><b>{Number(latest.cosine_shift).toFixed(4)}</b></div>
          </div> : <div className="monitor-empty"><Activity size={18}/><strong>No report yet</strong><span>Run production monitoring to populate this view.</span></div>}
        </section>

        <section className="card">
          <div className="section-head">
            <div><div className="section-title">Signals to review</div><div className="subtitle">Warnings reported by the monitoring engine.</div></div>
          </div>
          {warnings.length ? <div className="warning-list monitor-warnings">{warnings.map((w:string)=><span className="warning-chip" key={w}>{w.replaceAll("_"," ")}</span>)}</div>
            : <div className="monitor-empty"><ShieldCheck size={18}/><strong>No warnings</strong><span>The latest stored report contains no warning signals.</span></div>}
        </section>
      </div>

      <div className="card monitoring-note">
        <ShieldCheck size={15}/>
        <div><strong>Existing monitoring engine</strong><span>Studio displays the stored drift reports; it does not change the underlying drift calculations or thresholds.</span></div>
      </div>
    </>}
  </>;
}

function LivePage({apiHealthy}:{apiHealthy:boolean}) {
  const [files,setFiles]=useState<File[]>([]); const [camera,setCamera]=useState(false); const [cameraBusy,setCameraBusy]=useState(false); const [cameraAuto,setCameraAuto]=useState(false); const [cameraError,setCameraError]=useState(""); const [cameraFps,setCameraFps]=useState(0); const [cameraFrames,setCameraFrames]=useState(0); const [cameraInterval,setCameraInterval]=useState(500); const videoRef=useRef<HTMLVideoElement>(null); const streamRef=useRef<MediaStream|null>(null); const cameraLoopRef=useRef<number|null>(null); const cameraBusyRef=useRef(false); const fpsTimesRef=useRef<number[]>([]); const [result,setResult]=useState<any>(null); const [busy,setBusy]=useState(false); const [error,setError]=useState(""); const [history,setHistory]=useState<any[]>([]); const [totalMs,setTotalMs]=useState(0);
  const inferenceUrl=process.env.NEXT_PUBLIC_ANOMAVISION_INFERENCE_URL ?? "http://localhost:8001";
  const [studioConfig,setStudioConfig]=useState<any>(null); const activeThreshold=studioConfig?.thresholds?.[studioConfig?.algorithm]??null;
  useEffect(()=>{fetch(`${API_BASE}/api/config`,{cache:"no-store"}).then(r=>r.ok?r.json():null).then(setStudioConfig).catch(()=>setStudioConfig(null))},[]);
  useEffect(()=>()=>{if(cameraLoopRef.current!==null)window.clearInterval(cameraLoopRef.current);streamRef.current?.getTracks().forEach(t=>t.stop())},[]);
  async function startCamera(){setCameraError("");try{const stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});streamRef.current=stream;setCamera(true);if(videoRef.current)videoRef.current.srcObject=stream;}catch(e){setCameraError(e instanceof Error?e.message:"Camera access denied");}}
  function stopAutoInference(){if(cameraLoopRef.current!==null)window.clearInterval(cameraLoopRef.current);cameraLoopRef.current=null;setCameraAuto(false);}
  function stopCamera(){stopAutoInference();streamRef.current?.getTracks().forEach(t=>t.stop());streamRef.current=null;setCamera(false);setCameraFps(0);}
  async function predictFrame(){if(!videoRef.current||cameraBusyRef.current)return;cameraBusyRef.current=true;setCameraBusy(true);setCameraError("");try{const video=videoRef.current;if(video.readyState<2||!video.videoWidth||!video.videoHeight)throw new Error("Camera frame is not ready");const canvas=document.createElement("canvas");const [targetW,targetH]=studioConfig?.resize||[224,224];canvas.width=targetW;canvas.height=targetH;canvas.getContext("2d")?.drawImage(video,0,0);const blob=await new Promise<Blob|null>(resolve=>canvas.toBlob(resolve,"image/jpeg",0.9));if(!blob)throw new Error("Could not capture camera frame");const body=new FormData();body.append("files",blob,"camera.jpg");const started=performance.now();const r=await fetch(`${inferenceUrl}/predict-batch`,{method:"POST",body});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Camera inference failed");const rows=(d.batch_results||[]).map((x:any)=>({filename:"camera.jpg",...(x.result||{}),error:x.error}));setResult(d);setHistory(prev=>[...rows,...prev].slice(0,20));const elapsed=performance.now()-started;setTotalMs(rows.reduce((sum:number,x:any)=>sum+(Number(x.latency_ms)||0),0)||elapsed);setCameraFrames(prev=>prev+rows.length);const now=Date.now();fpsTimesRef.current=[...fpsTimesRef.current.filter(t=>now-t<5000),now];setCameraFps(fpsTimesRef.current.length/5);}catch(e){setCameraError(e instanceof Error?e.message:"Camera inference failed")}finally{cameraBusyRef.current=false;setCameraBusy(false)}}
  function startAutoInference(){if(!camera||cameraLoopRef.current!==null)return;setCameraAuto(true);void predictFrame();cameraLoopRef.current=window.setInterval(()=>void predictFrame(),cameraInterval);}
  async function predictBatch(){if(!files.length)return;setBusy(true);setError("");setResult(null);try{const body=new FormData();files.slice(0,10).forEach(f=>body.append("files",f,f.name));const r=await fetch(`${inferenceUrl}/predict-batch`,{method:"POST",body});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Batch inference failed");setResult(d);const rows=(d.batch_results||[]).map((x:any)=>({filename:x.filename,...(x.result||{}),error:x.error}));setHistory(prev=>[...rows,...prev].slice(0,20));setTotalMs(rows.reduce((sum:number,x:any)=>sum+(Number(x.latency_ms)||0),0))}catch(e){setError(e instanceof Error?e.message:"Batch inference failed")}finally{setBusy(false)}}
  return <><div className="page-head"><div><div className="eyebrow">Inference</div><h1>Live</h1><p className="subtitle">Run images or a local folder through the existing AnomaVision inference runtime.</p></div></div>
    <div className="live-top-stats"><Stat icon={<Wifi size={15}/>} label="Studio API" value={apiHealthy?"Online":"Offline"} meta="FastAPI connection"/><Stat icon={<CircleGauge size={15}/>} label="Inference API" value="External" meta="existing runtime"/><Stat icon={<BrainCircuit size={15}/>} label="Method" value={String(studioConfig?.algorithm||"—").toUpperCase()} meta={studioConfig?.resize?`${studioConfig.resize[0]}×${studioConfig.resize[1]} input`:"config.yml"}/><Stat icon={<ShieldCheck size={15}/>} label="Threshold" value={activeThreshold!=null?Number(activeThreshold).toFixed(3):"—"} meta="canonical config.yml"/><Stat icon={<CircleGauge size={15}/>} label="Batch latency" value={totalMs?`${totalMs.toFixed(0)} ms`:"—"} meta={totalMs?`${(1000/(totalMs/Math.max(1,files.length))).toFixed(1)} img/s`:"waiting"}/></div>
    <div className="card camera-panel"><div className="section-head"><div><div className="section-title">Camera stream</div><div className="subtitle">Continuous browser-camera inference using the existing runtime.</div></div><div className="badge">{cameraAuto?"Live inference":camera?"Camera on":"Off"}</div></div>{camera?<><div className="camera-frame"><video ref={videoRef} autoPlay playsInline muted className="camera-preview"/>{history[0]&&!history[0].error&&<div className={`camera-result ${(activeThreshold!=null?Number(history[0].anomaly_score)>=Number(activeThreshold):history[0].is_anomaly)?"anomaly":"normal"}`}><strong>{(activeThreshold!=null?Number(history[0].anomaly_score)>=Number(activeThreshold):history[0].is_anomaly)?"ANOMALY":"NORMAL"}</strong><span>Score {Number(history[0].anomaly_score||0).toFixed(3)}</span><small>Threshold {activeThreshold!=null?Number(activeThreshold).toFixed(3):"—"}</small></div>}</div><div className="grid-4"><Stat icon={<Activity size={15}/>} label="Live FPS" value={cameraFps.toFixed(1)} meta="processed frames / sec"/><Stat icon={<CircleGauge size={15}/>} label="Last latency" value={totalMs?totalMs.toFixed(0)+" ms":"—"} meta="latest frame"/><Stat icon={<ShieldCheck size={15}/>} label="Frames" value={String(cameraFrames)} meta="camera frames processed"/><Stat icon={<Wifi size={15}/>} label="Drift" value={history[0]?.drift_report?.drift_score!=null?Number(history[0].drift_report.drift_score).toFixed(3):"—"} meta={history[0]?.drift_report?.status||"not enabled"}/></div><div className="form-actions"><button className="primary" onClick={cameraAuto?stopAutoInference:startAutoInference} disabled={cameraBusy&&!cameraAuto}><Play size={13}/>{cameraAuto?"Stop live inference":"Start live inference"}</button><button className="secondary" onClick={predictFrame} disabled={cameraBusy||cameraAuto}>{cameraBusy?"Analyzing…":"Analyze frame"}</button><label className="compact-control">Interval<select value={cameraInterval} onChange={e=>setCameraInterval(Number(e.target.value))} disabled={cameraAuto}><option value={250}>250 ms</option><option value={500}>500 ms</option><option value={1000}>1000 ms</option></select></label><button className="secondary" onClick={stopCamera}>Stop camera</button></div></>:<button className="secondary" onClick={startCamera}><Wifi size={13}/> Start camera</button>}{cameraError&&<div className="form-error">{cameraError}</div>}</div>
    <div className="card"><div className="section-head"><div><div className="section-title">Run inference</div><div className="subtitle">Select images or a folder. Results come from the existing <code>/predict-batch</code> runtime.</div></div></div>
      <div className="form-grid"><label className="field"><span>Images</span><input type="file" accept="image/*" multiple onChange={e=>setFiles(Array.from(e.target.files||[]).slice(0,10))}/></label><label className="field"><span>Folder</span><input type="file" accept="image/*" multiple {...({webkitdirectory:""} as any)} onChange={e=>setFiles(Array.from(e.target.files||[]).slice(0,10))}/></label></div>
      {files.length>0&&<div className="file-list">{files.map((f,i)=><span key={i} className="status-chip">{f.name}</span>)}</div>}
      <button className="primary" onClick={predictBatch} disabled={!files.length||busy}><Play size={13}/>{busy?"Running…":"Run batch inference"}</button>{error&&<div className="form-error">{error}</div>}
      {result&&<div className="deployment-result"><div className="performance-grid"><div><span>Processed</span><b>{result.batch_results?.length??0}</b></div><div><span>Anomalies</span><b>{history.slice(0,result.batch_results?.length??0).filter(x=>!x.error&&(activeThreshold!=null?Number(x.anomaly_score||0)>=Number(activeThreshold):x.is_anomaly)).length}</b></div><div><span>Threshold</span><b>{activeThreshold!=null?Number(activeThreshold).toFixed(3):"—"}</b></div><div><span>Drift</span><b>{history.some(x=>x.drift_report)?"Observed":"Not enabled"}</b></div></div></div>}
    </div>
    {history.length>0&&<div className="live-charts">
      <div className="card"><div className="section-head"><div><div className="section-title">Anomaly score</div><div className="subtitle">Latest session · threshold {activeThreshold!=null?Number(activeThreshold).toFixed(3):"not available"}.</div></div></div><div className="spark-bars">{history.slice(0,12).reverse().map((x,i)=>{const v=x.error?0:Math.min(1,Math.max(0,Number(x.anomaly_score)||0)/Math.max(Number(activeThreshold)||1,...history.slice(0,12).map(y=>Number(y.anomaly_score)||0),1));return <div className="spark-column" key={i} title={x.filename}><div className="spark-bar" style={{height:`${Math.max(4,v*100)}%`}}/><span>{i+1}</span></div>})}</div></div>
      <div className="card"><div className="section-head"><div><div className="section-title">Latency</div><div className="subtitle">Per-image inference time.</div></div></div><div className="spark-bars">{history.slice(0,12).reverse().map((x,i)=>{const v=x.error?0:Number(x.latency_ms)||0;const max=Math.max(...history.slice(0,12).map(y=>Number(y.latency_ms)||0),1);return <div className="spark-column" key={i} title={x.filename}><div className="spark-bar" style={{height:`${Math.max(4,v/max*100)}%`}}/><span>{i+1}</span></div>})}</div></div>
    </div>}
    <div className="card"><div className="section-head"><div><div className="section-title">Recent inference events</div><div className="subtitle">Latest results from this Studio session.</div></div></div>
      {history.length===0?<div className="empty-state">No inference events yet.</div>:<div className="table-wrap"><table className="data-table"><thead><tr><th>Image</th><th>Score</th><th>Latency</th><th>Drift</th><th>Status</th><th>Result</th></tr></thead><tbody>{history.map((x,i)=><tr key={i}><td>{x.filename}</td><td>{x.error?"—":Number(x.anomaly_score).toFixed(4)}</td><td>{x.error?"—":`${Number(x.latency_ms||0).toFixed(1)} ms`}</td><td>{x.drift_report?`${Number(x.drift_report.drift_score||0).toFixed(3)} · ${x.drift_report.status||"observed"}`:"—"}</td><td>{x.error?"Error":activeThreshold!=null?(Number(x.anomaly_score||0)>=Number(activeThreshold)?"Anomaly":"Normal"):(x.is_anomaly?"Anomaly":"Normal")}</td><td>{x.error||"Completed"}</td></tr>)}</tbody></table></div>}
    </div></>;
}
function SettingsPage({apiHealthy}:{apiHealthy:boolean}) {
  const [config,setConfig]=useState<any>(null); const [error,setError]=useState(""); const [busy,setBusy]=useState(false);
  async function load(){
    setBusy(true);setError("");
    try{const r=await fetch(`${API_BASE}/api/config`,{cache:"no-store"});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Could not load configuration");setConfig(d);}
    catch(e){setError(e instanceof Error?e.message:"Could not load configuration")}finally{setBusy(false)}
  }
  useEffect(()=>{void load()},[]);
  const thresholds=config?.thresholds||{};
  return <>
    <div className="page-head"><div><div className="eyebrow">System</div><h1>Settings</h1><p className="subtitle">View the configuration Studio uses. Changes still belong in the canonical config.yml.</p></div><button className="secondary" onClick={load} disabled={busy}><Settings2 size={13}/>{busy?"Refreshing…":"Refresh"}</button></div>
    {!apiHealthy&&<div className="api-warning"><CircleGauge size={14}/> Studio API is offline. Settings are unavailable until FastAPI is running.</div>}
    {error&&<div className="form-error">{error}</div>}
    {config&&<div className="settings-grid">
      <section className="card"><div className="section-head"><div><div className="section-title">Inference</div><div className="subtitle">Read-only values from config.yml.</div></div><span className="badge">Canonical</span></div>
        <div className="settings-list"><div><span>Algorithm</span><b>{String(config.algorithm||"—").toUpperCase()}</b></div><div><span>Input size</span><b>{config.resize?.join(" × ")||"—"}</b></div><div><span>Normalize images</span><b>{config.normalize?"Enabled":"Disabled"}</b></div></div>
      </section>
      <section className="card"><div className="section-head"><div><div className="section-title">Thresholds</div><div className="subtitle">Used by Studio when displaying results.</div></div></div>
        <div className="settings-list">{Object.entries(thresholds).map(([name,value])=><div key={name}><span>{name.toUpperCase()}</span><b>{Number(value).toFixed(3)}</b></div>)}</div>
      </section>
      <section className="card"><div className="section-head"><div><div className="section-title">Drift monitoring</div><div className="subtitle">Existing monitoring configuration.</div></div></div>
        <div className="settings-list"><div><span>Status</span><b>{config.drift?.enabled?"Enabled":"Disabled"}</b></div><div><span>Window</span><b>{config.drift?.window??"—"}</b></div><div><span>Minimum samples</span><b>{config.drift?.min_samples??"—"}</b></div><div><span>Threshold</span><b>{config.drift?.threshold!=null?Number(config.drift.threshold).toFixed(3):"—"}</b></div><div><span>Evaluation interval</span><b>{config.drift?.evaluation_interval??"—"}</b></div></div>
      </section>
      <section className="card settings-note"><ShieldCheck size={17}/><div><strong>One source of truth</strong><span>Studio reads these values from the existing configuration API. It does not create a second configuration system or modify the anomaly-detection engine.</span></div></section>
    </div>}
  </>;
}

function Placeholder({page}:{page:Page}) { const descriptions:Record<Page,string>={Overview:"",Projects:"",Datasets:"",Training:"",Models:"",Deployments:"Export and validate models for production targets without changing the algorithm core.",Live:"Inspect camera or stream inference using the existing AnomaVision runtime.",Monitoring:"Track runtime health, latency and production data drift.",Settings:"View the canonical Studio configuration."};return <><div className="page-head"><div><div className="eyebrow">Workspace</div><h1>{page}</h1><p className="subtitle">{descriptions[page]}</p></div><button className="secondary"><SlidersHorizontal size={13}/> Configure</button></div><div className="card placeholder"><div className="icon-box"><Sparkles size={18}/></div><div><b>Connected to the Studio architecture</b><p className="subtitle">This view is ready to consume the same Python services through the Studio API. No ML logic is duplicated in the frontend.</p></div></div></>; }

function Stat({icon,label,value,meta,green}:{icon:React.ReactNode;label:string;value:string;meta:string;green?:boolean}){return <div className="card"><div className="stat-label">{icon}<span>{label}</span></div><div className="stat-value">{value}</div><div className="stat-meta">{green&&<span className="status-dot"/>}{meta}</div></div>}
function ActivityRow({icon,title,sub,badge}:{icon:React.ReactNode;title:string;sub:string;badge:string}){return <div className="row"><div className="row-main"><div className="icon-box">{icon}</div><div><div className="row-title">{title}</div><div className="row-sub">{sub}</div></div></div><div className="badge">{badge}</div></div>}
function HealthRow({title,sub,ok}:{title:string;sub:string;ok:boolean}){return <div className="row"><div className="row-main"><div className="icon-box"><CircleGauge size={14}/></div><div><div className="row-title">{title}</div><div className="row-sub">{sub}</div></div></div><div className="badge"><span className={`status-dot ${ok?"":"offline-dot"}`}/>{ok?"Healthy":"Offline"}</div></div>}
