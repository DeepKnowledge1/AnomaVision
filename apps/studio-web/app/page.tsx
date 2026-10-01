"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, Box, BrainCircuit, CircleGauge, Database, FlaskConical, LoaderCircle,
  LayoutDashboard, MonitorCog, Play, Rocket, Settings2, ShieldCheck,
  SlidersHorizontal, Sparkles, Wifi, FolderOpen, Menu, X, ChevronDown,
} from "lucide-react";

type Page = "Overview" | "Projects" | "Datasets" | "Training" | "Models" | "Deployments" | "Live" | "Inference" | "Results" | "Performance" | "Monitoring" | "Settings";
type Project = { id: string; name: string; description?: string; algorithm: string; status: string; };
type Model = { id: string; algorithm: string; class_name: string; run_name: string; status: string; path?: string; };
type DatasetReport = {
  image_count: number; valid_count: number; failed_count: number; duplicate_count: number;
  resolutions: Record<string, number>; failures?: { path: string; error: string }[];
};
type Catalog = {
  algorithms: Record<string, { name?: string; description?: string }>;
  deployment_targets: Record<string, unknown>;
};

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
    { label: "Inference", icon: Wifi },
    { label: "Results", icon: ShieldCheck },
    { label: "Performance", icon: CircleGauge },
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
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [inferenceSession, setInferenceSession] = useState<{result:any;history:any[];config:any}|null>(null);

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
            <button className="help-button" aria-label="Open settings" title="Settings" onClick={() => navigate("Settings")}><Settings2 size={16}/></button>
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
          {page === "Training" && <TrainingPage key={selectedProject || "no-project"} project={project} onFinished={() => loadModels(selectedProject)} onNavigate={navigate}/>}
          {page === "Models" && <ModelsPage models={models} onRefresh={() => loadModels(selectedProject)} onNavigate={navigate} onDeploy={(id) => { setDeploymentModelId(id); navigate("Deployments"); }}/>}
          {page === "Deployments" && <DeploymentsPage project={project} models={models} initialModelId={deploymentModelId}/>}
          {page === "Performance" && <PerformancePage project={project} session={inferenceSession}/>}
          <MonitoringPage project={project} visible={page === "Monitoring"} />
          
          {page === "Results" && <ResultsPage session={inferenceSession} onNavigate={navigate}/>}
          
          {page === "Settings" && <SettingsPage apiHealthy={apiHealthy}/>}
          <div className={page === "Inference" || page === "Live" ? "" : "inference-session-background"}>
            <LivePage apiHealthy={apiHealthy} projectId={selectedProject} projectAlgorithm={project?.algorithm} visible={page === "Inference" || page === "Live"} onResult={(result, history, config) => setInferenceSession({result, history, config})} onResults={() => navigate("Results")}/>
          </div>
        </div>
      </main>
    </div>
  );
}

function Overview({ project, projects, models, apiHealthy, onNavigate }: { project?: Project; projects: Project[]; models: Model[]; apiHealthy: boolean; onNavigate: (p: Page) => void }) {
 const latest=models[0]; const workflow=[["01","Dataset","Connect and validate inspection images.","Datasets",false],["02","Train","Build a PaDiM or PatchCore detector.","Training",Boolean(latest)],["03","Model","Review the trained artifact and metrics.","Models",Boolean(latest)],["04","Deploy","Export and validate your runtime target.","Deployments",false]] as const;
 return <><section className="overview-hero"><div className="overview-hero-copy"><div className="eyebrow">PROJECT WORKSPACE</div><h1>{project?.name??"Build your first anomaly detector"}</h1><p>{project?.description||"A focused workspace for preparing inspection data, training an anomaly model, and taking it into production."}</p><div className="hero-actions"><button className="primary hero-primary" onClick={()=>onNavigate(project?"Datasets":"Projects")}>{project?"Continue workflow":"Create your first project"} <span>→</span></button>{project&&<button className="secondary" onClick={()=>onNavigate("Inference")}>Run inference</button>}</div>{!apiHealthy&&<div className="hero-warning"><span className="status-dot offline-dot"/> Studio API is offline. Start the API service to continue.</div>}</div><div className="hero-visual" aria-hidden="true"><div className="hero-orbit hero-orbit-a"/><div className="hero-orbit hero-orbit-b"/><div className="hero-center"><span>AV</span><small>VISION</small></div><div className="hero-node hero-node-a">DATA</div><div className="hero-node hero-node-b">MODEL</div><div className="hero-node hero-node-c">LIVE</div></div></section>
 <section className="overview-metrics"><div className="metric-card metric-primary"><span>ACTIVE PROJECT</span><strong>{project?.name||"—"}</strong><small>{project?project.algorithm.toUpperCase()+" detector":"Create a project to begin"}</small></div><div className="metric-card"><span>TRAINED MODELS</span><strong>{models.length}</strong><small>{latest?"Latest model available":"No model trained yet"}</small></div><div className="metric-card"><span>WORKSPACES</span><strong>{projects.length}</strong><small>{projects.length===1?"1 project configured":"Projects in this Studio"}</small></div><div className="metric-card"><span>SERVICE</span><strong className={apiHealthy?"metric-ok":"metric-bad"}>{apiHealthy?"Ready":"Offline"}</strong><small>{apiHealthy?"Studio API connected":"Connection required"}</small></div></section>
 <section className="overview-section"><div className="section-heading-row"><div><div className="eyebrow">BUILD PIPELINE</div><h2>From image to production</h2><p>Complete each step in order. Your project state stays available across the workspace.</p></div></div><div className="pipeline">{workflow.map(([key,label,desc,target,done],i)=><div className="pipeline-item-wrap" key={key}><button className={`pipeline-item ${done?"completed":""}`} onClick={()=>onNavigate(target as Page)}><div className="pipeline-number">{done?"✓":key}</div><div className="pipeline-copy"><strong>{label}</strong><span>{desc}</span></div><span className="pipeline-arrow">→</span></button>{i<3&&<div className="pipeline-connector"/>}</div>)}</div></section>
 <section className="overview-lower"><div className="card recent-panel"><div className="panel-head"><div><div className="eyebrow">RECENT MODELS</div><h3>Model library</h3></div><button className="text-button" onClick={()=>onNavigate("Models")}>View models →</button></div>{models.length?<div className="model-list">{models.slice(0,3).map(m=><button className="model-row" key={m.id} onClick={()=>onNavigate("Models")}><span className="model-avatar"><BrainCircuit size={16}/></span><span className="model-row-copy"><strong>{m.algorithm.toUpperCase()} · {m.class_name}</strong><small>{m.run_name}</small></span><span className="model-status">{m.status}</span><span>→</span></button>)}</div>:<div className="panel-empty"><div className="empty-icon"><BrainCircuit size={18}/></div><strong>No trained models</strong><span>Validate your dataset, then train the first detector.</span><button className="secondary" onClick={()=>onNavigate("Datasets")}>Check dataset</button></div>}</div><div className="card quick-panel"><div className="panel-head"><div><div className="eyebrow">QUICK ACTIONS</div><h3>What do you want to do?</h3></div></div><button className="quick-action" onClick={()=>onNavigate("Datasets")}><span className="quick-icon"><Database size={16}/></span><span><strong>Check a dataset</strong><small>Validate images and structure</small></span><b>→</b></button><button className="quick-action" onClick={()=>onNavigate("Inference")}><span className="quick-icon"><Wifi size={16}/></span><span><strong>Inspect an image</strong><small>Run anomaly detection</small></span><b>→</b></button><button className="quick-action" onClick={()=>onNavigate("Monitoring")}><span className="quick-icon"><Activity size={16}/></span><span><strong>Check monitoring</strong><small>Review drift and production health</small></span><b>→</b></button></div></section></>;
}
function ProjectsPage({ projects, selectedProject, onSelect, onCreated }: { projects: Project[]; selectedProject: string; onSelect: (id:string)=>void; onCreated:()=>Promise<void> }) {
  const [name,setName]=useState(""); const [description,setDescription]=useState(""); const [algorithm,setAlgorithm]=useState("patchcore"); const [error,setError]=useState(""); const [busy,setBusy]=useState(false);
  const [jobId,setJobId]=useState("");
  const [jobStatus,setJobStatus]=useState("");
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

function TrainingPage({ project, onFinished, onNavigate }: { project?: Project; onFinished:()=>Promise<void>; onNavigate:(page:Page)=>void }) {
  const [dataset,setDataset]=useState("");
  const [algorithm,setAlgorithm]=useState(project?.algorithm||"patchcore");
  const [className,setClassName]=useState("");
  const [batch,setBatch]=useState("");
  const [backbone,setBackbone]=useState("");
  const [resize,setResize]=useState("");
  const [configLoaded,setConfigLoaded]=useState(false);
  const [result,setResult]=useState<Record<string,any>|null>(null);
  const [error,setError]=useState("");
  const [busy,setBusy]=useState(false);
  const [validating,setValidating]=useState(false);
  const [pickerBusy,setPickerBusy]=useState(false);
  const [jobId,setJobId]=useState("");
  const [jobStatus,setJobStatus]=useState("");
  const [resolved,setResolved]=useState<{dataset_path:string;class_name:string;train_good:string}|null>(null);
  const [availableAlgorithms,setAvailableAlgorithms]=useState<Catalog["algorithms"]>({});
  const [catalogLoading,setCatalogLoading]=useState(true);
  const [catalogError,setCatalogError]=useState("");
  const onFinishedRef=useRef(onFinished);

  const algorithmOptions=useMemo(()=>Object.entries(availableAlgorithms),[availableAlgorithms]);

  useEffect(()=>{onFinishedRef.current=onFinished},[onFinished]);

  useEffect(()=>{
    let cancelled=false;
    fetch(`${API_BASE}/api/catalog`,{cache:"no-store"})
      .then(async r=>{const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.detail||"Could not load supported training methods");return d as Catalog})
      .then(d=>{if(cancelled)return;const methods=d.algorithms||{};setAvailableAlgorithms(methods);if(!Object.keys(methods).length)setCatalogError("The Studio API did not report any supported training methods.")})
      .catch(e=>{if(!cancelled)setCatalogError(e instanceof Error?e.message:"Could not load supported training methods")})
      .finally(()=>{if(!cancelled)setCatalogLoading(false)});
    return()=>{cancelled=true};
  },[]);

  useEffect(()=>{
    if(!project)return;
    const first=Object.keys(availableAlgorithms)[0];
    setAlgorithm(availableAlgorithms[project.algorithm]?project.algorithm:first||project.algorithm);
  },[project?.id,project?.algorithm,availableAlgorithms]);

  useEffect(()=>{
    if(!project || typeof window==="undefined") return;
    const key="anomavision:training-job:"+project.id;
    const saved=window.localStorage.getItem(key);
    if(!saved) return;
    try{
      const job=JSON.parse(saved);
      if(!job.job_id){window.localStorage.removeItem(key);return}
      setJobId(String(job.job_id));setBusy(true);setJobStatus("Resuming training…");
    }catch{window.localStorage.removeItem(key)}
  },[project?.id]);

  useEffect(()=>{
    if(!project || !jobId || typeof window==="undefined")return;
    let cancelled=false;
    let timer:number|undefined;
    const key="anomavision:training-job:"+project.id;
    const poll=async()=>{
      try{
        const r=await fetch(`${API_BASE}/api/projects/${project.id}/training/${jobId}`,{cache:"no-store"});
        const d=await r.json().catch(()=>({}));
        if(!r.ok)throw new Error(d.detail||"Could not read training status");
        if(cancelled)return;
        setJobStatus(d.message|| (d.status==="queued"?"Waiting to start…":"Training in progress"));
        if(d.status==="completed"){
          window.localStorage.removeItem(key);setResult(d.result||{});setBusy(false);setJobId("");setJobStatus("Training completed");
          await onFinishedRef.current();return;
        }
        if(d.status==="failed"){
          window.localStorage.removeItem(key);setError(d.message||"Training failed");setBusy(false);setJobId("");return;
        }
        timer=window.setTimeout(poll,1500);
      }catch(e){if(!cancelled){setError(e instanceof Error?e.message:"Could not read training status");setBusy(false)}}
    };
    void poll();
    return()=>{cancelled=true;if(timer!==undefined)window.clearTimeout(timer)};
  },[project?.id,jobId]);


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

  async function chooseTrainingFolder(){
    if(!project){setError("Select a project first.");return;}
    setPickerBusy(true);setError("");
    try{
      const r=await fetch(API_BASE+"/api/dataset/pick-folder",{cache:"no-store"});
      const d=await r.json().catch(()=>({}));
      if(!r.ok)throw new Error(d.detail||"Could not open the folder dialog");
      if(d.path){setDataset(String(d.path));setResolved(null);}
    }catch(e){setError(e instanceof Error?e.message:"Could not open the folder dialog")}
    finally{setPickerBusy(false)}
  }

  async function validateDataset(){
    if(!project){setError("Select a project first.");return null;}
    if(!dataset.trim()){setError("Choose a dataset folder first.");return null;}
    setValidating(true);setError("");
    try{
      const r=await fetch(`${API_BASE}/api/projects/${project.id}/datasets/resolve`,{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({path:dataset,recursive:true,class_name:className.trim()||undefined})
      });
      const d=await r.json();
      if(!r.ok)throw new Error(d.detail||"Invalid training dataset");
      setResolved(d);
      setDataset(d.dataset_path);
      setClassName(d.class_name);
      return d;
    }catch(e){setResolved(null);setError(e instanceof Error?e.message:"Invalid training dataset");return null}
    finally{setValidating(false)}
  }

  async function train(){
    if(!project){setError("Select a project first.");return;}
    if(!dataset.trim()){setError("Add your dataset path first.");return;}
    if(!availableAlgorithms[algorithm]){setError("Choose a training method supported by the connected Studio API.");return;}
    setBusy(true);setError("");setResult(null);setJobStatus("Validating dataset…");
    try{
      const canonical=await validateDataset();
      if(!canonical){setBusy(false);return;}
      const body:{dataset_path:string;algorithm:string;class_name?:string;batch_size?:number;resize?:number[];backbone?:string}={dataset_path:canonical.dataset_path,algorithm};
      const selectedClass=String(canonical.class_name||className).trim();
      if(selectedClass) body.class_name=selectedClass;
      if(batch)body.batch_size=Number(batch);
      if(backbone)body.backbone=backbone;
      const dims=resize.split(",").map(Number);
      if(dims.length===2&&dims.every(Number.isFinite))body.resize=dims;
      const r=await fetch(`${API_BASE}/api/projects/${project.id}/training`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
      const d=await r.json();
      if(!r.ok)throw new Error(d.detail||"Could not start training");
      if(!d.job_id){setResult(d);setJobStatus("Training completed");setBusy(false);await onFinishedRef.current();return;}
      setJobId(d.job_id);setJobStatus("Training queued…");
      if(typeof window!=="undefined") window.localStorage.setItem("anomavision:training-job:"+project.id,JSON.stringify({job_id:d.job_id}));
    }catch(e){setError(e instanceof Error?e.message:"Training failed");setJobStatus("Training stopped");setBusy(false)}
  }

  return (
    <div className="training-page">
      <div className="page-head">
        <div>
          <div className="eyebrow">Build a model</div>
          <h1>Training</h1>
          <p className="subtitle">Set up your experiment, then let the existing AnomaVision engine do the training.</p>
        </div>
      </div>

      {!project && (
        <div className="card empty-state">
          <strong>Select a project first</strong>
          <span>Your training run will be saved inside the selected project.</span>
        </div>
      )}

      {project && (
        <div className="training-content">
          <div className="training-steps">
            <div className={`training-step ${resolved?"is-complete":"active"}`} aria-current={!resolved?"step":undefined}><span>{resolved?"✓":"1"}</span><div><strong>Data</strong><small>{resolved?"Validated":"Choose images"}</small></div></div>
            <div className="training-line"/>
            <div className={`training-step ${busy||result?"is-complete":"active"}`}><span>{busy||result?"✓":"2"}</span><div><strong>Method</strong><small>Choose detector</small></div></div>
            <div className="training-line"/>
            <div className={`training-step ${result?"is-complete":busy?"active":""}`} aria-current={busy?"step":undefined}><span>{result?"✓":"3"}</span><div><strong>Run</strong><small>{busy?"Training":result?"Completed":"Start training"}</small></div></div>
          </div>

          <div className="two-column training-layout">
            <section className="card">
              <div className="section-title">1. Training data</div>
              <p className="form-help">Point Studio to the same local image folder you checked in Data Readiness.</p>
              <label>Dataset path<input value={dataset} disabled={busy||validating} onChange={e=>{setDataset(e.target.value);setResolved(null)}} placeholder={configLoaded?"From config.yml":"Loading config…"}/></label>
              <div className="form-actions">
                <button className="secondary" type="button" onClick={chooseTrainingFolder} disabled={busy||validating||pickerBusy}><FolderOpen size={13}/>{pickerBusy?"Opening…":"Choose folder"}</button>
                <button className="secondary" type="button" onClick={validateDataset} disabled={busy||validating||!dataset.trim()}>{validating?<><LoaderCircle className="training-spinner" size={13}/>Checking…</>:"Check training layout"}</button>
              </div>
              <label>Class name<input value={className} disabled={busy||validating} onChange={e=>{setClassName(e.target.value);setResolved(null)}} placeholder={configLoaded?"From config.yml":"Loading config…"}/></label>
              {resolved&&<div className="config-field-note"><span>✓ Training folder: <code>{resolved.train_good}</code></span></div>}
              <div className="config-field-note">{className ? <span>Using <strong>{className}</strong> from {configLoaded ? "config.yml" : "the current setup"}.</span> : <span>Class name will be taken from <strong>config.yml</strong> if you leave it empty.</span>}</div>
            </section>

            <section className="card">
              <div className="section-title">2. Detection method</div>
              <p className="form-help">Choose the anomaly detector for this experiment.</p>
              <div className="algorithm-options">
                {algorithmOptions.length?algorithmOptions.map(([value,details])=>
                  <button key={value} type="button" className={`algorithm-option ${algorithm===value?"selected":""}`} onClick={()=>setAlgorithm(value)} disabled={busy} aria-pressed={algorithm===value}>
                    <span className="algorithm-radio">{algorithm===value?"✓":""}</span>
                    <span><strong>{details.name||value}</strong><small>{details.description||"Supported by the connected AnomaVision engine."}</small></span>
                  </button>
                ):<div className="catalog-empty" role={catalogError?"alert":"status"}>{catalogLoading?"Loading supported training methods…":catalogError||"No supported training methods are available."}</div>}
              </div>
            </section>
          </div>

          <section className="card training-advanced">
            <div className="section-head">
              <div><div className="section-title">Optional settings</div><div className="subtitle">Leave these empty to use the canonical config.yml values.</div></div>
              <div className="section-link">Config-aware</div>
            </div>
            <div className="form-grid">
              <label>Image size<input value={resize} disabled={busy} onChange={e=>setResize(e.target.value)} placeholder="From config.yml"/></label>
              <label>Batch size<input value={batch} disabled={busy} inputMode="numeric" onChange={e=>setBatch(e.target.value)} placeholder="Use config default"/></label>
              <label>Backbone<input value={backbone} disabled={busy} onChange={e=>setBackbone(e.target.value)} placeholder="Use config default"/></label>
            </div>
          </section>

          <div className={`training-action ${busy?"is-running":result?"is-complete":""}`} aria-live="polite" aria-busy={busy}>
            <div className="training-action-copy">
              <strong>{busy?"Training run in progress":result?"Training complete":"Ready to train"}</strong>
              <span>{busy?(jobStatus||"Training in progress"):result?"Your model is ready to review in Models.":`${(availableAlgorithms[algorithm]?.name||algorithm).toUpperCase()} · ${resize||"config.yml image size"} · ${className||"config.yml class"}`}</span>
              {busy&&<div className="training-progress" role="progressbar" aria-label="Training progress" aria-valuetext={jobStatus||"Training in progress"}><span/></div>}
            </div>
            <button className="primary" onClick={train} disabled={busy||catalogLoading||!algorithmOptions.length}>
              {busy?<><LoaderCircle className="training-spinner" size={14}/>{jobStatus||"Training…"}</>:<><Play size={13}/>Start training</>}
            </button>
          </div>

          {error && <div className="form-error" role="alert">{error}</div>}

          {result && (
            <section className="card training-result">
              <div className="section-head">
                <div><div className="section-title">Training completed</div><div className="subtitle">The model is now available in Models.</div></div>
                <div className="badge success-badge">Completed</div>
              </div>
              <div className="training-success">
                <div className="training-success-icon"><ShieldCheck size={18}/></div>
                <div><strong>Your model was created successfully.</strong><span>Open Models to review the artifact or continue to validation.</span></div>
                <button className="secondary" onClick={()=>onNavigate("Models")}>Review model</button>
              </div>
              <div className="training-result-grid">
                <div><span>Algorithm</span><b>{String(result.algorithm ?? algorithm).toUpperCase()}</b></div>
                <div><span>Class</span><b>{String(result.class_name ?? className ?? "From config")}</b></div>
                <div><span>Run</span><b>{String(result.run_name ?? result.model_id ?? result.run_dir ?? "Created")}</b></div>
                <div><span>Status</span><b>{String(result.status ?? "trained")}</b></div>
              </div>
              {(result.model_path || result.path || result.model) && (
                <div className="artifact-box">
                  <span>Model artifact</span>
                  <code>{String(result.model_path ?? result.path ?? result.model)}</code>
                </div>
              )}
              <details className="technical-details">
                <summary>Technical details</summary>
                <pre className="result-box">{JSON.stringify(result,null,2)}</pre>
              </details>
            </section>
          )}
        </div>
      )}
    </div>
  );
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
      <div className="deployment-readiness"><ShieldCheck size={16}/><div><strong>Validate before you deploy</strong><span>Studio runs the existing checks against the selected artifact. A failed check means the result needs review before deployment.</span></div></div>
      <div className="deployment-steps">
        <div className="deployment-step active"><span>1</span><div><b>Select model</b><small>Choose a trained artifact</small></div></div>
        <div className="deployment-step"><span>2</span><div><b>Choose target</b><small>Pick the runtime format</small></div></div>
        <div className="deployment-step"><span>3</span><div><b>Validate</b><small>Check integrity and performance</small></div></div>
      </div>

      <div className="two-column">
        <section className="card deployment-config">
          <div className="section-title">Prepare validation</div>
          <p className="form-help">Choose an existing trained model and target. Studio orchestrates export and validation without changing the model or algorithm.</p>

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
            <div><div className="section-title">Validation result</div><div className="subtitle">{result?"Latest validation run":"The validation outcome will appear here."}</div></div>
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

            <details className="technical-details deployment-technical"><summary>Technical details</summary><div className="technical-grid"><div><span>Target</span><b>{String(target).toUpperCase()}</b></div><div><span>Format</span><b>{String(result.validation.format||target).toUpperCase()}</b></div><div><span>Validation runs</span><b>{result.validation.runs ?? "—"}</b></div><div><span>Warm-up runs</span><b>{result.validation.warmup_runs ?? "—"}</b></div></div><div className="artifact-box"><span>Artifact path</span><code>{result.artifact}</code></div></details>
          </>}
        </section>
      </div>
    </>}
  </>;
}

function PerformancePage({project,session}:{project?:Project;session:{result:any;history:any[];config:any}|null}) {
  const config=session?.config;
  const benchmark=config?.inference_benchmark;
  const latest=session?.history?.[0];
  const latency=latest?.latency_ms!=null?Number(latest.latency_ms):null;
  const fps=latency&&latency>0?1000/latency:null;
  const device=latest?.device||config?.device||"auto";
  const algorithm=config?.algorithm||latest?.algorithm||"—";
  return <>
    <div className="page-head">
      <div>
        <div className="eyebrow">Runtime performance</div>
        <h1>Performance</h1>
        <p className="subtitle">Review inference speed and the benchmark contract without changing how AnomaVision measures performance.</p>
      </div>
    </div>
    {!project ? <div className="card empty-state"><div className="dataset-empty-icon"><CircleGauge size={20}/></div><strong>Select a project first</strong><span>Performance information is shown for the active Studio workspace.</span></div> : <>
      <div className="performance-status">
        <div><span className="performance-kicker">CURRENT INFERENCE</span><strong>{latest ? "Performance data available" : "Ready to measure"}</strong><p>{latest ? "The values below come from your latest inference session." : "Run an image inference to see live latency and throughput here."}</p></div>
        <div className="performance-method"><b>{String(algorithm).toUpperCase()}</b><span>{config?.resize ? `${config.resize[0]}×${config.resize[1]} input` : "config.yml"}</span></div>
      </div>
      <div className="grid-4 section">
        <Stat icon={<CircleGauge size={15}/>} label="Latency" value={latency!=null?`${latency.toFixed(1)} ms`:"—"} meta="reported by inference runtime"/>
        <Stat icon={<Activity size={15}/>} label="Throughput" value={fps!=null?`${fps.toFixed(1)} img/s`:"—"} meta="derived from latest latency"/>
        <Stat icon={<BrainCircuit size={15}/>} label="Algorithm" value={String(algorithm).toUpperCase()} meta="active configuration"/>
        <Stat icon={<MonitorCog size={15}/>} label="Device" value={String(device)} meta="runtime selection"/>
      </div>
      <div className="two-column">
        <section className="card">
          <div className="section-head"><div><div className="section-title">What the numbers mean</div><div className="subtitle">A quick interpretation for day-to-day engineering work.</div></div></div>
          <div className="performance-explanations">
            <div><b>Latency</b><span>Time reported for one inference result. Lower values mean faster response.</span></div>
            <div><b>Throughput</b><span>Approximate images per second calculated from the latest single-image latency.</span></div>
            <div><b>Device</b><span>The runtime device selected by the existing configuration. Studio does not override it.</span></div>
          </div>
        </section>
        <section className="card">
          <div className="section-head"><div><div className="section-title">Benchmark configuration</div><div className="subtitle">The existing reproducible benchmark settings from config.yml.</div></div></div>
          {benchmark ? <div className="check-list">
            <div className="check-row"><span>Status</span><b>{benchmark.enabled ? "Enabled" : "Disabled"}</b></div>
            <div className="check-row"><span>Warm-up runs</span><b>{benchmark.warmup_runs ?? "—"}</b></div>
            <div className="check-row"><span>Timed runs</span><b>{benchmark.test_runs ?? "—"}</b></div>
            <div className="check-row"><span>Batch size</span><b>{benchmark.batch_size ?? "—"}</b></div>
          </div> : <div className="monitor-empty"><CircleGauge size={18}/><strong>Benchmark settings unavailable</strong><span>The active config did not expose benchmark settings.</span></div>}
        </section>
      </div>
      <details className="technical-details card"><summary>Technical details</summary>
        <div className="technical-grid">
          <div><span>Input size</span><b>{config?.resize ? `${config.resize[0]}×${config.resize[1]}` : "—"}</b></div>
          <div><span>Batch size</span><b>{config?.batch_size ?? benchmark?.batch_size ?? "—"}</b></div>
          <div><span>Workers</span><b>{config?.num_workers ?? "—"}</b></div>
          <div><span>Pin memory</span><b>{config?.pin_memory == null ? "—" : String(config.pin_memory)}</b></div>
        </div>
      </details>
      <div className="performance-note"><CircleGauge size={15}/><div><strong>Benchmarking stays reproducible</strong><span>Studio only presents existing configuration and observed inference results. It does not modify benchmark methodology or core runtime calculations.</span></div></div>
    </>}
  </>;
}

function MonitoringPage({project,visible}:{project?:Project;visible:boolean}) {
  const [summary,setSummary]=useState<Record<string,any>|null>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");

  async function load(){
    if(!project){setSummary(null);return}
    setBusy(true);setError("");
    try{
      const inferenceUrl=process.env.NEXT_PUBLIC_ANOMAVISION_INFERENCE_URL ?? (typeof window !== "undefined" ? window.location.protocol + "//" + window.location.hostname + ":8001" : "http://localhost:8001");
      const [summaryResponse, liveResponse] = await Promise.all([
        fetch(`${API_BASE}/api/projects/${project.id}/monitoring`,{cache:"no-store"}),
        fetch(`${inferenceUrl}/monitoring/status?project_id=${encodeURIComponent(project.id)}`,{cache:"no-store"}).catch(()=>null),
      ]);
      const d=await summaryResponse.json().catch(()=>({}));
      if(!summaryResponse.ok)throw new Error(d.detail||"Could not load monitoring");
      const live=liveResponse?.ok ? await liveResponse.json().catch(()=>null) : null;
      if(live){
        setSummary({...d,status:live.status||d.status||"no_data",latest:{...(d.latest||{}),...live,current_samples:live.window_fill,reference_samples:d.latest?.reference_samples,feature_dimensions:d.latest?.feature_dimensions},live:true,samples_seen:live.samples_seen,window_fill:live.window_fill,window_size:live.window_size,min_samples:live.min_samples});
      } else setSummary(d);
    }catch(e){setError(e instanceof Error?e.message:"Could not load monitoring")}
    finally{setBusy(false)}
  }

  useEffect(()=>{
    void load();
    if(!project) return;
    const timer=window.setInterval(()=>void load(),3000);
    return ()=>window.clearInterval(timer);
  },[project?.id]);

  const latest=summary?.latest;
  const status=String(summary?.status||"no_data");
  const warnings=(latest?.warnings||[]) as string[];
  const driftScore=latest?.drift_score!=null?Number(latest.drift_score):null;
  const psi=latest?.psi!=null?Number(latest.psi):null;
  const driftDetected=status==="drift" || (latest?.threshold!=null && driftScore!=null && driftScore>=Number(latest.threshold));
  const healthy=!driftDetected && (status==="ok" || status==="healthy" || status==="stable" || status==="no_data");
  const statusLabel=status==="no_data"?"No data":status.replaceAll("_"," ");

  return <div className={visible ? "" : "studio-page-hidden"}>\n  return <>
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
      <div className={`monitor-status ${driftDetected?"drift":healthy?"healthy":"attention"}`}>
        <div className="monitor-status-icon"><span className="status-dot"/></div>
        <div>
          <strong>{latest ? (driftDetected?"Drift detected":healthy?"Monitoring is healthy":"Review drift signals") : "Monitoring is ready"}</strong>
          <span>{latest ? `Latest report · ${warnings.length} warning${warnings.length===1?"":"s"} · ${summary?.report_count||0} stored reports` : "No monitoring state yet. Start inference to begin collecting production samples."}</span>
        </div>
        <div className="monitor-status-value">{statusLabel}</div>
      </div>

      {error&&<div className="form-error">{error}</div>}

      {driftDetected && (
        <div className="drift-alarm" role="alert">
          <div className="drift-alarm-icon"><Activity size={19}/></div>
          <div className="drift-alarm-copy">
            <strong>Significant data change detected</strong>
            <span>The current images are noticeably different from the reference data used by the model. Check the input data before relying on production results.</span>
          </div>
          <div className="drift-alarm-metric">
            <span>Drift level</span>
            <b>{driftScore!=null ? (driftDetected ? "Major" : driftScore >= 0.5 ? "Significant" : driftScore >= 0.2 ? "Slight" : "Normal") : "—"}</b>
            <small>Technical score {driftScore!=null?driftScore.toFixed(3):"—"} · threshold {latest?.threshold!=null?Number(latest.threshold).toFixed(3):"—"}</small>
          </div>
        </div>
      )}

      <div className="grid-4 section">
        <Stat
          icon={<Activity size={15}/>}
          label="Data drift"
          value={driftScore==null ? "No data" : driftDetected ? "Major change" : driftScore >= 0.5 ? "Significant change" : driftScore >= 0.2 ? "Slight change" : "Normal"}
          meta={driftScore==null ? "Waiting for monitoring data" : `Score ${driftScore.toFixed(3)} · ${driftDetected ? "attention required" : "within expected range"}`}
          green={driftScore!=null && !driftDetected}
          alert={driftDetected}
        />
        <Stat icon={<Database size={15}/>} label="Distribution change" value={psi!=null?psi.toFixed(3):"—"} meta="PSI · technical detail"/>
        <Stat icon={<ShieldCheck size={15}/>} label="Warnings" value={String(warnings.length)} meta={warnings.length?"Review latest signals":"No warnings reported"} green={!warnings.length}/>
        <Stat icon={<CircleGauge size={15}/>} label="Reports" value={String(summary?.report_count||0)} meta="Stored monitoring reports"/>
      </div>
      {latest && (
        <div className={`drift-metric-grid ${driftDetected?"drift-metric-alert":""}`}>
          <div><span>Mean shift</span><b>{Number(latest.mean_shift).toFixed(4)}</b></div>
          <div><span>Std shift</span><b>{Number(latest.std_shift).toFixed(4)}</b></div>
          <div><span>Cosine shift</span><b>{Number(latest.cosine_shift).toFixed(4)}</b></div>
          <div><span>Current samples</span><b>{latest.current_samples ?? "—"}</b></div>
        </div>
      )}

      <div className="two-column">
        <section className="card">
          <div className="section-head">
            <div><div className="section-title">Latest drift report</div><div className="subtitle">The most recent report produced by the existing monitoring engine.</div></div>
          </div>
          {latest ? <div className="check-list">
            <div className="check-row"><span>Reference samples</span><b>{latest.reference_samples}</b></div>
            <div className="check-row"><span>Current samples</span><b>{latest.current_samples ?? latest.samples_seen ?? "—"}</b></div>
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

      <details className="technical-details card">
        <summary>Technical details</summary>
        <div className="technical-grid">
          <div><span>Reference samples</span><b>{latest?.reference_samples ?? "—"}</b></div>
          <div><span>Feature dimensions</span><b>{latest?.feature_dimensions ?? "—"}</b></div>
          <div><span>Window</span><b>{summary?.window ?? latest?.window ?? "—"}</b></div>
          <div><span>Minimum samples</span><b>{summary?.min_samples ?? latest?.min_samples ?? "—"}</b></div>
        </div>
      </details>
      <div className="card monitoring-note">
        <ShieldCheck size={15}/>
        <div><strong>Observer only</strong><span>Studio displays the stored drift reports. It does not change anomaly scores, drift calculations, reference embeddings, or monitoring thresholds.</span></div>
      </div>
    </>}
  </>;
  </div>;
}

function LivePage({apiHealthy,projectId,projectAlgorithm,visible,onResult,onResults}:{apiHealthy:boolean;projectId:string;projectAlgorithm?:string;visible:boolean;onResult:(result:any,history:any[],config:any)=>void;onResults:()=>void}) {
  const [files,setFiles]=useState<File[]>([]); const [camera,setCamera]=useState(false); const [cameraBusy,setCameraBusy]=useState(false); const [cameraAuto,setCameraAuto]=useState(false); const [cameraError,setCameraError]=useState(""); const [cameraFps,setCameraFps]=useState(0); const [cameraFrames,setCameraFrames]=useState(0); const [cameraInterval,setCameraInterval]=useState(500); const videoRef=useRef<HTMLVideoElement>(null); const streamRef=useRef<MediaStream|null>(null); const cameraLoopRef=useRef<number|null>(null); const cameraBusyRef=useRef(false); const fpsTimesRef=useRef<number[]>([]); const [result,setResult]=useState<any>(null); const [busy,setBusy]=useState(false); const [error,setError]=useState(""); const [history,setHistory]=useState<any[]>([]); const [totalMs,setTotalMs]=useState(0);
  const inferenceUrl=process.env.NEXT_PUBLIC_ANOMAVISION_INFERENCE_URL ?? (typeof window !== "undefined" ? window.location.protocol + "//" + window.location.hostname + ":8001" : "http://localhost:8001");
  const [studioConfig,setStudioConfig]=useState<any>(null);
  const [inferenceHealth,setInferenceHealth]=useState<"checking"|"online"|"offline"|"wrong-service">("checking");
  const [inferenceHealthMessage,setInferenceHealthMessage]=useState(""); const activeThreshold=studioConfig?.thresholds?.[projectAlgorithm || studioConfig?.algorithm]??null;
  async function checkInferenceApi(){
    setInferenceHealth("checking"); setInferenceHealthMessage("Checking inference service…");
    try{
      const r=await fetch(inferenceUrl+"/health",{cache:"no-store"});
      const d=await r.json().catch(()=>({}));
      if(r.ok && (d.status==="healthy" || d.model_loaded===true)){
        setInferenceHealth("online"); setInferenceHealthMessage("Inference API is connected and the model is loaded."); return true;
      }
      if(r.ok && d.service==="anomavision-studio"){
        setInferenceHealth("wrong-service"); setInferenceHealthMessage("The configured inference URL points to the Studio API. Use port 8001 for inference.");
      }else{
        setInferenceHealth("offline"); setInferenceHealthMessage("Inference API responded but the model is not ready (HTTP "+r.status+"). Start inference on port 8001.");
      }
      return false;
    }catch{
      setInferenceHealth("offline"); setInferenceHealthMessage("Cannot reach the inference API at "+inferenceUrl+". Studio uses port 8000; inference uses port 8001. Start scripts/run_studio.ps1.");
      return false;
    }
  }
  async function reloadProjectModel(){
    if(!projectId)return false;
    try{
      const r=await fetch(inferenceUrl+"/reload-model?project_id="+encodeURIComponent(projectId),{method:"POST"});
      const d=await r.json().catch(()=>({}));
      if(!r.ok) throw new Error(d.detail || "The selected project has no loadable model yet.");
      setInferenceHealth("online");
      setInferenceHealthMessage("The selected project's trained model is loaded.");
      return true;
    }catch(e){
      const message=e instanceof TypeError
        ? "Cannot reach the inference service at "+inferenceUrl+". Studio uses port 8000 and inference uses port 8001."
        : (e instanceof Error ? e.message : "Could not load the selected project model.");
      setInferenceHealth("offline");
      setInferenceHealthMessage(message);
      return false;
    }
  }
  useEffect(()=>{
    fetch(`${API_BASE}/api/config`,{cache:"no-store"}).then(r=>r.ok?r.json():null).then(d=>setStudioConfig(d ? {...d, algorithm: projectAlgorithm || d.algorithm} : null)).catch(()=>setStudioConfig(null));
    void reloadProjectModel().then(loaded=>{
      if(!loaded) void checkInferenceApi();
    });
  },[projectId]);
  useEffect(()=>{if(result) onResult(result,history,studioConfig)},[result,history,studioConfig]);
  useEffect(()=>{
    if(!camera || !streamRef.current || !videoRef.current)return;
    const video=videoRef.current;
    video.srcObject=streamRef.current;
    void video.play().catch(()=>{});
  },[camera]);
  useEffect(()=>()=>{if(cameraLoopRef.current!==null)window.clearInterval(cameraLoopRef.current);streamRef.current?.getTracks().forEach(t=>t.stop())},[]);
  async function startCamera(){
    setCameraError("");
    try{
      if(!navigator.mediaDevices?.getUserMedia)throw new Error("Camera access is not available in this browser.");
      const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"},audio:false});
      streamRef.current=stream;
      setCamera(true);
    }catch(e){setCameraError(e instanceof Error?e.message:"Camera access denied");}
  }
  function stopAutoInference(){if(cameraLoopRef.current!==null)window.clearInterval(cameraLoopRef.current);cameraLoopRef.current=null;setCameraAuto(false);setCameraBusy(false);}
  function stopCamera(){stopAutoInference();streamRef.current?.getTracks().forEach(t=>t.stop());streamRef.current=null;setCamera(false);setCameraFps(0);}
  async function waitForCameraFrame(video:HTMLVideoElement){
    if(video.readyState<2 || !video.videoWidth || !video.videoHeight){
      await new Promise<void>((resolve,reject)=>{
        const timeout=window.setTimeout(()=>{cleanup();reject(new Error("Camera frame is not ready. Please wait a moment and try again."));},3000);
        const cleanup=()=>{window.clearTimeout(timeout);video.removeEventListener("loadeddata",ready);video.removeEventListener("canplay",ready);};
        const ready=()=>{if(video.videoWidth&&video.videoHeight){cleanup();resolve();}};
        video.addEventListener("loadeddata",ready);
        video.addEventListener("canplay",ready);
        if(video.readyState>=2&&video.videoWidth&&video.videoHeight)ready();
      });
    }
    if("requestVideoFrameCallback" in video){
      await new Promise<void>(resolve=>(video as HTMLVideoElement & {requestVideoFrameCallback:(cb:()=>void)=>number}).requestVideoFrameCallback(()=>resolve()));
    }
  }
  async function predictFrame(showBusy=true){if(!videoRef.current||cameraBusyRef.current)return;cameraBusyRef.current=true;if(showBusy)setCameraBusy(true);setCameraError("");try{const video=videoRef.current;await waitForCameraFrame(video);const canvas=document.createElement("canvas");const [targetW,targetH]=studioConfig?.resize||[224,224];canvas.width=targetW;canvas.height=targetH;canvas.getContext("2d")?.drawImage(video,0,0);const blob=await new Promise<Blob|null>(resolve=>canvas.toBlob(resolve,"image/jpeg",0.9));if(!blob)throw new Error("Could not capture camera frame");const body=new FormData();body.append("file",blob,"camera.jpg");const started=performance.now();const r=await fetch(inferenceUrl+"/predict?include_visualizations=false",{method:"POST",body});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.detail||"Inference API error ("+r.status+")");setInferenceHealth("online");setInferenceHealthMessage("Inference API is connected and responding.");const row={filename:"camera.jpg",...d};setResult(d);setHistory(prev=>[row,...prev].slice(0,20));const elapsed=performance.now()-started;setTotalMs(Number(d.latency_ms)||elapsed);setCameraFrames(prev=>prev+1);const now=Date.now();fpsTimesRef.current=[...fpsTimesRef.current.filter(t=>now-t<5000),now];setCameraFps(fpsTimesRef.current.length/5);}catch(e){const message=e instanceof TypeError?"Cannot reach inference API at "+inferenceUrl+". Check that the inference service is running on port 8001.":(e instanceof Error?e.message:"Camera inference failed");setInferenceHealth("offline");setInferenceHealthMessage(message);setCameraError(message)}finally{cameraBusyRef.current=false;if(showBusy)setCameraBusy(false)}}
  function startAutoInference(){if(!camera||cameraLoopRef.current!==null)return;setCameraAuto(true);void predictFrame(false);cameraLoopRef.current=window.setInterval(()=>void predictFrame(false),cameraInterval);}
  async function predictBatch(){
    if(!files.length)return;
    setBusy(true);setError("");setResult(null);
    try{
      const started=performance.now();
      const rows:any[]=[];
      for(const file of files.slice(0,10)){
        const body=new FormData();body.append("file",file,file.name);
        const r=await fetch(inferenceUrl+"/predict?include_visualizations=true",{method:"POST",body});
        const d=await r.json().catch(()=>({}));
        if(!r.ok)throw new Error(d.detail||"Inference API error ("+r.status+")");
        setInferenceHealth("online");setInferenceHealthMessage("Inference API is connected and responding.");
        rows.push({filename:file.name,...d});
      }
      const d=rows.length===1?rows[0]:{batch_results:rows.map(({filename,...result})=>({filename,result}))};
      setResult(d);
      setHistory(prev=>[...rows,...prev].slice(0,20));
      setTotalMs(rows.reduce((sum:number,x:any)=>sum+(Number(x.latency_ms)||0),0)||performance.now()-started);
    }catch(e){const message=e instanceof TypeError?"Cannot reach inference API at "+inferenceUrl+". Studio API is on port 8000; inference API must be on port 8001. Start scripts/run_studio.ps1.":(e instanceof Error?e.message:"Inference failed");setInferenceHealth("offline");setInferenceHealthMessage(message);setError(message)}
    finally{setBusy(false)}
  }
  const inferenceStatus = inferenceHealth==="online" ? "Online" : inferenceHealth==="checking" ? "Checking…" : "Unavailable";
  return <div className={visible ? "" : "inference-page-hidden"}><div className="page-head"><div><div className="eyebrow">Test & inference</div><h1>Inference</h1><p className="subtitle">Test images with the existing AnomaVision inference runtime. Camera mode stays available when you need continuous inspection.</p></div></div>
    <div className="live-top-stats"><Stat icon={<Wifi size={15}/>} label="Studio API" value={apiHealthy?"Online":"Offline"} meta="FastAPI connection"/><Stat icon={<CircleGauge size={15}/>} label="Inference API" value={inferenceStatus} meta={inferenceHealth==="online"?"Port 8001 · model ready":inferenceHealth==="checking"?"Checking port 8001…":"Expected port 8001"}/><Stat icon={<BrainCircuit size={15}/>} label="Method" value={String(projectAlgorithm || studioConfig?.algorithm||"—").toUpperCase()} meta={studioConfig?.resize?`${studioConfig.resize[0]}×${studioConfig.resize[1]} input`:"config.yml"}/><Stat icon={<ShieldCheck size={15}/>} label="Threshold" value={activeThreshold!=null?Number(activeThreshold).toFixed(3):"—"} meta="canonical config.yml"/><Stat icon={<CircleGauge size={15}/>} label="Batch latency" value={totalMs?`${totalMs.toFixed(0)} ms`:"—"} meta={totalMs?`${(1000/(totalMs/Math.max(1,files.length))).toFixed(1)} img/s`:"waiting"}/></div>
    {inferenceHealth!=="online"&&<div className="api-warning"><CircleGauge size={14}/><div><strong>Inference service unavailable</strong><span>{inferenceHealthMessage}</span><button className="text-button" onClick={()=>void checkInferenceApi()}>Check again</button></div></div>}
    <div className="card camera-panel"><div className="section-head"><div><div className="section-title">Camera stream</div><div className="subtitle">Continuous browser-camera inference using the existing runtime.</div></div><div className="badge">{cameraAuto?"Live inference":camera?"Camera on":"Off"}</div></div>{camera?<><div className="camera-frame"><video ref={videoRef} autoPlay playsInline muted className="camera-preview"/>{history[0]&&!history[0].error&&<div className={`camera-result ${(activeThreshold!=null?Number(history[0].anomaly_score)>=Number(activeThreshold):history[0].is_anomaly)?"anomaly":"normal"}`}><strong>{(activeThreshold!=null?Number(history[0].anomaly_score)>=Number(activeThreshold):history[0].is_anomaly)?"ANOMALY":"NORMAL"}</strong><span>Score {Number(history[0].anomaly_score||0).toFixed(3)}</span><small>Threshold {activeThreshold!=null?Number(activeThreshold).toFixed(3):"—"}</small></div>}</div><div className="grid-4"><Stat icon={<Activity size={15}/>} label="Live FPS" value={cameraFps.toFixed(1)} meta="processed frames / sec"/><Stat icon={<CircleGauge size={15}/>} label="Last latency" value={totalMs?totalMs.toFixed(0)+" ms":"—"} meta="latest frame"/><Stat icon={<ShieldCheck size={15}/>} label="Frames" value={String(cameraFrames)} meta="camera frames processed"/><Stat icon={<Wifi size={15}/>} label="Drift" value={history[0]?.drift_report?.drift_score!=null?Number(history[0].drift_report.drift_score).toFixed(3):history[0]?.drift_report?.status==="warming_up"?"Warming":"—"} meta={history[0]?.drift_report?.status==="warming_up"?"Collecting samples…":history[0]?.drift_report?.status||"not enabled"} green={history[0]?.drift_report?.status!=="drift"} alert={history[0]?.drift_report?.status==="drift"}/></div><div className="form-actions"><button className="primary live-control-button" onClick={cameraAuto?stopAutoInference:startAutoInference} disabled={cameraBusy&&!cameraAuto}><Play size={13}/><span>{cameraAuto?"Stop live inference":"Start live inference"}</span></button><button className="secondary live-control-button" onClick={()=>void predictFrame(true)} disabled={cameraBusy||cameraAuto}><span>{cameraBusy?"Analyzing…":"Analyze frame"}</span></button><label className="compact-control">Interval<select value={cameraInterval} onChange={e=>setCameraInterval(Number(e.target.value))} disabled={cameraAuto}><option value={250}>250 ms</option><option value={500}>500 ms</option><option value={1000}>1000 ms</option></select></label><button className="secondary" onClick={stopCamera}>Stop camera</button></div></>:<button className="secondary" onClick={startCamera}><Wifi size={13}/> Start camera</button>}{cameraError&&<div className="form-error">{cameraError}</div>}</div>
    <div className="card"><div className="section-head"><div><div className="section-title">Run inference</div><div className="subtitle">Select images or a folder. Results come from the existing <code>/predict-batch</code> runtime.</div></div></div>
      <div className="form-grid"><label className="field"><span>Images</span><input type="file" accept="image/*" multiple onChange={e=>setFiles(Array.from(e.target.files||[]).slice(0,10))}/></label><label className="field"><span>Folder</span><input type="file" accept="image/*" multiple {...({webkitdirectory:""} as any)} onChange={e=>setFiles(Array.from(e.target.files||[]).slice(0,10))}/></label></div>
      {files.length>0&&<div className="file-list">{files.map((f,i)=><span key={i} className="status-chip">{f.name}</span>)}</div>}
      <button className="primary" onClick={predictBatch} disabled={!files.length||busy}><Play size={13}/>{busy?"Analyzing…":files.length===1?"Analyze image":"Run batch inference"}</button>{error&&<div className="form-error">{error}</div>}
      {result&&<div className="deployment-result"><div className="performance-grid"><div><span>Processed</span><b>{result.batch_results?.length??0}</b></div><div><span>Anomalies</span><b>{history.slice(0,result.batch_results?.length??0).filter(x=>!x.error&&(activeThreshold!=null?Number(x.anomaly_score||0)>=Number(activeThreshold):x.is_anomaly)).length}</b></div><div><span>Threshold</span><b>{activeThreshold!=null?Number(activeThreshold).toFixed(3):"—"}</b></div><div><span>Drift</span><b>{history.some(x=>x.drift_report)?"Observed":"Not enabled"}</b></div></div></div>}
      {result&&<button className="secondary results-cta" onClick={onResults}><ShieldCheck size={13}/> Open detailed results</button>}
    </div>
    {history.length>0&&<div className="live-charts">
      <div className="card"><div className="section-head"><div><div className="section-title">Anomaly score</div><div className="subtitle">Latest session · threshold {activeThreshold!=null?Number(activeThreshold).toFixed(3):"not available"}.</div></div></div><div className="spark-bars">{history.slice(0,12).reverse().map((x,i)=>{const v=x.error?0:Math.min(1,Math.max(0,Number(x.anomaly_score)||0)/Math.max(Number(activeThreshold)||1,...history.slice(0,12).map(y=>Number(y.anomaly_score)||0),1));return <div className="spark-column" key={i} title={x.filename}><div className="spark-bar" style={{height:`${Math.max(4,v*100)}%`}}/><span>{i+1}</span></div>})}</div></div>
      <div className="card"><div className="section-head"><div><div className="section-title">Latency</div><div className="subtitle">Per-image inference time.</div></div></div><div className="spark-bars">{history.slice(0,12).reverse().map((x,i)=>{const v=x.error?0:Number(x.latency_ms)||0;const max=Math.max(...history.slice(0,12).map(y=>Number(y.latency_ms)||0),1);return <div className="spark-column" key={i} title={x.filename}><div className="spark-bar" style={{height:`${Math.max(4,v/max*100)}%`}}/><span>{i+1}</span></div>})}</div></div>
    </div>}
    <div className="card"><div className="section-head"><div><div className="section-title">Recent inference events</div><div className="subtitle">Latest results from this Studio session.</div></div></div>
      {history.length===0?<div className="empty-state">No inference events yet.</div>:<div className="table-wrap"><table className="data-table"><thead><tr><th>Image</th><th>Score</th><th>Latency</th><th>Drift</th><th>Status</th><th>Result</th></tr></thead><tbody>{history.map((x,i)=><tr key={i}><td>{x.filename}</td><td>{x.error?"—":Number(x.anomaly_score).toFixed(4)}</td><td>{x.error?"—":`${Number(x.latency_ms||0).toFixed(1)} ms`}</td><td>{x.drift_report?`${Number(x.drift_report.drift_score||0).toFixed(3)} · ${x.drift_report.status||"observed"}`:"—"}</td><td>{x.error?"Error":activeThreshold!=null?(Number(x.anomaly_score||0)>=Number(activeThreshold)?"Anomaly":"Normal"):(x.is_anomaly?"Anomaly":"Normal")}</td><td>{x.error||"Completed"}</td></tr>)}</tbody></table></div>}
    </div></div>; }function ResultsPage({session,onNavigate}:{session:{result:any;history:any[];config:any}|null;onNavigate:(p:Page)=>void}) {
  const result=session?.result;
  const history=session?.history||[];
  const config=session?.config;
  const threshold=config?.thresholds?.[config?.algorithm] ?? null;
  const rows=history.filter(x=>!x.error);
  const anomalies=rows.filter(x=>threshold!=null?Number(x.anomaly_score)>=Number(threshold):Boolean(x.is_anomaly));
  const latest=rows[0];

  if(!session || !result) return <div className="page-head">
    <div><div className="eyebrow">Results</div><h1>No results yet</h1><p className="subtitle">Run an image or batch inference first. Your latest session will appear here.</p></div>
    <button className="primary" onClick={()=>onNavigate("Inference")}><Play size={13}/> Run inference</button>
  </div>;

  return <>
    <div className="page-head">
      <div><div className="eyebrow">Inference results</div><h1>Results</h1><p className="subtitle">A clear view of the latest anomaly decisions, scores and visual evidence.</p></div>
      <button className="secondary" onClick={()=>onNavigate("Inference")}><Play size={13}/> New inference</button>
    </div>

    <div className="grid-4 section">
      <Stat icon={<Database size={15}/>} label="Processed" value={String(rows.length)} meta="images completed"/>
      <Stat icon={<ShieldCheck size={15}/>} label="Anomalies" value={String(anomalies.length)} meta={anomalies.length?"review required":"no anomalies detected"} green={!anomalies.length}/>
      <Stat icon={<CircleGauge size={15}/>} label="Threshold" value={threshold!=null?Number(threshold).toFixed(3):"—"} meta="from config.yml"/>
      <Stat icon={<Activity size={15}/>} label="Latest score" value={latest?Number(latest.anomaly_score).toFixed(3):"—"} meta={latest?(threshold!=null&&Number(latest.anomaly_score)>=Number(threshold)?"Anomaly":"Normal"):"waiting"}/>
    </div>

    {latest && <section className="card result-hero">
      <div className="section-head"><div><div className="section-title">Latest result</div><div className="subtitle">{latest.filename}</div></div><span className={`result-status ${threshold!=null&&Number(latest.anomaly_score)>=Number(threshold)?"anomaly":"normal"}`}>{threshold!=null&&Number(latest.anomaly_score)>=Number(threshold)?"ANOMALY":"NORMAL"}</span></div>
      <div className="result-score"><strong>{Number(latest.anomaly_score).toFixed(4)}</strong><span>Anomaly score</span><small>Decision threshold: {threshold!=null?Number(threshold).toFixed(4):"not available"}</small></div>
      <div className="batch-result-list">
        {rows.map((r:any,index:number)=>{
          const isAnomaly=threshold!=null ? Number(r.anomaly_score)>=Number(threshold) : Boolean(r.is_anomaly);
          return <article className="batch-result-item" key={r.filename||index}>
            <div className="batch-result-head">
              <div><strong>{r.filename||`Image ${index+1}`}</strong><span>Score {Number(r.anomaly_score||0).toFixed(4)} · {Number(r.latency_ms||0).toFixed(1)} ms</span></div>
              <span className={`result-status ${isAnomaly?"anomaly":"normal"}`}>{isAnomaly?"ANOMALY":"NORMAL"}</span>
            </div>
            <div className="result-visual-grid">
              {r.heatmap_image_base64 && <div className="result-visual"><span>Heatmap</span><img src={`data:image/png;base64,${r.heatmap_image_base64}`} alt={`Anomaly heatmap for ${r.filename||"image"}`}/></div>}
              {r.boundary_image_base64 && <div className="result-visual"><span>Boundary</span><img src={`data:image/png;base64,${r.boundary_image_base64}`} alt={`Anomaly boundary for ${r.filename||"image"}`}/></div>}
              {r.highlighted_image_base64 && <div className="result-visual"><span>Highlighted regions</span><img src={`data:image/png;base64,${r.highlighted_image_base64}`} alt={`Highlighted anomaly regions for ${r.filename||"image"}`}/></div>}
              {!r.heatmap_image_base64&&!r.boundary_image_base64&&!r.highlighted_image_base64&&<div className="result-no-visual">No visual evidence was returned for this image.</div>}
            </div>
          </article>;
        })}
      </div>
    </section>}

    <section className="card">
      <div className="section-head"><div><div className="section-title">Inference history</div><div className="subtitle">Results from the current Studio session.</div></div></div>
      <div className="table-wrap"><table className="data-table"><thead><tr><th>Image</th><th>Score</th><th>Latency</th><th>Decision</th><th>Status</th></tr></thead><tbody>
        {history.map((x,i)=>{const anomaly=!x.error&&(threshold!=null?Number(x.anomaly_score)>=Number(threshold):Boolean(x.is_anomaly));return <tr key={i}><td>{x.filename}</td><td>{x.error?"—":Number(x.anomaly_score).toFixed(4)}</td><td>{x.error?"—":`${Number(x.latency_ms||0).toFixed(1)} ms`}</td><td>{x.error?"—":anomaly?"Anomaly":"Normal"}</td><td>{x.error?"Error":"Completed"}</td></tr>})}
      </tbody></table></div>
    </section>

    <details className="technical-details card">
      <summary>Technical details</summary>
      <div className="technical-grid">
        <div><span>Algorithm</span><b>{String(config?.algorithm||"—").toUpperCase()}</b></div>
        <div><span>Input size</span><b>{config?.resize?.join(" × ")||"—"}</b></div>
        <div><span>Threshold</span><b>{threshold!=null?Number(threshold).toFixed(4):"—"}</b></div>
        <div><span>Drift</span><b>{history.some(x=>x.drift_report)?"Observed":"Not enabled"}</b></div>
      </div>
    </details>
  </>;
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

function Placeholder({page}:{page:Page}) { const descriptions:Record<Page,string>={Overview:"",Performance:"",Projects:"",Datasets:"",Training:"",Models:"",Deployments:"Export and validate models for production targets without changing the algorithm core.",Live:"Run continuous camera inference using the existing AnomaVision runtime.",Inference:"Test images and camera frames with the existing inference runtime.",Results:"Review anomaly scores, decisions and visual evidence.",Monitoring:"Track runtime health, latency and production data drift.",Settings:"View the canonical Studio configuration."};return <><div className="page-head"><div><div className="eyebrow">Workspace</div><h1>{page}</h1><p className="subtitle">{descriptions[page]}</p></div><button className="secondary"><SlidersHorizontal size={13}/> Configure</button></div><div className="card placeholder"><div className="icon-box"><Sparkles size={18}/></div><div><b>Connected to the Studio architecture</b><p className="subtitle">This view is ready to consume the same Python services through the Studio API. No ML logic is duplicated in the frontend.</p></div></div></>; }

function Stat({icon,label,value,meta,green,alert}:{icon:React.ReactNode;label:string;value:string;meta:string;green?:boolean;alert?:boolean}){return <div className={`card stat-card ${alert?"drift-alert":""}`} tabIndex={0}><div className="stat-label">{icon}<span>{label}</span></div><div className="stat-value">{value}</div><div className="stat-meta">{green&&<span className="status-dot"/>}{meta}</div></div>}
function ActivityRow({icon,title,sub,badge}:{icon:React.ReactNode;title:string;sub:string;badge:string}){return <div className="row"><div className="row-main"><div className="icon-box">{icon}</div><div><div className="row-title">{title}</div><div className="row-sub">{sub}</div></div></div><div className="badge">{badge}</div></div>}
function HealthRow({title,sub,ok}:{title:string;sub:string;ok:boolean}){return <div className="row"><div className="row-main"><div className="icon-box"><CircleGauge size={14}/></div><div><div className="row-title">{title}</div><div className="row-sub">{sub}</div></div></div><div className="badge"><span className={`status-dot ${ok?"":"offline-dot"}`}/>{ok?"Healthy":"Offline"}</div></div>}
