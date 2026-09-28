      </div>
    </section>
  </>;
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

function TrainingPage({ project, onFinished }: { project?: Project; onFinished:()=>Promise<void> }) {
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
  const [pickerBusy,setPickerBusy]=useState(false);
  const [resolved,setResolved]=useState<{dataset_path:string;class_name:string;train_good:string}|null>(null);

  useEffect(()=>{if(project)setAlgorithm(project.algorithm)},[project]);

  useEffect(()=>{
    if(!project || typeof window==="undefined") return;
    const key="anomavision:training-job:"+project.id;
    const saved=window.localStorage.getItem(key);
    if(!saved) return;
    let cancelled=false;
    let job:any;
    try{job=JSON.parse(saved)}catch{window.localStorage.removeItem(key);return}
    setJobId(String(job.job_id||"")); setBusy(true); setJobStatus("Resuming training…");
    const poll=async()=>{
      try{
        const r=await fetch(`${API_BASE}/api/projects/${project.id}/training/${job.job_id}`,{cache:"no-store"});
        const d=await r.json().catch(()=>({}));
        if(!r.ok)throw new Error(d.detail||"Could not restore training job");
        if(cancelled)return;
        setJobStatus(d.message||d.status||"Training…");
        if(d.status==="completed"){setResult(d.result||{});setBusy(false);window.localStorage.removeItem(key);await onFinished();return}
        if(d.status==="failed"){setError(d.message||"Training failed");setBusy(false);window.localStorage.removeItem(key);return}
        window.setTimeout(poll,1200);
      }catch(e){if(!cancelled){setError(e instanceof Error?e.message:"Could not restore training job");setBusy(false)}}
    };
    void poll();
    return()=>{cancelled=true};
  },[project?.id]);


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
  }

  async function train(){
    if(!project){setError("Select a project first.");return;}
    if(!dataset.trim()){setError("Add your dataset path first.");return;}
    setBusy(true);setError("");setResult(null);setJobStatus("Validating dataset…");
    try{
      const canonical=await validateDataset();
      if(!canonical)return;
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
      if(!d.job_id){setResult(d);await onFinished();return;}
      setJobId(d.job_id);setJobStatus("Training queued…");
      if(typeof window!=="undefined") window.localStorage.setItem("anomavision:training-job:"+project.id,JSON.stringify({job_id:d.job_id}));
      for(;;){
        await new Promise(resolve=>window.setTimeout(resolve,1200));
        const statusResponse=await fetch(`${API_BASE}/api/projects/${project.id}/training/${d.job_id}`,{cache:"no-store"});
        const status=await statusResponse.json().catch(()=>({}));
        if(!statusResponse.ok)throw new Error(status.detail||"Could not read training status");
        setJobStatus(status.message||status.status||"Training…");
        if(status.status==="completed"){setResult(status.result||{});setJobStatus("Training completed");if(typeof window!=="undefined")window.localStorage.removeItem("anomavision:training-job:"+project.id);await onFinished();break;}
        if(status.status==="failed"){if(typeof window!=="undefined")window.localStorage.removeItem("anomavision:training-job:"+project.id);throw new Error(status.message||"Training failed");}
      }
    }catch(e){setError(e instanceof Error?e.message:"Training failed");setJobStatus("Training stopped")}
    finally{setBusy(false)}
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
              <label>Dataset path<input value={dataset} onChange={e=>{setDataset(e.target.value);setResolved(null)}} placeholder={configLoaded?"From config.yml":"Loading config…"}/></label>
              <div className="form-actions">
                <button className="secondary" type="button" onClick={chooseTrainingFolder} disabled={pickerBusy}><FolderOpen size={13}/>{pickerBusy?"Opening…":"Choose folder"}</button>
                <button className="secondary" type="button" onClick={validateDataset} disabled={busy||!dataset.trim()}>Check training layout</button>
              </div>
              <label>Class name<input value={className} onChange={e=>{setClassName(e.target.value);setResolved(null)}} placeholder={configLoaded?"From config.yml":"Loading config…"}/></label>
              {resolved&&<div className="config-field-note"><span>✓ Training folder: <code>{resolved.train_good}</code></span></div>}
              <div className="config-field-note">{className ? <span>Using <strong>{className}</strong> from {configLoaded ? "config.yml" : "the current setup"}.</span> : <span>Class name will be taken from <strong>config.yml</strong> if you leave it empty.</span>}</div>
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
            <button className="primary" onClick={train} disabled={busy}><Play size={13}/>{busy?(jobStatus||"Working…"):"Start training"}</button>
          </div>

          {error && <div className="form-error">{error}</div>}

          {result && (
            <section className="card training-result">
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
              {(result.model_path || result.path) && (
                <div className="artifact-box">
                  <span>Model artifact</span>
                  <code>{String(result.model_path ?? result.path)}</code>
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