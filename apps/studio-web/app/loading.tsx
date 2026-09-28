export default function Loading() {
  return (
    <div className="studio-shell" aria-busy="true">
      <aside className="sidebar"><div className="brand"><div className="brand-mark">AV</div><div><div className="brand-name">AnomaVision</div><div className="brand-sub">Studio · loading</div></div></div></aside>
      <main className="main"><header className="topbar"><div className="breadcrumbs"><span>AnomaVision</span><b>/</b><strong>Loading</strong></div></header>
        <div className="content"><div className="page-head"><div><div className="eyebrow">Workspace</div><h1>Loading Studio</h1><p className="subtitle">Connecting to your AnomaVision workspace…</p></div></div>
          <div className="grid-4">{[1,2,3,4].map(i=><div className="card stat-card" key={i}><div className="skeleton" style={{height:12,width:"42%"}}/><div className="skeleton" style={{height:28,width:"58%",marginTop:14}}/><div className="skeleton" style={{height:10,width:"72%",marginTop:10}}/></div>)}</div>
          <div className="card" style={{height:260,marginTop:24}}><div className="skeleton" style={{height:18,width:"28%"}}/><div className="skeleton" style={{height:12,width:"52%",marginTop:14}}/></div>
        </div>
      </main>
    </div>
  );
}
