"""AnomaVision Studio MVP.

Run:
    streamlit run apps/studio/app.py

Studio is intentionally an orchestration/UI layer. It does not reimplement
PaDiM, PatchCore, inference backends, deployment validation, or drift logic.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from apps.studio.services.catalog import ALGORITHMS, DEPLOYMENT_TARGETS
from apps.studio.services.project_store import ProjectStore

st.set_page_config(
    page_title="AnomaVision Studio",
    page_icon="AV",
    layout="wide",
    initial_sidebar_state="expanded",
)

STORE_ROOT = Path(os.getenv("ANOMAVISION_STUDIO_ROOT", "~/.anomavision/projects")).expanduser()
store = ProjectStore(STORE_ROOT)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1400px; padding-top: 2rem; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
        .studio-card {
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 12px;
            padding: 1.1rem;
            min-height: 130px;
            background: rgba(128,128,128,.035);
        }
        .studio-muted { color: rgba(128,128,128,.9); font-size: .9rem; }
        .studio-status { font-weight: 650; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="studio-card"><div class="studio-muted">{title}</div>'
        f'<div style="font-size:1.8rem;font-weight:700;margin:.35rem 0">{value}</div>'
        f'<div class="studio-muted">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def sidebar() -> str:
    with st.sidebar:
        st.markdown("# AnomaVision Studio")
        st.caption("Train · Validate · Deploy · Inspect · Monitor")
        page = st.radio(
            "Workspace",
            ["Dashboard", "Projects", "Datasets", "Models", "Deployments", "Live", "Monitoring", "Settings"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Storage\n`{STORE_ROOT}`")
        return page


def dashboard() -> None:
    projects = store.list_projects()
    st.title("Dashboard")
    st.caption("Your local anomaly-detection workspace.")

    cols = st.columns(4)
    with cols[0]: metric_card("Projects", str(len(projects)), "local projects")
    with cols[1]: metric_card("Models", "0", "ready to deploy")
    with cols[2]: metric_card("Deployments", "0", "active runtimes")
    with cols[3]: metric_card("Health", "Ready", "Studio storage online")

    st.subheader("Workflow")
    flow = st.columns(5)
    for col, label in zip(flow, ["01  Data", "02  Train", "03  Validate", "04  Deploy", "05  Monitor"]):
        with col:
            st.markdown(f'<div class="studio-card"><b>{label}</b><div class="studio-muted">Next step in the Studio workflow</div></div>', unsafe_allow_html=True)

    st.subheader("Recent projects")
    if not projects:
        st.info("No projects yet. Create your first project from Projects.")
        return
    for project in projects[:5]:
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.write(project["name"])
        c2.caption(f'{project["algorithm"].upper()} · {project["status"]}')
        if c3.button("Open", key=f'open-{project["id"]}'):
            st.session_state["selected_project"] = project["id"]
            st.session_state["page"] = "Projects"
            st.rerun()


def projects_page() -> None:
    st.title("Projects")
    st.caption("Each project contains its dataset, experiments, models, deployments and monitoring state.")

    with st.expander("+ Create project", expanded=not bool(store.list_projects())):
        with st.form("create-project"):
            name = st.text_input("Project name", placeholder="Bottle Cap Inspection")
            algorithm = st.selectbox("Initial algorithm", list(ALGORITHMS), format_func=lambda key: ALGORITHMS[key]["name"])
            description = st.text_area("Description", placeholder="Detect abnormal bottle caps")
            submitted = st.form_submit_button("Create project", type="primary")
            if submitted:
                if not name.strip():
                    st.error("Project name is required.")
                else:
                    try:
                        project = store.create(name, algorithm)
                        project["description"] = description.strip()
                        project_path = STORE_ROOT / project["id"] / "project.json"
                        import json
                        project_path.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")
                        st.success(f'Created "{project["name"]}".')
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

    projects = store.list_projects()
    if not projects:
        return
    st.subheader("Your projects")
    for project in projects:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.subheader(project["name"])
            c1.caption(project.get("description") or "No description")
            c2.write(f'**Algorithm:** {project["algorithm"].upper()}')
            c2.write(f'**Status:** {project["status"]}')
            if c3.button("Select", key=f'select-{project["id"]}'):
                st.session_state["selected_project"] = project["id"]
            if st.session_state.get("selected_project") == project["id"]:
                st.success("Selected project")
                st.write("Next: connect a dataset, then train a model.")


def datasets_page() -> None:
    st.title("Datasets")
    st.caption("Inspect an image folder before it enters the training workflow.")

    selected = st.session_state.get("selected_project")
    projects = store.list_projects()
    if not selected and projects:
        selected = projects[0]["id"]
    if projects:
        ids = [p["id"] for p in projects]
        selected = st.selectbox(
            "Project",
            ids,
            index=ids.index(selected) if selected in ids else 0,
            format_func=lambda pid: next(p["name"] for p in projects if p["id"] == pid),
        )
        st.session_state["selected_project"] = selected

    source = st.text_input(
        "Image folder",
        placeholder=r"C:\datasets\bottle\train\good",
        help="A local folder containing PNG, JPG, JPEG, BMP or WebP images. Subfolders are included.",
    )
    analyze = st.button("Analyze dataset", type="primary", disabled=not source.strip())

    if analyze:
        from apps.studio.services.dataset_service import inspect_dataset, save_dataset_manifest
        try:
            report = inspect_dataset(source)
            if selected:
                manifest = STORE_ROOT / selected / "datasets" / "dataset_report.json"
                save_dataset_manifest(manifest, report)
            st.session_state["dataset_report"] = report
            st.success(f'Analyzed {report["image_count"]} image(s).')
        except ValueError as exc:
            st.error(str(exc))

    report = st.session_state.get("dataset_report")
    if not report:
        st.info("Enter an image folder and analyze it. The report is saved inside the selected project.")
        return

    cols = st.columns(4)
    with cols[0]:
        metric_card("Images", str(report["image_count"]), "valid images")
    with cols[1]:
        metric_card("Duplicates", str(report["duplicate_count"]), "extra copies")
    with cols[2]:
        metric_card("Failed", str(report["failed_count"]), "could not be opened")
    with cols[3]:
        resolution = next(iter(report["resolutions"]), "—")
        metric_card("Top resolution", resolution, "most common")

    st.subheader("Dataset quality")
    if report["failed_count"] == 0 and report["duplicate_count"] == 0:
        st.success("No duplicate or unreadable images detected.")
    else:
        if report["failed_count"]:
            st.warning(f'{report["failed_count"]} image(s) could not be opened.')
        if report["duplicate_count"]:
            st.warning(f'{report["duplicate_count"]} duplicate image(s) detected.')

    left, right = st.columns(2)
    with left:
        st.subheader("Resolutions")
        st.dataframe(
            [{"resolution": key, "images": count} for key, count in report["resolutions"].items()],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Image preview")
        records = report["records"][:12]
        if records:
            preview_root = Path(report["root"])
            preview_paths = [preview_root / record["path"] for record in records]
            st.image([str(path) for path in preview_paths], width=150)
        else:
            st.caption("No readable images to preview.")

    if report["duplicate_groups"]:
        with st.expander("Duplicate groups"):
            for group in report["duplicate_groups"]:
                st.write(group)

    with st.expander("Image inventory"):
        st.dataframe(
            [
                {
                    "file": record["path"],
                    "size": f'{record["width"]} × {record["height"]}',
                    "MB": round(record["bytes"] / (1024 * 1024), 2),
                    "brightness": record["brightness"],
                }
                for record in report["records"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def models_page() -> None:
    st.title("Models")
    st.caption("Model artifacts will be registered here after training and evaluation.")
    st.info("No Studio model registry entries yet.")
    st.subheader("Algorithms")
    for key, item in ALGORITHMS.items():
        with st.container(border=True):
            st.write(f'**{item["name"]}**')
            st.caption(item["description"])


def deployments_page() -> None:
    st.title("Deployments")
    st.caption("Deployment targets reuse AnomaVision's existing export and validation infrastructure.")
    for key, label in DEPLOYMENT_TARGETS.items():
        st.checkbox(label, value=(key == "cpu"), disabled=True, key=f"target-{key}")
    st.info("Deployment orchestration is intentionally not implemented yet. The existing deployment-validation engine remains the source of truth.")


def live_page() -> None:
    st.title("Live Inspection")
    st.caption("Live camera/stream inference will use the existing AnomaVision runtime.")
    st.info("Live runtime wiring is the next Studio milestone. No duplicate inference engine is being introduced.")


def monitoring_page() -> None:
    st.title("Monitoring")
    st.caption("Production health, latency and data drift.")
    cols = st.columns(4)
    with cols[0]: metric_card("Runtime", "Idle", "no deployment selected")
    with cols[1]: metric_card("FPS", "—", "live runtime")
    with cols[2]: metric_card("Latency", "—", "milliseconds")
    with cols[3]: metric_card("Drift", "—", "reference comparison")
    st.info("The existing drift monitor remains unchanged. Studio will consume its observer-only status and metrics here.")


def settings_page() -> None:
    st.title("Settings")
    st.caption("Local Studio configuration.")
    st.text_input("Project storage", value=str(STORE_ROOT), disabled=True)
    st.text_input("Core package", value="anomavision", disabled=True)
    st.text_input("Studio version", value="0.1.0", disabled=True)


inject_style()
page = sidebar()
if page == "Dashboard": dashboard()
elif page == "Projects": projects_page()
elif page == "Datasets": datasets_page()
elif page == "Models": models_page()
elif page == "Deployments": deployments_page()
elif page == "Live": live_page()
elif page == "Monitoring": monitoring_page()
else: settings_page()
