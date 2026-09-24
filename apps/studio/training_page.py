"""Training workspace for AnomaVision Studio."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from apps.studio.services.training_service import train_project


def render_training_page(store, store_root: Path, algorithms: dict) -> None:
    st.title("Training")
    st.caption("Train with AnomaVision's existing PaDiM/PatchCore implementation.")

    projects = store.list_projects()
    if not projects:
        st.info("Create a project first.")
        return

    selected = st.session_state.get("selected_project") or projects[0]["id"]
    ids = [p["id"] for p in projects]
    selected = st.selectbox(
        "Project",
        ids,
        index=ids.index(selected) if selected in ids else 0,
        format_func=lambda pid: next(p["name"] for p in projects if p["id"] == pid),
    )
    st.session_state["selected_project"] = selected

    project = store.get(selected)
    project_dir = store_root / selected

    default_source = ""
    report = st.session_state.get("dataset_report")
    if report:
        default_source = report.get("root", "")

    source = st.text_input(
        "Dataset",
        value=default_source,
        placeholder=r"C:\datasets\bottle",
        help="Select a dataset root or class folder containing train/good images.",
    )

    algorithm = st.selectbox(
        "Algorithm",
        list(algorithms),
        index=list(algorithms).index(project.get("algorithm", "patchcore"))
        if project.get("algorithm", "patchcore") in algorithms
        else 0,
        format_func=lambda key: algorithms[key]["name"],
    )

    class_name = st.text_input(
        "Class name",
        value="bottle",
        help="Used when the dataset root contains train/good directly.",
    )

    with st.expander("Training options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            backbone = st.selectbox("Backbone", ["resnet18", "wide_resnet50"])
        with col2:
            batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=2)
        with col3:
            resize = st.number_input("Image size", min_value=64, max_value=1024, value=224)

        if algorithm == "padim":
            feat_dim = st.number_input("PaDiM feature dimensions", min_value=1, max_value=1024, value=50)
            coreset_ratio = 0.02
        else:
            feat_dim = 50
            coreset_ratio = st.number_input(
                "PatchCore coreset ratio",
                min_value=0.001,
                max_value=1.0,
                value=0.02,
                step=0.005,
            )

    train_clicked = st.button(
        "Start training",
        type="primary",
        disabled=not source.strip(),
    )

    if train_clicked:
        try:
            with st.spinner("Training model..."):
                metadata = train_project(
                    project_dir=project_dir,
                    dataset_source=source,
                    algorithm=algorithm,
                    class_name=class_name.strip() or "default",
                    backbone=backbone,
                    batch_size=int(batch_size),
                    resize=[int(resize), int(resize)],
                    feat_dim=int(feat_dim),
                    coreset_ratio=float(coreset_ratio),
                )
            st.session_state["training_result"] = metadata
            st.success("Training completed.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    result = st.session_state.get("training_result")
    if result:
        st.subheader("Latest training run")
        st.json(result)
