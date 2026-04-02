from __future__ import annotations

import time

import streamlit as st
from PIL import Image

from app.config import SETTINGS
from app.services.face_pipeline import FacePipelineService, ProcessedFace
from app.workers.background import BackgroundTaskRunner


def _init_state() -> None:
    defaults = {
        "history_images": [],
        "history_faces": [],
        "step": "upload",
        "processed_faces": [],
        "annotations": {},
        "deleted_faces": set(),
        "original_image": None,
        "pipeline": FacePipelineService(),
        "runner": BackgroundTaskRunner(),
        "task_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _submit_processing(image_bytes: bytes) -> None:
    task_id = st.session_state.runner.submit(
        st.session_state.pipeline.process_uploaded_image,
        image_bytes,
    )
    st.session_state.task_id = task_id


def _poll_processing_task() -> None:
    task_id = st.session_state.task_id
    if not task_id:
        return

    task = st.session_state.runner.status(task_id)
    if task.status in {"queued", "running"}:
        st.info("Traitement en arrière-plan en cours…")
        if st.button("Rafraîchir le statut"):
            st.rerun()
        return

    if task.status == "failed":
        st.error(f"Le traitement a échoué: {task.error}")
        st.session_state.task_id = None
        return

    if task.status == "done":
        st.session_state.processed_faces = task.result
        st.session_state.step = "annotate"
        st.session_state.deleted_faces = set()
        st.session_state.task_id = None
        st.rerun()


def _save_and_reset() -> None:
    st.session_state.pipeline.save_annotations(
        st.session_state.processed_faces,
        st.session_state.annotations,
        source_image=f"upload_{int(time.time())}.jpg",
        deleted_indices=st.session_state.deleted_faces,
    )

    for idx, label in st.session_state.annotations.items():
        if idx in st.session_state.deleted_faces:
            continue
        clean_label = label.strip() or SETTINGS.unknown_label
        st.session_state.history_faces.append(
            {
                "name": clean_label,
                "image": st.session_state.processed_faces[idx].face_image,
                "date": time.strftime("%H:%M:%S"),
            }
        )

    if st.session_state.original_image is not None:
        st.session_state.history_images.append(st.session_state.original_image)

    st.session_state.step = "upload"
    st.session_state.processed_faces = []
    st.session_state.annotations = {}
    st.session_state.deleted_faces = set()
    st.session_state.original_image = None


def _render_upload_tab() -> None:
    if not st.session_state.pipeline.ensure_milvus_connection():
        error = st.session_state.pipeline.milvus_error or "connexion impossible"
        st.warning(
            "Milvus indisponible: suggestions de similarité et indexation désactivées "
            f"(détail: {error})."
        )

    uploaded_file = st.file_uploader("Choisir une image…", type=["jpg", "jpeg", "png"])
    _poll_processing_task()

    if uploaded_file and st.button("Lancer détection + vectorisation"):
        st.session_state.original_image = Image.open(uploaded_file)
        _submit_processing(uploaded_file.getvalue())
        st.rerun()


def _render_gallery_tab() -> None:
    search_query = st.text_input("Filtrer par nom…", "")
    faces = [
        f
        for f in st.session_state.history_faces
        if search_query.lower() in f["name"].lower()
    ]
    if not faces:
        st.info("Aucun visage dans la galerie.")
        return

    columns = st.columns(4)
    for idx, item in enumerate(faces):
        with columns[idx % 4]:
            st.image(item["image"], channels="BGR", use_container_width=True)
            st.caption(item["name"])


def _render_annotation_view() -> None:
    st.subheader("Annotation")
    similarity_enabled = st.session_state.pipeline.milvus_available
    if not similarity_enabled:
        st.info("Mode dégradé actif: suggestions Milvus et indexation désactivées.")
    for idx, processed in enumerate(st.session_state.processed_faces):
        st.image(processed.face_image, channels="BGR", width=180)
        suggested_labels = [x.annotation for x in processed.suggestions]
        default_label = suggested_labels[0] if suggested_labels else SETTINGS.unknown_label
        if similarity_enabled:
            suggestions = ", ".join(
                [
                    f"{x.person_name} ({x.similarity_percent:.1f}%)"
                    for x in processed.suggestions
                ]
            )
            st.caption(f"Suggestions Milvus: {suggestions or 'aucune'}")
        else:
            st.caption("Suggestions Milvus: indisponibles (Milvus non connecté)")
        should_delete = st.checkbox(
            f"Supprimer ce visage de l'annotation (#{idx + 1})",
            value=idx in st.session_state.deleted_faces,
            key=f"delete_face_{idx}",
        )
        if should_delete:
            st.session_state.deleted_faces.add(idx)
        else:
            st.session_state.deleted_faces.discard(idx)

        manual_option = "✍️ Saisie manuelle"
        select_options = suggested_labels + [manual_option]
        saved_value = st.session_state.annotations.get(idx, default_label)
        default_select = saved_value if saved_value in suggested_labels else manual_option
        selected = st.selectbox(
            f"Suggestion visage #{idx + 1}",
            options=select_options,
            index=select_options.index(default_select),
            key=f"suggestion_face_{idx}",
            disabled=should_delete,
        )
        if selected == manual_option:
            st.session_state.annotations[idx] = st.text_input(
                f"Label visage #{idx + 1}",
                value=saved_value if saved_value not in suggested_labels else "",
                key=f"custom_label_{idx}",
                disabled=should_delete,
            )
        else:
            st.session_state.annotations[idx] = selected

    if st.button("Valider"):
        _save_and_reset()
        st.success("Annotations enregistrées")
        st.rerun()


def run() -> None:
    st.set_page_config(page_title="Annotateur de visages", layout="wide")
    st.title("👤 Annotateur de Visages")
    _init_state()

    if st.session_state.step == "upload":
        tab_upload, tab_gallery, tab_history = st.tabs(
            ["📤 Upload", "🗂️ Galerie", "🖼️ Historique"]
        )
        with tab_upload:
            _render_upload_tab()
        with tab_gallery:
            _render_gallery_tab()
        with tab_history:
            for img in st.session_state.history_images:
                st.image(img, use_container_width=True)
    else:
        _render_annotation_view()
