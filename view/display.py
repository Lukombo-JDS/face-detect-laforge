import time

import dotenv
import streamlit as st
from PIL import Image

from interface import detect_faces

# Configuration de la page
st.set_page_config(page_title="Annotateur de Visages", layout="wide")

output_dir = dotenv.get_key(".env", "OUTPUT_IMAGES_DIR")

# --- Gestion de l'état de l'application ---
if "step" not in st.session_state:
    st.session_state.step = "upload"
if "faces" not in st.session_state:
    st.session_state.faces = []
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "annotations" not in st.session_state:
    st.session_state.annotations = {}


def reset_app():
    keys: list = list(st.session_state.keys())
    for key in keys:
        if key.startswith("face_"):
            del st.session_state[key]
    st.session_state.step = "upload"
    st.session_state.faces = []
    st.session_state.original_image = None
    st.session_state.annotations = {}
    st.rerun()


# --- Interface Utilisateur ---

st.title("👤 Annotateur de Visages")

# ÉTAPE 1 : Upload de l'image
if st.session_state.step == "upload":
    st.subheader("Téléchargez une photo pour commencer")
    uploaded_file = st.file_uploader(
        "Choisir une image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.original_image = image

        with st.spinner("Détection des visages en cours..."):
            if output_dir:
                detected = detect_faces(image, output_dir)
                if detected:
                    st.session_state.faces = detected
                    st.session_state.step = "annotate"
                    st.rerun()
                else:
                    st.error(
                        "Aucun visage n'a été détecté sur cette image. Essayez-en une autre."
                    )

# ÉTAPE 2 : Annotation des vignettes
elif st.session_state.step == "annotate":
    tab1, tab2 = st.tabs(["📝 Annotation", "🔍 Visages Identifiés"])

    with tab1:
        st.subheader("Saisie rapide")
        st.caption("Les vignettes sont optimisées pour minimiser le défilement.")

        # Utilisation de 10 colonnes pour réduire drastiquement la largeur de chaque élément
        cols = st.columns(10)

        for i, face_info in enumerate(st.session_state.faces):
            col_idx = i % 10
            with cols[col_idx]:
                # Affichage de la vignette (taille réduite par thumbnail + largeur colonne)
                st.image(face_info["image"])

                # Champ de saisie compact
                label = st.text_input(
                    f"ID:{i + 1}",
                    key=f"face_{i}",
                    placeholder="Nom",
                    label_visibility="collapsed",
                )

                if label:
                    st.session_state.annotations[i] = label
                elif i in st.session_state.annotations:
                    del st.session_state.annotations[i]

                st.divider()

    with tab2:
        st.subheader("Récapitulatif")
        identified_count = len(st.session_state.annotations)

        if identified_count == 0:
            st.warning("Aucun visage n'a été annoté.")
        else:
            res_cols = st.columns(12)
            idx = 0
            for i, name in st.session_state.annotations.items():
                if name.strip():
                    with res_cols[idx % 12]:
                        st.image(
                            st.session_state.faces[i]["image"], use_container_width=True
                        )
                        st.caption(name)
                    idx += 1

    st.markdown("---")

    # Zone d'action
    col_btn1, col_btn2, _ = st.columns([2, 1, 4])

    with col_btn1:
        can_validate = any(v.strip() for v in st.session_state.annotations.values())

        if st.button(
            "✅ Valider et Terminer",
            disabled=not can_validate,
            use_container_width=True,
        ):
            st.success(f"{len(st.session_state.annotations)} visages enregistrés.")
            time.sleep(1)
            reset_app()

    with col_btn2:
        if st.button("❌ Annuler", use_container_width=True):
            reset_app()

# Sidebar
with st.sidebar:
    st.header("Image Source")
    if st.session_state.original_image:
        st.image(st.session_state.original_image, use_container_width=True)
        st.write(f"Visages détectés : {len(st.session_state.faces)}")
