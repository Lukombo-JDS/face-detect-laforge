import io
import os
import time

import cv2
import dotenv
import numpy as np
import streamlit as st
from PIL import Image

from interface import detect_faces

# Configuration de la page
st.set_page_config(page_title="Annotateur de Visages", layout="wide")


# Simulation de l'import ou définition de detect_faces
# def detect_faces(image, output_dir=None):
#     img_array = np.array(image.convert("RGB"))
#     gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

#     # Correction de l'erreur cv2.data : on cherche le chemin du fichier xml de manière plus sûre
#     cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     if not os.path.exists(cascade_path):
#         # Fallback au cas où cv2.data n'est pas accessible
#         cascade_path = cv2.path.join(
#             cv2.__path__[0], "data", "haarcascade_frontalface_default.xml"
#         )

#     face_cascade = cv2.CascadeClassifier(cascade_path)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     face_data = []
#     for x, y, w, h in faces:
#         face_img = img_array[y : y + h, x : x + w]
#         pil_face = Image.fromarray(face_img)
#         # On force une taille de vignette compacte pour éviter le scroll excessif
#         pil_face.thumbnail((120, 120))
#         face_data.append({"image": pil_face, "coords": (x, y, w, h)})
#     return face_data


# Récupération de la configuration
output_dir = dotenv.get_key(".env", "OUTPUT_IMAGES_DIR")


# --- Initialisation de l'historique global ---
if "history_images" not in st.session_state:
    st.session_state.history_images = []
if "history_faces" not in st.session_state:
    st.session_state.history_faces = []

# --- Gestion de l'état de la session actuelle ---
if "step" not in st.session_state:
    st.session_state.step = "upload"
if "faces" not in st.session_state:
    st.session_state.faces = []
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "annotations" not in st.session_state:
    st.session_state.annotations = {}


def reset_app():
    # Sauvegarde dans l'historique avant de reset
    if st.session_state.original_image:
        st.session_state.history_images.append(st.session_state.original_image)

    for i, name in st.session_state.annotations.items():
        if name.strip():
            st.session_state.history_faces.append(
                {
                    "name": name,
                    "image": st.session_state.faces[i][0],
                    "date": time.strftime("%H:%M:%S"),
                }
            )

    # Nettoyage
    # Correction de l'erreur : vérification du type de la clé pour éviter l'erreur sur int
    keys = list(st.session_state.keys())
    for key in keys:
        if isinstance(key, str) and key.startswith("face_"):
            del st.session_state[key]

    st.session_state.step = "upload"
    st.session_state.faces = []
    st.session_state.original_image = None
    st.session_state.annotations = {}
    st.rerun()


# --- Interface Utilisateur ---
st.title("👤 Annotateur de Visages")

# Navigation principale par onglets si on n'est pas en cours d'annotation
if st.session_state.step == "upload":
    tab_up, tab_gal, tab_hist = st.tabs(
        ["📤 Nouvel Upload", "🗂️ Galerie Annotée", "🖼️ Historique Uploads"]
    )

    with tab_up:
        st.subheader("Téléchargez une photo pour commencer")
        uploaded_file = st.file_uploader("Choisir une image...", type=["jpg"])

        if uploaded_file:
            image_bytes = uploaded_file.read()
            image = Image.open(uploaded_file)
            st.session_state.original_image = image

            print("file info: ", image.info)
            with st.spinner("Détection des visages en cours..."):
                if output_dir:
                    detected = detect_faces(image_bytes, output_dir)
                    #     else detect_faces(image)
                    print("detected: ", type(detected))

                    st.session_state.faces = detected["results"]
                    st.session_state.step = "annotate"
                    st.rerun()

    with tab_gal:
        st.subheader("Rechercher un visage")
        search_query = st.text_input(
            "Filtrer par nom...", placeholder="Tapez un nom pour filtrer la galerie"
        )

        filtered_faces = [
            f
            for f in st.session_state.history_faces
            if search_query.lower() in f["name"].lower()
        ]

        if not st.session_state.history_faces:
            st.info(
                "La galerie est vide. Annotez des visages pour les voir apparaître ici."
            )
        elif not filtered_faces:
            st.warning(f"Aucun résultat pour '{search_query}'")
        else:
            g_cols = st.columns(10)
            for idx, face in enumerate(filtered_faces):
                with g_cols[idx % 10]:
                    st.image(face["image"], channels="BGR", width="content")
                    st.caption(f"**{face['name']}**")

    with tab_hist:
        st.subheader("Images précédemment traitées")
        if not st.session_state.history_images:
            st.info("Aucun historique d'upload.")
        else:
            h_cols = st.columns(5)
            for idx, img in enumerate(reversed(st.session_state.history_images)):
                with h_cols[idx % 5]:
                    st.image(
                        img,
                        use_container_width=True,
                        caption=f"Image #{len(st.session_state.history_images) - idx}",
                    )

# ÉTAPE 2 : Annotation (Interface dédiée)
elif st.session_state.step == "annotate":
    st.subheader("Mode Annotation")
    tab_ann, tab_pre = st.tabs(["📝 Annotation en cours", "🔍 Aperçu Sélection"])

    with tab_ann:
        st.caption("Utilisez les champs ci-dessous pour nommer les visages détectés.")
        # Grille de 10 colonnes pour des vignettes compactes
        cols = st.columns(2)

        for i, (face_info, _) in enumerate(st.session_state.faces):
            with cols[i % 2]:
                st.image(face_info, channels="BGR", width="content")
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

    with tab_pre:
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
                            st.session_state.faces[i][0],
                            channels="BGR",
                            use_container_width=True,
                        )
                        st.caption(name)
                    idx += 1

    st.markdown("---")
    c1, c2, _ = st.columns([2, 1, 4])
    with c1:
        can_val = any(v.strip() for v in st.session_state.annotations.values())
        if st.button(
            "✅ Valider et Sauvegarder", disabled=not can_val, use_container_width=True
        ):
            st.success("Enregistrement dans la galerie...")
            time.sleep(1)
            reset_app()
    with c2:
        if st.button("❌ Annuler", use_container_width=True):
            st.session_state.step = "upload"
            st.rerun()

# Sidebar
with st.sidebar:
    st.header("Session actuelle")
    if st.session_state.original_image:
        st.image(st.session_state.original_image, use_container_width=True)
        st.write(f"Détections : {len(st.session_state.faces)}")
    else:
        st.write("Prêt pour un nouvel upload.")
