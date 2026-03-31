import os

import cv2
import numpy as np
import streamlit as st

# TODO: Ajouter un component de matching


def load_image_from_bytes(image_bytes):
    print("type:", type(image_bytes))
    print("len:", len(image_bytes) if image_bytes is not None else None)
    print("header:", image_bytes[:12] if image_bytes else None)

    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    print("nparr shape:", nparr.shape)
    print("nparr head:", nparr[:12])

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("❌ OpenCV n'arrive pas à décoder l'image")

    return img


def detect(img):
    # Conversion en gris (plus rapide et efficace)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Chargement du modèle
    face_cascade = cv2.CascadeClassifier("data/face_model.xml")

    if face_cascade.empty():
        raise RuntimeError("❌ Erreur chargement du modèle Haar Cascade")

    # Détection (paramètres améliorés)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # plus précis que 1.3
        minNeighbors=5,  # filtre les faux positifs
        minSize=(30, 30),  # ignore petits faux visages
    )

    return faces


def extract_faces(img, faces):
    face_images = []

    for x, y, w, h in faces:
        # Sécurisation des coordonnées
        x, y = max(0, x), max(0, y)

        face = img[y : y + h, x : x + w]

        if face.size == 0:
            continue

        # Normalisation taille (important pour ML ensuite)
        face = cv2.resize(face, (160, 160))

        face_images.append((face, (x, y, w, h)))

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return {
        "faces": face_images,
        "image": img,
    }


def save_bytes_as_jpeg(face_images, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    # f_img = face_images["faces"]
    for i, (face, _) in enumerate(face_images["faces"]):
        if i == 5:
            break
        filename = f"{output_dir}/faces/faces_{i}.jpg"

        cv2.imwrite(filename, face, [cv2.IMWRITE_JPEG_QUALITY, 95])


def detect_faces(image_uploaded, output_img_dir: str):
    img = load_image_from_bytes(image_uploaded)
    print("img loaded: ", img)
    results = {}

    faces_detected = detect(img)

    results = extract_faces(img, faces_detected)
    save_bytes_as_jpeg(results, output_img_dir)

    filename = f"{output_img_dir}/img.jpg"

    cv2.imwrite(filename, results["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {"results": results["faces"], "img": results["image"]}


def path_to_images(images_dir, images: list) -> list[str]:
    path = []
    for img in images:
        path.append(images_dir + "/" + img)

    return path


def image_uploded(output_img_dir: str):

    uploaded_file = st.file_uploader("Choose an image", type="jpg")
    if uploaded_file is not None:
        # To read file as bytes:

        detect_faces(uploaded_file.getvalue(), output_img_dir)
        images = os.listdir(output_img_dir)
        faces = os.listdir(output_img_dir + "/faces")
        print("path images: ", images.remove("faces"))
        print("path faces: ", faces)

        st.image(path_to_images(output_img_dir + "/faces", faces))
        st.image(path_to_images(output_img_dir, images))
