import os

import dotenv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetModel

# TODO: Implementer le model d'embedding pour Image

IMAGES_FOLDER = dotenv.get_key("./.env", "OUTPUT_IMAGES_DIR")


def embed_images(img):

    # 1. Charger le processeur et le modèle de base (SANS la tête de classification)
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetModel.from_pretrained("microsoft/resnet-50")

    # 2. Charger votre image de visage
    image = Image.open(img)

    # 3. Préparer l'image pour le modèle
    inputs = processor(image, return_tensors="pt")

    # 4. Extraire le vecteur
    with torch.no_grad():
        outputs = model(**inputs)

        # Le vecteur se trouve dans 'pooler_output' pour ResNet
        # C'est une représentation compressée de l'image de dimension [1, 2048]
        face_vector = outputs.pooler_output.flatten().numpy()

    print(f"Dimension du vecteur : {face_vector.shape}")
    # C'est ce 'face_vector' qu'on envoi dans Milvus

    return face_vector


def stacking_img_vectors():

    folder_path = "data/images/faces/faces_tagged"
    images_names = os.listdir(folder_path)
    path = []
    img_embeded = []

    for name in images_names:
        path.append(folder_path + "/" + name)

    for img in path:
        img_embeded.append(embed_images(img))

    # print("images vectorized: ")
    # print("size: ", np.size(img_embeded))
    # print("head: ", img_embeded[:1])

    return img_embeded


def insert_embedding(embedding, collection):

    data = [[embedding.tolist()]]
    collection.insert(data)
