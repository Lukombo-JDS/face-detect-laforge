import numpy as np

def embed_images(img):

    app = #TODO: mettre le model d'embedding d'image
    app.prepare(ctx_id=0)

    faces = app.get(img)

    embedding = faces[0].embedding  # vecteur ~512 dimensions
    embedding = embedding / np.linalg.norm(embedding)


def insert_embedding(embedding, collection):
    data = [[embedding.tolist()]]
    collection.insert(data)
