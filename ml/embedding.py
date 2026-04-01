"""Legacy wrapper for embedding helpers."""

from app.ml.embedder import FaceEmbedder


def embed_images(img_bgr):
    return FaceEmbedder().embed_bgr_face(img_bgr)
