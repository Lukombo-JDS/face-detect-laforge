from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    output_images_dir: str = os.getenv("OUTPUT_IMAGES_DIR", "data/images")
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "face_embeddings")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "2048"))
    unknown_label: str = os.getenv("UNKNOWN_LABEL", "__unknown__")
    rebuild_threshold: int = int(os.getenv("REBUILD_THRESHOLD", "20"))


SETTINGS = Settings()
