from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.logging import get_logger
from app.ml.embedder import FaceEmbedder
from app.storage.milvus_store import MilvusFaceStore

LOGGER = get_logger(__name__)


def run(folder: str, label: str, source_prefix: str) -> None:
    store = MilvusFaceStore()
    store.connect()
    embedder = FaceEmbedder()

    base = Path(folder)
    images = [p for p in base.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            LOGGER.warning("skip_invalid_image path=%s", path)
            continue
        vector = embedder.embed_bgr_face(img)
        store.add_face(label, vector, source_image=f"{source_prefix}:{path.name}")

    store.ensure_index()
    LOGGER.info("rebuild_done inserted=%s", len(images))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild/add embeddings in Milvus")
    parser.add_argument("--folder", default="data/images/faces")
    parser.add_argument("--label", default="__unknown__")
    parser.add_argument("--source-prefix", default="batch")
    args = parser.parse_args()
    run(args.folder, args.label, args.source_prefix)
