from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import SETTINGS
from app.logging import get_logger
from app.ml.embedder import FaceEmbedder
from app.services.annotation_logic import normalize_label, should_rebuild_index
from app.storage.milvus_store import MilvusFaceStore, SearchResult
from app.vision.face_detection import FaceDetector

LOGGER = get_logger(__name__)


@dataclass
class ProcessedFace:
    face_image: np.ndarray
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    suggestions: list[SearchResult]


class FacePipelineService:
    def __init__(
        self,
        store: MilvusFaceStore | None = None,
        detector: FaceDetector | None = None,
        embedder: FaceEmbedder | None = None,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()
        self.store = store or MilvusFaceStore()
        self.store.connect()
        self._new_faces_since_rebuild = 0

    def process_uploaded_image(self, image_bytes: bytes) -> list[ProcessedFace]:
        image = self.detector.load_image_from_bytes(image_bytes)
        detected = self.detector.detect(image)
        faces: list[ProcessedFace] = []
        for item in detected:
            embedding = self.embedder.embed_bgr_face(item.image)
            suggestions = self.store.search(embedding, limit=3)
            faces.append(
                ProcessedFace(
                    face_image=item.image,
                    bbox=item.bbox,
                    embedding=embedding,
                    suggestions=suggestions,
                )
            )
        return faces

    def save_annotations(
        self,
        processed_faces: list[ProcessedFace],
        annotations: dict[int, str],
        source_image: str,
    ) -> None:
        for idx, face in enumerate(processed_faces):
            label = normalize_label(annotations.get(idx, ""))
            self.store.add_face(label, face.embedding, source_image=source_image)
        self._new_faces_since_rebuild += len(processed_faces)
        self._rebuild_index_if_needed()

    def _rebuild_index_if_needed(self) -> None:
        if not should_rebuild_index(self._new_faces_since_rebuild, SETTINGS.rebuild_threshold):
            return
        LOGGER.info("index_rebuild_triggered new_faces=%s", self._new_faces_since_rebuild)
        self.store.ensure_index()
        self._new_faces_since_rebuild = 0
