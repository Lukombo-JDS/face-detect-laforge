from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DetectedFace:
    image: np.ndarray
    bbox: tuple[int, int, int, int]


class FaceDetector:
    def __init__(self, cascade_path: str = "data/face_model.xml") -> None:
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Impossible de charger le modèle Haar Cascade: {cascade_path}")

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image invalide: OpenCV n'a pas pu décoder le fichier")
        return img

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        detected: list[DetectedFace] = []
        for x, y, w, h in faces:
            x, y = max(0, x), max(0, y)
            face = image[y : y + h, x : x + w]
            if face.size == 0:
                continue
            detected.append(DetectedFace(image=cv2.resize(face, (160, 160)), bbox=(x, y, w, h)))

        LOGGER.info("faces_detected=%s", len(detected))
        return detected
