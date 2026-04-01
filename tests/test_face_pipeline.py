from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from app.vision.face_detection import DetectedFace
from app.config import SETTINGS
from app.services.face_pipeline import FacePipelineService
from app.services.annotation_logic import normalize_label, should_rebuild_index


class FacePipelineTests(unittest.TestCase):
    def test_normalize_label_unknown(self):
        self.assertEqual(normalize_label("  "), SETTINGS.unknown_label)

    def test_normalize_label_known(self):
        self.assertEqual(normalize_label("Alice"), "Alice")

    def test_rebuild_threshold_trigger(self):
        self.assertTrue(should_rebuild_index(SETTINGS.rebuild_threshold, SETTINGS.rebuild_threshold))
        self.assertFalse(should_rebuild_index(1, SETTINGS.rebuild_threshold))

    def test_init_does_not_connect_milvus(self):
        store = Mock()
        FacePipelineService(store=store, detector=Mock(), embedder=Mock())
        store.connect.assert_not_called()

    def test_process_uploaded_image_degrades_when_milvus_unavailable(self):
        detector = Mock()
        detector.load_image_from_bytes.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        detector.detect.return_value = [
            DetectedFace(image=np.zeros((4, 4, 3), dtype=np.uint8), bbox=(0, 0, 4, 4))
        ]
        embedder = Mock()
        embedder.embed_bgr_face.return_value = np.array([1.0, 2.0], dtype=np.float32)

        store = Mock()
        store.connect.side_effect = RuntimeError("milvus down")

        service = FacePipelineService(store=store, detector=detector, embedder=embedder)
        result = service.process_uploaded_image(b"fake-image")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].suggestions, [])
        store.search.assert_not_called()
        self.assertFalse(service.milvus_available)
        self.assertIn("milvus down", service.milvus_error or "")

    def test_save_annotations_skips_when_milvus_unavailable(self):
        store = Mock()
        store.connect.side_effect = RuntimeError("milvus down")
        service = FacePipelineService(store=store, detector=Mock(), embedder=Mock())

        processed = [
            Mock(
                embedding=np.array([0.1, 0.2], dtype=np.float32),
                face_image=np.zeros((4, 4, 3), dtype=np.uint8),
                bbox=(0, 0, 4, 4),
            )
        ]
        service.save_annotations(processed, {0: "Alice"}, "upload.jpg")

        store.add_face.assert_not_called()
        store.ensure_index.assert_not_called()

    def test_save_annotations_ignores_deleted_faces(self):
        store = Mock()
        service = FacePipelineService(store=store, detector=Mock(), embedder=Mock())

        processed = [
            Mock(embedding=np.array([0.1, 0.2], dtype=np.float32)),
            Mock(embedding=np.array([0.3, 0.4], dtype=np.float32)),
        ]

        service.save_annotations(
            processed_faces=processed,
            annotations={0: "Alice", 1: "Bob"},
            source_image="upload.jpg",
            deleted_indices={1},
        )

        store.add_face.assert_called_once_with("Alice", processed[0].embedding, source_image="upload.jpg")


if __name__ == "__main__":
    unittest.main()
