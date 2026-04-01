from __future__ import annotations

import unittest

from app.config import SETTINGS
from app.services.annotation_logic import normalize_label, should_rebuild_index


class FacePipelineTests(unittest.TestCase):
    def test_normalize_label_unknown(self):
        self.assertEqual(normalize_label("  "), SETTINGS.unknown_label)

    def test_normalize_label_known(self):
        self.assertEqual(normalize_label("Alice"), "Alice")

    def test_rebuild_threshold_trigger(self):
        self.assertTrue(should_rebuild_index(SETTINGS.rebuild_threshold, SETTINGS.rebuild_threshold))
        self.assertFalse(should_rebuild_index(1, SETTINGS.rebuild_threshold))


if __name__ == "__main__":
    unittest.main()
