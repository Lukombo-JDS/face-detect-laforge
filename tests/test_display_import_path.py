from __future__ import annotations

import sys
import unittest
from pathlib import Path

from view.display import _ensure_project_root_on_path


class DisplayImportPathTests(unittest.TestCase):
    def test_ensure_project_root_adds_repo_path(self):
        expected_root = str(Path(__file__).resolve().parents[1])
        previous_path = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path if p != expected_root]
            _ensure_project_root_on_path()
            self.assertEqual(sys.path[0], expected_root)
        finally:
            sys.path[:] = previous_path


if __name__ == "__main__":
    unittest.main()
