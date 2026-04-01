from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    """Garantit l'import de `app` même si Streamlit change le dossier courant."""
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()


def run() -> None:
    from app.ui.streamlit_app import run as streamlit_run

    streamlit_run()


if __name__ == "__main__":
    run()
