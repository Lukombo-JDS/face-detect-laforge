from __future__ import annotations

from app.config import SETTINGS


def normalize_label(value: str) -> str:
    normalized = value.strip()
    return normalized or SETTINGS.unknown_label


def should_rebuild_index(new_faces_since_rebuild: int, threshold: int) -> bool:
    return new_faces_since_rebuild >= threshold
