from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LabelPrototype:
    center: np.ndarray
    count: int = 1


@dataclass
class OnlineFaceLearner:
    """Learner léger basé sur des prototypes moyens par label."""

    prototypes: dict[str, LabelPrototype] = field(default_factory=dict)

    def learn(self, embedding: np.ndarray, label: str) -> None:
        normalized = self._normalize(embedding)
        current = self.prototypes.get(label)
        if current is None:
            self.prototypes[label] = LabelPrototype(center=normalized, count=1)
            return
        updated_count = current.count + 1
        new_center = ((current.center * current.count) + normalized) / updated_count
        self.prototypes[label] = LabelPrototype(center=self._normalize(new_center), count=updated_count)

    def score(self, embedding: np.ndarray, label: str) -> float | None:
        prototype = self.prototypes.get(label)
        if prototype is None:
            return None
        query = self._normalize(embedding)
        score = float(np.dot(query, prototype.center))
        return max(0.0, min(1.0, (score + 1.0) / 2.0))

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        casted = vector.astype(np.float32)
        norm = float(np.linalg.norm(casted))
        if norm == 0.0:
            return casted
        return casted / norm
