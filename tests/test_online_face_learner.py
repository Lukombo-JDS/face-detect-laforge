from __future__ import annotations

import unittest

import numpy as np

from app.ml.online_face_learner import OnlineFaceLearner


class OnlineFaceLearnerTests(unittest.TestCase):
    def test_online_learner_returns_none_for_unknown_label(self) -> None:
        learner = OnlineFaceLearner()
        score = learner.score(np.array([1.0, 0.0], dtype=np.float32), "Alice")
        self.assertIsNone(score)

    def test_online_learner_increases_score_for_learned_label(self) -> None:
        learner = OnlineFaceLearner()
        embedding = np.array([1.0, 0.0], dtype=np.float32)
        learner.learn(embedding, "Alice")
        score = learner.score(embedding, "Alice")
        self.assertIsNotNone(score)
        assert score is not None
        self.assertGreater(score, 0.9)
