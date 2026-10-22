"""Interface and leakage checks for the offline MLP predictor."""

import unittest
from unittest.mock import patch

import numpy as np

from mlp import model


def neutral_weights():
    return (
        np.zeros((model.FEATURE_DIM, 128), dtype=np.float32),
        np.zeros(128, dtype=np.float32),
        np.zeros((128, 64), dtype=np.float32),
        np.zeros(64, dtype=np.float32),
        np.zeros((64, 1), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )


class MLPTests(unittest.TestCase):
    def setUp(self):
        self.subject = {"normalized_name": "subject-a", "provider": "provider-a"}
        self.item = {"benchmark_id": "anonymous-1", "item_content": "Solve 2 + 2"}
        self.target = [self.subject, self.item]

    def observed(self, label, *, benchmark="anonymous-1", subject=None):
        return [[self.subject if subject is None else subject,
                 {"benchmark_id": benchmark, "item_content": "Solve 3 + 3"}], label]

    def test_encode_is_deterministic_and_does_not_memorize_benchmark_id(self):
        first = model.encode(self.target)
        changed_id = model.encode([self.subject, dict(self.item, benchmark_id="secret-test-id")])
        changed_text = model.encode([self.subject, dict(self.item, item_content="Write a program")])
        self.assertEqual(first.shape, (model.FEATURE_DIM,))
        np.testing.assert_array_equal(first, changed_id)
        self.assertFalse(np.array_equal(first, changed_text))

    def test_forward_and_shrunk_group_correction(self):
        with patch.object(model, "_load_weights", return_value=neutral_weights()):
            self.assertEqual(model.predict(self.target, []), 0.5)
            labels = [self.observed(1), self.observed(0, benchmark="anonymous-2"),
                      self.observed(0, subject={"normalized_name": "other"})]
            self.assertAlmostEqual(model.predict(self.target, labels),
                                   0.5 + 0.5 / (1 + model.PRIOR_STRENGTH))
            self.assertEqual(model.predict(self.target, [self.observed(1), self.observed(0)]), 0.5)
            with self.assertRaisesRegex(ValueError, "benchmark_id"):
                model.predict([self.subject, {"item_content": "x"}], labels)

    def test_missing_trained_artifact_is_explicit(self):
        model._load_weights.cache_clear()
        if not model._WEIGHTS_PATH.exists():
            with self.assertRaisesRegex(FileNotFoundError, "weights.npz"):
                model.predict(self.target, [])


if __name__ == "__main__":
    unittest.main()
