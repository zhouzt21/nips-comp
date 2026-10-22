"""Check benchmark-grouped CV folds exclude the frozen held-out pools."""

import unittest

from mlp.data.make_cv_splits import make_folds


class CVSplitTests(unittest.TestCase):
    def test_each_scoreable_training_benchmark_is_held_once(self):
        training = [f"train{i}" for i in range(12)] + ["empty"]
        parent = {
            "source_revision": "synthetic",
            "scoreable_pair_counts": {**{f"train{i}": i + 1 for i in range(12)},
                                      "empty": 0},
            "train_benchmarks": training,
            "development_benchmarks": ["original-development"],
            "test_benchmarks": ["original-test"],
        }
        folds = make_folds(parent, seed=2027, folds=3, pair_cap=5)
        held = [directory for fold in folds for directory in fold["development_benchmarks"]]
        self.assertEqual(set(held), set(training) - {"empty"})
        self.assertEqual(len(held), len(set(held)))
        for fold in folds:
            self.assertIn("empty", fold["train_benchmarks"])
            self.assertEqual(set(fold["test_benchmarks"]),
                             {"original-development", "original-test"})
            self.assertFalse(set(fold["train_benchmarks"]) &
                             set(fold["development_benchmarks"]))


if __name__ == "__main__":
    unittest.main()
