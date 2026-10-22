"""Check six-budget replay with repeated trials kept on the same item side."""

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from mlp import model
from mlp.empirical_bayes import fit_lookup
from mlp.evaluate import BUDGETS, _score_group, evaluate


class MLPEvaluationTests(unittest.TestCase):
    def test_empirical_bayes_lookup_uses_only_training_rates(self):
        cache = {"source_revision": "synthetic", "by_benchmark": {
            "train_a": {"pair_rates": [0.2], "pair_response_counts": [100]},
            "train_b": {"pair_rates": [0.8], "pair_response_counts": [100]},
            "heldout": {"pair_rates": [1.0], "pair_response_counts": [100]},
        }}
        table = np.array(fit_lookup(cache, ["train_a", "train_b"])["posterior_lookup"])
        self.assertAlmostEqual(table[0, 0], 0.5, places=6)
        self.assertLess(table[1, 0], 0.5)
        self.assertGreater(table[1, 1], 0.5)
        self.assertTrue(np.isfinite(table).all())

    def test_batch_forward_matches_single_forward(self):
        rng = np.random.default_rng(7)
        weights = (
            rng.normal(0, 0.01, (model.FEATURE_DIM, 128)).astype(np.float32),
            np.zeros(128, np.float32),
            rng.normal(0, 0.01, (128, 64)).astype(np.float32),
            np.zeros(64, np.float32),
            rng.normal(0, 0.01, (64, 1)).astype(np.float32),
            np.zeros(1, np.float32),
        )
        features = rng.normal(size=(3, model.FEATURE_DIM)).astype(np.float32)
        batch = model.forward_batch(features, weights)
        for i in range(3):
            self.assertAlmostEqual(float(batch[i]), model.forward(features[i], weights), places=6)

    def test_group_replay_scores_every_evaluation_trial(self):
        group = pd.DataFrame([
            {"item_id": str(i), "response": outcome, "interactors": ""}
            for i in range(80) for outcome in (0, 1)
        ])
        items = {str(i): {"content": f"Question {i}", "item_features": ""} for i in range(80)}
        weights = (
            np.zeros((model.FEATURE_DIM, 128), np.float32), np.zeros(128, np.float32),
            np.zeros((128, 64), np.float32), np.zeros(64, np.float32),
            np.zeros((64, 1), np.float32), np.zeros(1, np.float32),
        )
        scores, n_responses = _score_group(
            group, {"normalized_name": "subject"}, items, "synthetic", "s", 0,
            40, {"neutral": weights}, (8.0,), mixes=(0.0, 1.0)
        )
        self.assertEqual(n_responses, 80)
        self.assertEqual(len(scores["neutral:residual_8"]), len(BUDGETS))
        self.assertEqual(scores["constant_0.5"], [0.25] * len(BUDGETS))
        self.assertEqual(scores["neutral:prior"], [0.25] * len(BUDGETS))
        self.assertAlmostEqual(scores["empirical_mean"][1], 0.5)
        self.assertAlmostEqual(scores["neutral:residual_8"][1], 0.25 + (0.5 / 9) ** 2)
        for mixed, smoothed in zip(scores["neutral:mix_0:residual_8"],
                                   scores["smoothed_mean_8"]):
            self.assertAlmostEqual(mixed, smoothed, places=6)
        eb_scores, _ = _score_group(
            group, {"normalized_name": "subject"}, items, "synthetic", "s", 0,
            40, {}, (2.0,), eb_table=np.full((32, 32), 0.3)
        )
        self.assertTrue(all(abs(value - 0.29) < 1e-6
                            for value in eb_scores["empirical_bayes"]))

    def test_pair_weighting_with_unequal_response_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = "synthetic/formatted_tables"
            tables = root / directory
            tables.mkdir(parents=True)
            pd.DataFrame([{"subject_id": identifier,
                           **{field: "" for field in model._SUBJECT_FIELDS}}
                          for identifier in ("a", "b")]).to_parquet(tables / "subjects.parquet")
            pd.DataFrame({"item_id": [str(i) for i in range(80)],
                          "content": [f"Question {i}" for i in range(80)],
                          "item_features": [""] * 80}).to_parquet(tables / "items.parquet")
            rows = []
            for item in range(80):
                rows.append({"subject_id": "a", "item_id": str(item),
                             "interactors": "", "response": 1})
                rows.extend({"subject_id": "b", "item_id": str(item),
                             "interactors": "", "response": trial % 2}
                            for trial in range(10))
            pd.DataFrame(rows).to_parquet(tables / "response.parquet")
            (root / "corpus_manifest.json").write_text(json.dumps({
                "revision": "synthetic", "eligible_dirs": [directory],
            }), encoding="utf-8")
            split_file = root / "split.json"
            split_file.write_text(json.dumps({
                "source_revision": "synthetic", "development_benchmarks": [directory],
            }), encoding="utf-8")
            result = evaluate(root, split_file, "development", {}, seeds=1,
                              max_eval_items=64, alphas=(2.0,))
            self.assertEqual(result["pairs_per_seed"], [2])
            self.assertEqual(result["evaluation_responses_per_seed"], [440])
            self.assertAlmostEqual(result["results"]["empirical_mean"]["brier_mean"][1], 0.25)
            # A fitted artifact may not be evaluated on a benchmark it used
            # for training, even when its NPZ file otherwise exists.
            leaked = root / "leaked.npz"
            leaked.write_bytes(b"not needed: metadata rejects first")
            leaked.with_suffix(".json").write_text(json.dumps({
                "train_benchmarks": [directory], "validation_benchmarks": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fitted on a development benchmark"):
                evaluate(root, split_file, "development", {"leaked": leaked}, seeds=1)
            eb_file = root / "leaked_eb.json"
            eb_file.write_text(json.dumps({
                "source_revision": "synthetic", "train_benchmarks": [directory],
                "posterior_lookup": [[0.5] * 32 for _ in range(32)],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "has seen an evaluated benchmark"):
                evaluate(root, split_file, "development", {}, seeds=1,
                         eb_table_file=eb_file)


if __name__ == "__main__":
    unittest.main()
