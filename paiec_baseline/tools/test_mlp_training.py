"""Verify public-table joins and export are compatible with MLP inference."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mlp import model
from mlp.train import _sample_scoreable_pairs, train


class MLPTrainingTests(unittest.TestCase):
    def test_scoreable_pair_sampler_is_balanced_and_deterministic(self):
        rows = pd.DataFrame([
            {"subject_id": f"s{subject}", "item_id": str(item),
             "response": item % 2, "interactors": ""}
            for subject in range(10) for item in range(100)
        ] + [
            {"subject_id": "short", "item_id": str(item),
             "response": item % 2, "interactors": ""}
            for item in range(79)
        ])
        first = _sample_scoreable_pairs(rows, 400, 2026)
        second = _sample_scoreable_pairs(rows, 400, 2026)
        self.assertTrue(first.equals(second))
        self.assertEqual(len(first), 400)
        self.assertNotIn("short", set(first["subject_id"]))
        self.assertEqual(set(first.groupby("subject_id").size()), {40})

    def test_fit_and_export_from_two_benchmark_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dirs = ["one/formatted_tables", "two/formatted_tables"]
            (root / "corpus_manifest.json").write_text(json.dumps({
                "revision": "synthetic", "eligible_dirs": dirs
            }), encoding="utf-8")
            for directory in dirs:
                path = root / directory
                path.mkdir(parents=True)
                pd.DataFrame([{
                    "subject_id": "s", "normalized_name": "model-a", "provider": "lab",
                    "release_date": "", "access_date": "", "harness": "",
                    "harness_version": "", "reasoning_effort": "",
                    "subject_features_extra": "",
                }]).to_parquet(path / "subjects.parquet", index=False)
                pd.DataFrame([{
                    "item_id": str(i), "content": f"Question {i}", "item_features": ""
                } for i in range(8)]).to_parquet(path / "items.parquet", index=False)
                pd.DataFrame([{
                    "subject_id": "s", "item_id": str(i), "response": i % 2,
                    "interactors": "",
                } for i in range(8)]).to_parquet(path / "response.parquet", index=False)
            out = root / "weights.npz"
            with patch("torch.cuda.is_available", return_value=False):
                result = train(root, out, epochs=1, batch_size=8, max_rows=8)
            self.assertEqual(result["train_rows"], 8)
            self.assertEqual(result["validation_rows"], 8)
            self.assertTrue(out.is_file())
            with np.load(out, allow_pickle=False) as artifact:
                weights = tuple(artifact[key] for key in ("W1", "b1", "W2", "b2", "W3", "b3"))
            probability = model.forward(model.encode([
                {"normalized_name": "model-a"}, {"item_content": "Question 4"}
            ]), weights)
            self.assertTrue(0 <= probability <= 1)
            split_file = root / "split.json"
            split_file.write_text(json.dumps({
                "source_revision": "synthetic", "train_benchmarks": [dirs[0]],
                "development_benchmarks": [dirs[1]], "test_benchmarks": [],
            }), encoding="utf-8")
            with patch("torch.cuda.is_available", return_value=False):
                refit = train(root, root / "refit.npz", split_file=split_file,
                              fixed_epochs=1, batch_size=8, max_rows=8)
            self.assertEqual(refit["train_rows"], 16)
            self.assertEqual(refit["validation_rows"], 0)
            self.assertEqual(refit["early_stopping_pool"], "none_fixed_refit")
            with patch("torch.cuda.is_available", return_value=False):
                controlled = train(root, root / "controlled.npz", epochs=1,
                                   batch_size=2, max_rows=8,
                                   max_validation_rows=4, rows_per_epoch=4)
            self.assertEqual(controlled["train_rows"], 8)
            self.assertEqual(controlled["validation_rows"], 4)
            self.assertEqual(controlled["unique_training_rows_exposed"], 4)
            self.assertEqual(controlled["optimizer_steps_executed"], 2)
            self.assertTrue(0 <= controlled["selected_model_training_pool_pair_brier"] <= 1)

    def test_internal_early_stopping_keeps_development_and_test_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dirs = [f"train{i}/formatted_tables" for i in range(5)]
            development = "development/formatted_tables"
            test = "test/formatted_tables"
            all_dirs = [*train_dirs, development, test]
            (root / "corpus_manifest.json").write_text(json.dumps({
                "revision": "synthetic", "eligible_dirs": all_dirs,
            }), encoding="utf-8")
            for directory in all_dirs:
                path = root / directory
                path.mkdir(parents=True)
                pd.DataFrame([{"subject_id": "s", **{field: "" for field in model._SUBJECT_FIELDS}}])\
                    .to_parquet(path / "subjects.parquet", index=False)
                pd.DataFrame([{"item_id": str(i), "content": f"Question {i}",
                               "item_features": ""} for i in range(8)])\
                    .to_parquet(path / "items.parquet", index=False)
                pd.DataFrame([{"subject_id": "s", "item_id": str(i),
                               "response": i % 2, "interactors": ""} for i in range(8)])\
                    .to_parquet(path / "response.parquet", index=False)
            split_file = root / "split.json"
            split_file.write_text(json.dumps({
                "source_revision": "synthetic", "train_benchmarks": train_dirs,
                "development_benchmarks": [development], "test_benchmarks": [test],
                "scoreable_pair_counts": {directory: 1 for directory in train_dirs},
            }), encoding="utf-8")
            with patch("torch.cuda.is_available", return_value=False):
                result = train(root, root / "internal.npz", split_file=split_file,
                               internal_validation=True, epochs=1,
                               batch_size=16, max_rows=8)
            self.assertEqual(result["early_stopping_pool"], "training")
            self.assertEqual(result["development_benchmarks_untouched"], [development])
            self.assertTrue(set(result["train_benchmarks"] + result["validation_benchmarks"])
                            .isdisjoint({development, test}))


if __name__ == "__main__":
    unittest.main()
