"""Check the public-data proxy scorer on a small, controlled parquet payload."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from evaluate_mean import evaluate


class MeanEvaluationTests(unittest.TestCase):
    def test_budgets_and_pair_weighting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "synthetic" / "formatted_tables"
            directory.mkdir(parents=True)
            rows = []
            for item in range(80):
                # The small group is perfectly predictable after one label.
                rows.append({"subject_id": "a", "item_id": f"item-{item}",
                             "response": 1})
                # Every item in the large group has ten trials with equal
                # positives and negatives. A one-label mean has Brier 0.5.
                rows.extend({"subject_id": "b", "item_id": f"item-{item}",
                             "response": int(trial % 2 == 0)}
                            for trial in range(10))
            pd.DataFrame(rows).to_parquet(directory / "response.parquet", index=False)
            (root / "corpus_manifest.json").write_text(json.dumps({
                "revision": "synthetic", "eligible_dirs": ["synthetic/formatted_tables"]
            }), encoding="utf-8")
            result = evaluate(root, seeds=2)
            self.assertEqual(result["pairs_per_seed"], [2, 2])
            self.assertEqual(result["budgets"], [0, 1, 3, 7, 15, 31])
            self.assertEqual(result["brier_mean"][0], 0.25)
            self.assertEqual(result["evaluation_responses_per_seed"], [440, 440])
            # Equal pair weighting gives (0 + 0.5)/2, rather than weighting
            # the group with ten times as many response trials more heavily.
            self.assertAlmostEqual(result["brier_mean"][1], 0.25)
            self.assertTrue(0 <= result["brier_alc_mean"] <= 1)


if __name__ == "__main__":
    unittest.main()
