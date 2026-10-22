"""Check the independently packaged smoothed-mean comparator."""

import tempfile
from pathlib import Path
import unittest

from smoothed_mean.model import predict
from tools.build_smoothed_zip import build_zip
from check_submission_zip import validate_submission_zip


class SmoothedMeanTests(unittest.TestCase):
    def test_matching_labels_and_other_groups(self):
        subject = {"normalized_name": "one"}
        item = {"benchmark_id": "anonymous", "item_content": "target"}
        input_value = [subject, item]
        observed = [[subject, {"benchmark_id": "anonymous", "item_content": "other"}], 1]
        unrelated = [[subject, {"benchmark_id": "another"}], 0]
        self.assertEqual(predict(input_value, []), 0.5)
        self.assertAlmostEqual(predict(input_value, [observed, unrelated]), 2 / 3)
        self.assertEqual(predict(input_value, [unrelated]), 0.5)

    def test_zip_passes_local_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = build_zip(Path(temporary) / "smoothed.zip")
            validate_submission_zip(archive)


if __name__ == "__main__":
    unittest.main()
