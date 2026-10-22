"""Checks that research variants preserve their intended information boundary."""
import unittest
import json
import numpy as np
import torch
from mlp.model import encode, encode_mode, forward_batch, FEATURE_DIM
from mlp.structured_text import extract_text


class LoopModelsTests(unittest.TestCase):
    def test_structured_content_preserves_question_and_limits_boilerplate(self):
        value = json.dumps({"published_system_prompt": "boilerplate "*2000,
                            "location": "images/123abcaadeadbeef123456789.png",
                            "question": "which constructor had zero points?"})
        text = extract_text(value)
        self.assertIn("which constructor had zero points", text[:200])
        self.assertNotIn("123abcaadeadbeef123456789", text)
        self.assertLess(len(text.split()), 100)
        self.assertEqual(extract_text("plain question"), "plain question")

    def test_blocks_separate_subject_from_items_and_exclude_benchmark_id(self):
        subject = {"normalized_name": "a"}
        item = {"item_content": "question one", "benchmark_id": "x"}
        base = encode_mode([subject, item], "block")
        changed = encode_mode([subject, {**item, "item_content": "different question"}], "block")
        np.testing.assert_array_equal(base[:512], changed[:512])
        self.assertFalse(np.array_equal(base[512:1536], changed[512:1536]))
        np.testing.assert_array_equal(base, encode_mode([subject, {**item, "benchmark_id": "y"}], "block"))
        np.testing.assert_array_equal(encode([subject, item]), encode_mode([subject, item], "all"))

    def test_linear_numpy_matches_torch(self):
        rng = np.random.default_rng(34)
        x = rng.normal(size=(9, FEATURE_DIM)).astype(np.float32)
        w = rng.normal(0, .01, size=(FEATURE_DIM, 1)).astype(np.float32)
        b = np.array([.2], np.float32)
        reference = torch.sigmoid(torch.from_numpy(x) @ torch.from_numpy(w) + torch.from_numpy(b)).numpy().ravel()
        np.testing.assert_allclose(forward_batch(x, (w, b)), reference, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
