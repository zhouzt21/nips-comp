"""Exercise the offline data-preparation resume path without network access."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from tools import prepare_data


class PrepareDataTests(unittest.TestCase):
    def test_resumes_same_revision_and_skips_traces(self):
        revision = "a" * 40
        directory = "sample/formatted_tables"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "sources"
            sources.mkdir()
            tables = {
                "benchmarks": pd.DataFrame([{"response_type": "binary", "granularity": "item"}]),
                "subjects": pd.DataFrame([{"subject_id": "s"}]),
                "items": pd.DataFrame([{"item_id": "i"}]),
                "response": pd.DataFrame([{"subject_id": "s", "item_id": "i", "response": 1}]),
            }
            for name, frame in tables.items():
                frame.to_parquet(sources / f"{name}.parquet")
            siblings = [SimpleNamespace(rfilename=f"{directory}/{name}.parquet")
                        for name in (*tables, "traces")]
            info = SimpleNamespace(sha=revision, siblings=siblings)
            api = SimpleNamespace(repo_info=lambda *args, **kwargs: info)
            requested_revisions = []
            def repo_info(*args, **kwargs):
                requested_revisions.append(kwargs["revision"])
                return info
            api.repo_info = repo_info
            failed = False
            def fetch(filename, source_revision):
                nonlocal failed
                self.assertEqual(source_revision, revision)
                name = Path(filename).stem
                if name == "response" and not failed:
                    failed = True
                    raise RuntimeError("synthetic interruption")
                return str(sources / f"{name}.parquet")
            out = root / "payload"
            args = ["prepare_data.py", "--out", str(out), "--no-embeddings", "--no-traces"]
            with patch.object(prepare_data, "HfApi", return_value=api), \
                 patch.object(prepare_data, "_fetch", side_effect=fetch), \
                 patch("sys.argv", args):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    prepare_data.main()
                self.assertTrue((out / ".prepare_state.json").is_file())
                with patch("sys.argv", args + ["--resume"]):
                    prepare_data.main()
            self.assertEqual(requested_revisions, ["main", revision])
            self.assertFalse((out / ".prepare_state.json").exists())
            self.assertTrue((out / "data" / directory / "response.parquet").is_file())
            self.assertFalse((out / "data" / directory / "traces.parquet").exists())
            manifest = json.loads((out / "data" / "corpus_manifest.json").read_text())
            self.assertEqual(manifest["revision"], revision)


if __name__ == "__main__":
    unittest.main()
