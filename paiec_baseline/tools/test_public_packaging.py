"""Verify that sharing a BLE archive cannot silently include local credentials."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile


class PublicPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copyfile(
            Path(__file__).resolve().parents[1] / "LICENSE",
            self.root / "LICENSE",
        )
        (self.root / "tools").mkdir()
        shutil.copyfile(
            Path(__file__).with_name("build_zip.sh"),
            self.root / "tools/build_zip.sh",
        )
        shutil.copyfile(
            Path(__file__).with_name("ble_packaging.py"),
            self.root / "tools/ble_packaging.py",
        )
        self.submission = self.root / "ble"
        self.submission.mkdir()
        for name in ("model.py", "labeling.py", "requirements.txt"):
            (self.submission / name).write_text("# synthetic fixture\n")
        for folder in ("src", "comp_pipeline"):
            directory = self.submission / folder
            directory.mkdir()
            (directory / "__init__.py").write_text("")
            (directory / ".env").write_text("do-not-distribute-local-settings")
            (directory / "trace.log").write_text("do-not-distribute-local-traces")
        self.example = {"api_keys": {"OPENAI_API_KEY": ""}, "max_steps": 5}
        self.private = {"api_keys": {"OPENAI_API_KEY": "synthetic-private-key"}}
        (self.submission / "submission_config.example.json").write_text(
            json.dumps(self.example)
        )
        (self.submission / "submission_config.json").write_text(
            json.dumps(self.private)
        )
        data = self.root / "payload/data"
        data.mkdir(parents=True)
        (data / "corpus_manifest.json").write_text(json.dumps({
            "corpus_scope": "paec-public-training",
            "db_repo": "aims-foundations/measurement-db",
            "revision": "a" * 40,
        }))

    def build(self, *args):
        # Git Bash cannot resolve native Windows temporary paths passed as
        # subprocess arguments. The shell wrapper only execs this Python file.
        command = ([sys.executable, str(self.root / "tools/ble_packaging.py")]
                   if os.name == "nt" else ["bash", str(self.root / "tools/build_zip.sh")])
        return subprocess.run(
            [*command, *args],
            capture_output=True, text=True,
        )

    def test_public_build_ignores_local_keys_and_generated_source_artifacts(self):
        result = self.build("--public", "bundle")
        self.assertEqual(result.returncode, 0, result.stderr)
        with ZipFile(self.root / "dist/ble_public.zip") as archive:
            self.assertEqual(archive.read("LICENSE"), (self.root / "LICENSE").read_bytes())
            self.assertEqual(
                json.loads(archive.read("submission_config.json")), self.example
            )
            for name in archive.namelist():
                self.assertNotIn(".env", name)
                self.assertFalse(name.endswith(".log"))
                self.assertNotIn(b"synthetic-private-key", archive.read(name))
        self.assertFalse((self.root / "dist/ble_submission.zip").exists())

    def test_private_build_requires_explicit_mode_and_keeps_separate_output(self):
        self.assertNotEqual(self.build("bundle").returncode, 0)
        result = self.build("--private", "bundle")
        self.assertEqual(result.returncode, 0, result.stderr)
        with ZipFile(self.root / "dist/ble_submission.zip") as archive:
            self.assertEqual(archive.read("LICENSE"), (self.root / "LICENSE").read_bytes())
            self.assertEqual(
                json.loads(archive.read("submission_config.json")), self.private
            )
        self.assertFalse((self.root / "dist/ble_public.zip").exists())

    def test_public_build_refuses_credentials_in_example_config(self):
        (self.submission / "submission_config.example.json").write_text(
            json.dumps(self.private)
        )
        result = self.build("--public", "bundle")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("synthetic-private-key", result.stdout + result.stderr)
        self.assertFalse((self.root / "dist/ble_public.zip").exists())

    def test_hf_public_build_and_missing_private_config(self):
        (self.submission / "submission_config.json").unlink()
        self.assertNotEqual(self.build("--private", "bundle").returncode, 0)
        result = self.build("--public", "hf", "example/public-payload")
        self.assertEqual(result.returncode, 0, result.stderr)
        with ZipFile(self.root / "dist/ble_public.zip") as archive:
            self.assertEqual(archive.read("LICENSE"), (self.root / "LICENSE").read_bytes())
            config = json.loads(archive.read("submission_config.json"))
            self.assertEqual(config["api_keys"], self.example["api_keys"])
            self.assertEqual(config["hf_data_repo"], "example/public-payload")
            self.assertEqual(
                archive.read("models.txt"), b"example/public-payload\n"
            )

    def test_only_manifest_listed_payload_files_are_packaged(self):
        data = self.root / "payload/data"
        (data / "unlisted.txt").write_text("do-not-distribute-unlisted-data")
        result = self.build("--public", "bundle")
        self.assertEqual(result.returncode, 0, result.stderr)
        with ZipFile(self.root / "dist/ble_public.zip") as archive:
            self.assertNotIn("data/unlisted.txt", archive.namelist())

    def test_payload_paths_cannot_escape_the_public_data_directory(self):
        path = self.root / "payload/data/corpus_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"] = ["../../ble/submission_config.json"]
        path.write_text(json.dumps(manifest))
        result = self.build("--public", "bundle")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "dist/ble_public.zip").exists())


class PublicSourceCheckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)

    def check(self, *args):
        return subprocess.run(
            [sys.executable, str(Path(__file__).with_name("check_public_source.py")), *args],
            cwd=self.root, capture_output=True, text=True,
        )

    def stage(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        subprocess.run(["git", "add", "--", name], cwd=self.root, check=True)
        return path

    def test_check_reads_staged_configuration_not_unstaged_replacement(self):
        name = "submission_config.example.json"
        path = self.stage(name, json.dumps({"api_keys": {"OPENAI_API_KEY": ""}}))
        path.write_text(json.dumps({"api_keys": {"OPENAI_API_KEY": "synthetic-private"}}))
        self.assertEqual(self.check().returncode, 0)
        subprocess.run(["git", "add", name], cwd=self.root, check=True)
        path.write_text(json.dumps({"api_keys": {"OPENAI_API_KEY": ""}}))
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("synthetic-private", result.stdout + result.stderr)

    def test_check_rejects_artifacts_regardless_of_contents(self):
        self.stage("dist/innocent.txt", "synthetic example")
        self.stage("submission_config.json", "{}")
        self.assertNotEqual(self.check().returncode, 0)

    def test_check_redacts_detected_credential(self):
        secret = "sk-" + "a" * 30
        self.stage("model.py", "KEY = " + repr(secret) + "\n")
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(secret, result.stdout + result.stderr)

    def test_research_records_require_explicit_option(self):
        self.stage("runs/R1/metrics.json", '{"alc": 0.25}')
        self.stage("paiec_baseline/runs/R1/train.log", "completed\n")
        self.stage("paiec_baseline/runs/R1/code/model.py", "# snapshot\n")
        self.assertNotEqual(self.check().returncode, 0)
        self.assertEqual(self.check("--research-records").returncode, 0)

    def test_research_option_still_rejects_data_and_private_paths(self):
        self.stage("paiec_baseline/runs/R1/weights.npz", "synthetic array")
        self.stage("paiec_baseline/runs/R1/predictions.csv", "y,p\n")
        self.stage("paiec_baseline/runs/R1/payload/labels.json", "{}")
        self.stage("paiec_baseline/runs/R1/submission_config.json", "{}")
        self.stage("paiec_baseline/runs/R1/.env", "synthetic settings")
        result = self.check("--research-records")
        self.assertNotEqual(result.returncode, 0)
        for name in ("weights.npz", "predictions.csv", "labels.json",
                     "submission_config.json", ".env"):
            self.assertIn(name, result.stderr)

    def test_research_option_scans_log_credentials(self):
        secret = "github_pat_" + "a" * 40
        self.stage("paiec_baseline/runs/R1/train.log", secret)
        result = self.check("--research-records")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("possible credential", result.stderr)
        self.assertNotIn(secret, result.stdout + result.stderr)

    def test_research_option_preserves_raw_markdown_search_export(self):
        self.stage("paiec_baseline/research_loop/literature/raw/search.json",
                   "## Query 1: historical query\n\nSearch results\n")
        self.assertNotEqual(self.check().returncode, 0)
        self.assertEqual(self.check("--research-records").returncode, 0)

    def test_research_option_does_not_allow_invalid_json_configuration(self):
        self.stage("paiec_baseline/runs/R1/config.json", "## Query invalid configuration")
        self.assertNotEqual(self.check("--research-records").returncode, 0)


if __name__ == "__main__":
    unittest.main()
