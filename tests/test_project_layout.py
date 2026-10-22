"""Exercise relocated report inputs and script outputs without running experiments."""

import ast
import importlib
from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT / "paiec_baseline"


class ProjectLayoutTests(unittest.TestCase):
    def test_reports_and_research_state_are_available_at_project_root(self):
        for path in ("reports/research_loop_v3_findings.md", "research_loop/HIGH_LEVEL_STATE.md"):
            with self.subTest(path=path):
                self.assertTrue((PROJECT / path).is_file(), path)

    def test_protocol_snapshot_inputs_resolve_from_experiment_directory(self):
        checked = 0
        for script in (BASELINE / "mlp").rglob("*.py"):
            tree = ast.parse(script.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, (ast.List, ast.Tuple)):
                    continue
                if not any(isinstance(t, ast.Name) and t.id in {"sources", "source", "paths"}
                           for t in node.targets):
                    continue
                for value in node.value.elts:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str) and "_protocol.md" in value.value:
                        checked += 1
                        with self.subTest(script=script.name, path=value.value):
                            path = (BASELINE / value.value).resolve()
                            self.assertTrue(path.is_file(), str(path))
                            self.assertTrue(path.is_relative_to(PROJECT / "research_loop"))
        self.assertGreater(checked, 30)

    def test_report_writers_target_project_reports(self):
        checked = 0
        for script in (BASELINE / "mlp").rglob("*.py"):
            for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "Path" and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and "reports/" in node.args[0].value):
                    checked += 1
                    with self.subTest(script=script.name):
                        self.assertTrue((BASELINE / node.args[0].value).resolve()
                                        .is_relative_to(PROJECT / "reports"))
        self.assertGreaterEqual(checked, 4)

    def test_categorized_modules_can_be_imported(self):
        modules = (
            "mlp.audit.audit_evaluation_items", "mlp.run.run_metadata_round",
            "mlp.verify.verify_metadata_penalty_round", "mlp.reporting.summarize_candidate",
            "mlp.data.make_cv_splits",
        )
        for name in modules:
            with self.subTest(module=name):
                self.assertTrue((BASELINE / (name.replace(".", "/") + ".py")).is_file())
                self.assertEqual(importlib.import_module(name).__name__, name)

    def test_archived_source_inputs_exist_after_module_moves(self):
        checked = 0
        for script in (BASELINE / "mlp").rglob("*.py"):
            for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
                if (not isinstance(node, ast.Assign) or not isinstance(node.value, (ast.List, ast.Tuple))
                        or not any(isinstance(t, ast.Name) and t.id in {"sources", "source", "paths"}
                                   for t in node.targets)):
                    continue
                for value in node.value.elts:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.endswith(".py"):
                        checked += 1
                        with self.subTest(script=script.name, source=value.value):
                            self.assertTrue((BASELINE / value.value).is_file())
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()
