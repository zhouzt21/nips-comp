"""Check staged source for private artifacts and common credential formats.

Run after selective git add and before committing. Only the Git index is read;
ignored local datasets and credentials are never opened. Review the staged diff
as well: pattern checks cannot establish that arbitrary example data are public.
"""

import argparse
import json
from pathlib import PurePosixPath
import re
import subprocess
import sys


PRIVATE_DIRECTORIES = {
    "dist", "payload", "runs", "blf_runs", "emb_cache", ".venv", "__pycache__",
    "paec_competition_submission",
}
PRIVATE_FILENAMES = {
    "submission_config.json", "models.txt", "excluded_benchmarks.txt",
    "ground_truth.csv", "stream_plan.json", "stream_state.json",
    "sample_manifest.json",
}
ARTIFACT_SUFFIXES = {
    ".zip", ".log", ".parquet", ".pyc", ".pkl", ".pickle",
    ".npz", ".pt", ".pth", ".safetensors", ".sqlite", ".sqlite3",
}
RESEARCH_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".txt", ".log"}
CREDENTIAL = re.compile(
    rb"(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}"
    rb"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    rb"|AIza[A-Za-z0-9_-]{30,}"
    rb"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-records", action="store_true",
        help="Allow reviewed text evidence in runs/ or paiec_baseline/runs/; still scan credentials",
    )
    args = parser.parse_args()
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    entries = subprocess.check_output(["git", "-C", root, "ls-files", "--stage", "-z"])
    problems = []
    count = 0
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, oid, stage = metadata.decode().split()
        path = PurePosixPath(raw_path.decode())
        count += 1
        in_research = (
            path.parts[:1] == ("runs",)
            or path.parts[:2] == ("paiec_baseline", "runs")
        )
        research_record = (
            args.research_records and in_research
            and path.suffix.lower() in RESEARCH_SUFFIXES
        )
        if (
            mode not in {"100644", "100755"} or stage != "0"
            or any(part in PRIVATE_DIRECTORIES and not (part == "runs" and research_record)
                   for part in path.parts)
            or path.name in PRIVATE_FILENAMES
            or (path.name.startswith(".env") and path.name != ".env.example")
            or (path.suffix.lower() in ARTIFACT_SUFFIXES
                and not (research_record and path.suffix.lower() == ".log"))
        ):
            problems.append((str(path), "private artifact or unsupported file type"))
            continue
        contents = subprocess.check_output(["git", "cat-file", "blob", oid])
        if CREDENTIAL.search(contents):
            problems.append((str(path), "possible credential; value withheld"))
        if path.suffix == ".json":
            try:
                config = json.loads(contents)
            except (ValueError, UnicodeDecodeError):
                # Historical search exports sometimes used a JSON filename
                # for a Markdown transcript. Preserve raw evidence; this is
                # not a configuration exception and credentials were scanned.
                local_parts = path.parts[1:] if path.parts[:1] == ("paiec_baseline",) else path.parts
                if (args.research_records
                    and local_parts[:3] == ("research_loop", "literature", "raw")
                    and contents.removeprefix(b"\xef\xbb\xbf").startswith(b"## Query ")):
                    print(f"{path}: preserved Markdown retrieval transcript with historical .json suffix")
                    continue
                problems.append((str(path), "invalid JSON"))
                continue
            if isinstance(config, dict) and any(config.get("api_keys", {}).values()):
                problems.append((str(path), "nonempty API credentials"))
    for path, reason in problems:
        print(f"{path}: {reason}", file=sys.stderr)
    if problems:
        return 1
    print(f"Checked {count} staged source files: no prohibited artifacts or credentials found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
