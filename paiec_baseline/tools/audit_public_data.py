"""Audit the prepared PAIEC public core tables without reading trace payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit(data_root: Path) -> dict:
    manifest = json.loads((data_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    expected = {"benchmarks.parquet", "subjects.parquet", "items.parquet", "response.parquet"}
    missing = [str(data_root / d / name) for d in manifest["eligible_dirs"] for name in expected
               if not (data_root / d / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} core tables: {missing[:3]}")
    summary = {
        "benchmark_directories": len(manifest["eligible_dirs"]),
        "response_rows": 0, "binary_response_rows": 0, "missing_response_rows": 0,
        "other_response_rows": 0, "positive_response_rows": 0,
        "subject_rows": 0, "item_rows": 0,
        "duplicate_subject_ids": 0, "duplicate_item_ids": 0,
        "duplicate_response_ids": 0, "orphan_subject_responses": 0,
        "orphan_item_responses": 0, "scoreable_pairs": 0,
        "scoreable_benchmarks": 0, "missing_item_content": 0,
        "missing_item_features": 0,
    }
    global_subjects = set()
    per_benchmark = {}
    for directory in manifest["eligible_dirs"]:
        root = data_root / directory
        benchmark = pd.read_parquet(root / "benchmarks.parquet").iloc[0]
        subjects = pd.read_parquet(root / "subjects.parquet", columns=["subject_id", "normalized_name"])
        items = pd.read_parquet(root / "items.parquet", columns=["item_id", "content", "item_features"])
        responses = pd.read_parquet(root / "response.parquet", columns=[
            "response_id", "subject_id", "item_id", "response"
        ])
        binary = responses["response"].isin((0, 1))
        valid = responses[binary].dropna(subset=["subject_id", "item_id"])
        eligible_pairs = int((valid.groupby("subject_id")["item_id"].nunique() >= 80).sum())
        record = {
            "release": str(benchmark.get("release", "")),
            "response_type": str(benchmark["response_type"]),
            "granularity": str(benchmark["granularity"]),
            "response_rows": len(responses),
            "binary_response_rows": int(binary.sum()),
            "missing_response_rows": int(responses["response"].isna().sum()),
            "other_response_rows": int((~binary & responses["response"].notna()).sum()),
            "positive_response_rows": int((responses["response"] == 1).sum()),
            "subject_rows": len(subjects), "item_rows": len(items),
            "distinct_subject_ids": int(subjects["subject_id"].nunique()),
            "distinct_item_ids": int(items["item_id"].nunique()),
            "duplicate_subject_ids": int(subjects["subject_id"].duplicated().sum()),
            "duplicate_item_ids": int(items["item_id"].duplicated().sum()),
            "duplicate_response_ids": int(responses["response_id"].duplicated().sum()),
            "orphan_subject_responses": int((~valid["subject_id"].isin(subjects["subject_id"])).sum()),
            "orphan_item_responses": int((~valid["item_id"].isin(items["item_id"])).sum()),
            "scoreable_pairs": eligible_pairs,
            "missing_item_content": int(items["content"].isna().sum()),
            "missing_item_features": int(items["item_features"].isna().sum()),
        }
        global_subjects.update(subjects["subject_id"].dropna().astype(str))
        for key in summary:
            if key in record and isinstance(record[key], int):
                summary[key] += record[key]
        if eligible_pairs:
            summary["scoreable_benchmarks"] += 1
        per_benchmark[directory] = record
        print(f"audited {directory}: {len(responses)} responses", flush=True)
    summary["distinct_subject_ids_global"] = len(global_subjects)
    summary["positive_fraction_among_binary"] = (
        summary["positive_response_rows"] / summary["binary_response_rows"]
        if summary["binary_response_rows"] else None
    )
    return {"source_revision": manifest["revision"], "summary": summary,
            "per_benchmark": per_benchmark}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
