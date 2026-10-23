"""Aggregate benchmark-grouped CV replays without pooling response rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def summarize(files: list[Path]) -> dict:
    fold_data = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if any(data["seeds"] != 1 or data["role"] != "development" for data in fold_data):
        raise ValueError("CV summary expects one-seed development replays")
    if len({data["source_revision"] for data in fold_data}) != 1:
        raise ValueError("CV files have different source revisions")
    total_pairs = sum(data["pairs_per_seed"][0] for data in fold_data)
    methods = {}
    for index, data in enumerate(fold_data):
        for name, record in data["results"].items():
            generic = re.sub(r"^fold\d+:", "mlp:", name)
            methods.setdefault(generic, {"alc_by_fold": [None] * len(files),
                                         "weighted_alc_sum": 0.0,
                                         "benchmark_alcs": []})
            methods[generic]["alc_by_fold"][index] = record["alc_mean"]
            methods[generic]["weighted_alc_sum"] += record["alc_mean"] * data["pairs_per_seed"][0]
            for benchmark in data["by_benchmark"].values():
                methods[generic]["benchmark_alcs"].append(
                    benchmark["alc_mean"][name])
    result_methods = {}
    for name, values in methods.items():
        if None in values["alc_by_fold"]:
            continue
        result_methods[name] = {
            "pair_equal_alc": values["weighted_alc_sum"] / total_pairs,
            "benchmark_equal_alc": sum(values["benchmark_alcs"]) / len(values["benchmark_alcs"]),
            "alc_by_fold": values["alc_by_fold"],
            "benchmarks": len(values["benchmark_alcs"]),
        }
    return {
        "source_revision": fold_data[0]["source_revision"],
        "input_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in files],
        "pairs_by_fold": [data["pairs_per_seed"][0] for data in fold_data],
        "total_pairs": total_pairs,
        "methods": result_methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, values in sorted(result["methods"].items(),
                               key=lambda item: item[1]["pair_equal_alc"]):
        print(f"{name}: pair={values['pair_equal_alc']:.6f} "
              f"benchmark={values['benchmark_equal_alc']:.6f}")


if __name__ == "__main__":
    main()
