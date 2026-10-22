"""Summarize already-observed grouped CV results; never selects a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(files: list[Path]) -> dict:
    budgets = [0, 1, 3, 7, 15, 31]
    weights = np.array([.1, .2, .2, .2, .2, .1])
    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    methods = ["smoothed_mean_2", "constant_0.5", "empirical_mean", "mlp:prior", "mlp:residual_2"]
    totals = {method: np.zeros(6) for method in methods}
    total_pairs = 0
    benchmark_rows = []
    fold_rows = []
    for record in records:
        pairs = record["pairs_per_seed"][0]
        total_pairs += pairs
        mapped = {method: method.replace("mlp:", "fold" + str(len(fold_rows)) + ":") for method in methods}
        for method, name in mapped.items():
            totals[method] += pairs * np.array(record["results"][name]["brier_mean"])
        fold_rows.append({"pairs": pairs, "alc": {
            method: record["results"][name]["alc_mean"] for method, name in mapped.items()}})
        for benchmark, info in record["by_benchmark"].items():
            base = info["alc_mean"]["smoothed_mean_2"]
            benchmark_rows.append({
                "benchmark": benchmark, "pairs": info["pairs_per_seed"][0],
                "baseline_alc": base,
                "prior_delta_alc": info["alc_mean"][mapped["mlp:prior"]] - base,
                "residual_delta_alc": info["alc_mean"][mapped["mlp:residual_2"]] - base,
            })
    means = {method: value / total_pairs for method, value in totals.items()}
    base = means["smoothed_mean_2"]
    delta = {method: {
        "delta_brier": (value-base).tolist(),
        "weighted_contribution": ((value-base)*weights).tolist(),
        "delta_alc": float((value-base) @ weights),
    } for method, value in means.items() if method != "smoothed_mean_2"}
    return {
        "source_files": [str(p) for p in files], "scope": "historical diagnostic only",
        "budgets": budgets, "alc_weights": weights.tolist(), "total_pairs": total_pairs,
        "brier": {k: v.tolist() for k, v in means.items()}, "delta_vs_smoothed2": delta,
        "folds": fold_rows,
        "benchmarks": sorted(benchmark_rows, key=lambda row: row["residual_delta_alc"], reverse=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": result["total_pairs"], "delta": result["delta_vs_smoothed2"],
                      "worst_benchmarks": result["benchmarks"][:5]}, indent=2))


if __name__ == "__main__":
    main()
