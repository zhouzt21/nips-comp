"""Paired fold summary for one frozen candidate versus fixed comparators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import ALC_WEIGHTS


def summarize(files: list[Path], candidate: str, baselines: list[str]) -> dict:
    methods = [candidate, *baselines]
    brier_sums = {method: np.zeros(6) for method in methods}
    folds, benchmarks = [], []
    total_pairs = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        count = data["pairs_per_seed"][0]
        total_pairs += count
        folds.append({"source": str(path), "pairs": count,
                      "alc": {method: data["results"][method]["alc_mean"] for method in methods}})
        for method in methods:
            brier_sums[method] += count*np.array(data["results"][method]["brier_mean"])
        for benchmark, info in data["by_benchmark"].items():
            benchmarks.append({"benchmark": benchmark, "pairs": info["pairs_per_seed"][0],
                               **{method: info["alc_mean"][method] for method in methods}})
    pair_equal = {method: float(value/total_pairs @ ALC_WEIGHTS)
                  for method, value in brier_sums.items()}
    bench_equal = {method: float(np.mean([row[method] for row in benchmarks]))
                   for method in methods}
    rng = np.random.default_rng(20261022)
    comparisons = {}
    for baseline in baselines:
        deltas = np.array([row[candidate]-row[baseline] for row in benchmarks])
        counts = np.array([row["pairs"] for row in benchmarks])
        draws = rng.integers(len(benchmarks), size=(10000, len(benchmarks)))
        boot = (deltas[draws]*counts[draws]).sum(axis=1)/counts[draws].sum(axis=1)
        comparisons[baseline] = {
            "pair_equal_delta": pair_equal[candidate]-pair_equal[baseline],
            "fold_deltas": [row["alc"][candidate]-row["alc"][baseline] for row in folds],
            "benchmarks_improved": int((deltas < 0).sum()),
            "benchmarks_total": len(deltas),
            "worst_benchmark": max(benchmarks, key=lambda row: row[candidate]-row[baseline])["benchmark"],
            "worst_benchmark_delta": float(np.max(deltas)),
            "cluster_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist()}
    return {"source_files": [str(p) for p in files],
            "scope": "reused public grouped CV folds; historical comparison, not blind test",
            "pair_equal_brier": {method: (value/total_pairs).tolist()
                                 for method, value in brier_sums.items()},
            "pair_equal_alc": pair_equal, "benchmark_equal_alc": bench_equal,
            "folds": folds, "by_benchmark": benchmarks, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = summarize(args.files, args.candidate, args.baseline)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("pair_equal_alc", "benchmark_equal_alc", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
