"""Training-only benchmark-balanced empirical prior for local binary labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_lookup(cache: dict, train_directories: list[str]) -> dict:
    probs = []
    weights = []
    contributing = []
    for directory in train_directories:
        entry = cache["by_benchmark"][directory]
        if not entry["pair_rates"]:
            continue
        rates = np.asarray(entry["pair_rates"], dtype=float)
        counts = np.asarray(entry["pair_response_counts"], dtype=float)
        # Jeffreys smoothing accounts weakly for finite public pair estimates.
        q = (rates*counts + .5)/(counts + 1)
        probs.extend(q.tolist())
        weights.extend([1/len(q)]*len(q))
        contributing.append(directory)
    if not probs:
        raise ValueError("No training pair rates")
    q = np.asarray(probs)
    w = np.asarray(weights)
    w /= w.sum()
    logq, log1q, logw = np.log(q), np.log1p(-q), np.log(w)
    table = np.full((32, 32), .5)
    for n in range(32):
        for s in range(n+1):
            likelihood = logw + s*logq + (n-s)*log1q
            likelihood -= likelihood.max()
            posterior_weights = np.exp(likelihood)
            table[n, s] = np.dot(posterior_weights, q)/posterior_weights.sum()
    return {"source_revision": cache["source_revision"],
            "train_benchmarks": train_directories,
            "method": "benchmark-balanced discrete empirical Bayes; Jeffreys-smoothed pair rates",
            "training_benchmarks_with_scoreable_pairs": len(contributing),
            "training_pairs": len(q), "posterior_lookup": table.tolist(),
            "prior_mean": float(table[0, 0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fold-splits", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.fold_splits:
        split = json.loads(path.read_text(encoding="utf-8"))
        if set(split["train_benchmarks"]) & set(split["development_benchmarks"]):
            raise ValueError("fold train/development overlap")
        fitted = fit_lookup(cache, split["train_benchmarks"])
        out = args.out_dir / f"fold{split['cv_fold']}_lookup.json"
        out.write_text(json.dumps(fitted, indent=2) + "\n", encoding="utf-8")
        print(f"{out}: mean={fitted['prior_mean']:.6f}; pairs={fitted['training_pairs']}")


if __name__ == "__main__":
    main()
