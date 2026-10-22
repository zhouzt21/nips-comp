"""Paired summary of training-selected statistical priors across grouped folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import ALC_WEIGHTS


def summarize(priors: list[Path], evaluations: list[Path]) -> dict:
    fold_rows = []
    benchmarks = []
    brier_sums = {"candidate": np.zeros(6), "frozen_smoothed2": np.zeros(6),
                  "training_selected_05": np.zeros(6), "empirical_mean": np.zeros(6),
                  "mlp_prior": np.zeros(6), "mlp_residual2": np.zeros(6)}
    total_pairs = 0
    for fold, (prior_path, eval_path) in enumerate(zip(priors, evaluations)):
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        result = json.loads(eval_path.read_text(encoding="utf-8"))
        if set(result["benchmarks_used"]) & set(prior["train_benchmarks"]):
            raise ValueError("training data contaminated held-out evaluation")
        selected = prior["selected"]
        methods = {
            "candidate": f"public_prior:smoothed_{selected['public_prior']['alpha']:g}",
            "frozen_smoothed2": "smoothed_mean_2",
            "training_selected_05": f"smoothed_mean_{selected['smoothed_05']['alpha']:g}",
            "empirical_mean": "empirical_mean",
        }
        count = result["pairs_per_seed"][0]
        total_pairs += count
        fold_rows.append({"fold": fold, "pairs": count,
                          "candidate_prior": selected["public_prior"]["prior"],
                          "candidate_alpha": selected["public_prior"]["alpha"],
                          "baseline_alpha": selected["smoothed_05"]["alpha"],
                          "alc": {key: result["results"][method]["alc_mean"]
                                  for key, method in methods.items()}})
        for key, method in methods.items():
            brier_sums[key] += count*np.array(result["results"][method]["brier_mean"])
        for benchmark, info in result["by_benchmark"].items():
            benchmarks.append({"benchmark": benchmark, "fold": fold,
                               "pairs": info["pairs_per_seed"][0],
                               **{key: info["alc_mean"][method]
                                  for key, method in methods.items()}})
    output = {"scope": "reused public grouped CV folds, not an untouched test",
              "folds": fold_rows, "benchmarks": benchmarks, "pairs": total_pairs,
              "pair_equal_brier": {key: (value/total_pairs).tolist()
                                   for key, value in brier_sums.items()
                                   if key not in {"mlp_prior", "mlp_residual2"}},
              "pair_equal_alc": {}, "benchmark_equal_alc": {}, "paired_bootstrap": {}}
    for key in ("candidate", "frozen_smoothed2", "training_selected_05", "empirical_mean"):
        output["pair_equal_alc"][key] = float(brier_sums[key]/total_pairs @ ALC_WEIGHTS)
        output["benchmark_equal_alc"][key] = float(np.mean([row[key] for row in benchmarks]))
    rng = np.random.default_rng(20261022)
    for comparator in ("frozen_smoothed2", "training_selected_05"):
        diffs = np.array([row["candidate"]-row[comparator] for row in benchmarks])
        counts = np.array([row["pairs"] for row in benchmarks])
        draws = rng.integers(len(benchmarks), size=(10000, len(benchmarks)))
        samples = (diffs[draws]*counts[draws]).sum(axis=1)/counts[draws].sum(axis=1)
        output["paired_bootstrap"][comparator] = {
            "delta_alc": output["pair_equal_alc"]["candidate"]-output["pair_equal_alc"][comparator],
            "ci_95": np.quantile(samples, [.025, .975]).tolist(),
            "fold_deltas": [row["alc"]["candidate"]-row["alc"][comparator]
                            for row in fold_rows],
            "benchmarks_improved": int(np.sum(diffs < 0)),
            "benchmarks_total": len(diffs), "worst_benchmark_delta": float(np.max(diffs))}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors", nargs="+", type=Path, required=True)
    parser.add_argument("--evaluations", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.priors) != len(args.evaluations):
        parser.error("one prior per evaluation required")
    result = summarize(args.priors, args.evaluations)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("pair_equal_alc", "benchmark_equal_alc", "paired_bootstrap")}, indent=2))


if __name__ == "__main__":
    main()
