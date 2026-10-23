"""All-method paired tables and frozen gate checks for a research round."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import tempfile
from mlp.reporting.summarize_candidate import summarize


def summarize_round(root: Path, candidates: list[str]) -> dict:
    files = [root / f"fold{fold}_evaluation.json" for fold in range(3)]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        augmented = []
        for fold, path in enumerate(files):
            data = json.loads(path.read_text(encoding="utf-8"))
            name = f"smoothed_mean_{[2,2,4][fold]}"
            data["results"]["training_selected_smoothed"] = data["results"][name]
            for entry in data["by_benchmark"].values():
                entry["alc_mean"]["training_selected_smoothed"] = entry["alc_mean"][name]
            target = Path(tmp) / path.name
            target.write_text(json.dumps(data), encoding="utf-8")
            augmented.append(target)
        for candidate in candidates:
            report = summarize(augmented, candidate, ["smoothed_mean_2", "training_selected_smoothed",
                                                     "empirical_mean", "original:residual_2"])
            report["source_files"] = [str(p) for p in files]
            for fold, entry in enumerate(report["folds"]):
                entry["source"] = str(files[fold])
            gates = {}
            for baseline in ("smoothed_mean_2", "training_selected_smoothed"):
                c = report["comparisons"][baseline]
                gates[baseline] = {
                    "delta_at_most_minus_005": c["pair_equal_delta"] <= -.005,
                    "all_folds_improve": all(d < 0 for d in c["fold_deltas"]),
                    "benchmark_equal_improves": report["benchmark_equal_alc"][candidate] < report["benchmark_equal_alc"][baseline],
                    "ci_upper_negative": c["cluster_bootstrap_ci95"][1] < 0,
                    "worst_regression_at_most_03": c["worst_benchmark_delta"] <= .03,
                }
            report["statistical_gate"] = gates
            report["gate_pass"] = all(all(g.values()) for g in gates.values())
            report["blind_generalization_gate"] = False
            results[candidate] = report
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (root / "per_benchmark_budget.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["fold", "benchmark", "pairs", "method", "B0", "B1", "B3", "B7", "B15", "B31", "ALC"])
        for fold, path in enumerate(files):
            data = json.loads(path.read_text(encoding="utf-8"))
            for benchmark, row in data["by_benchmark"].items():
                for method, value in row["alc_mean"].items():
                    writer.writerow([fold, benchmark, row["pairs_per_seed"][0], method,
                                     *row.get("brier_mean", {}).get(method, [None]*6), value])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    args = parser.parse_args()
    result = summarize_round(args.root, args.candidate)
    print(json.dumps({name: {key: r[key] for key in ("pair_equal_alc", "comparisons", "gate_pass")}
                      for name, r in result.items()}, indent=2))
