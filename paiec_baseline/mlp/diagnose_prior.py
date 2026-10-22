"""Post-hoc B0 diagnosis on historical held-out CV folds (not deployable)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import _load_directory, _load_weights, _stable_seed, _vectors
from .model import forward_batch


def diagnose(data_root: Path, split_files: list[Path], weight_files: list[Path],
             pair_cap: int = 100, item_cap: int = 64) -> dict:
    rows = []
    for split_path, weight_path in zip(split_files, weight_files):
        split = json.loads(split_path.read_text(encoding="utf-8"))
        metadata = json.loads(weight_path.with_suffix(".json").read_text(encoding="utf-8"))
        directories = split["development_benchmarks"]
        if set(directories) & set(metadata["train_benchmarks"] + metadata["validation_benchmarks"]):
            raise ValueError("Training/validation benchmark leaked into diagnostic fold")
        weights = _load_weights(weight_path)
        for directory in directories:
            responses, subjects, items = _load_directory(data_root, directory)
            groups = [(sid, group) for sid, group in responses.groupby("subject_id", sort=False)
                      if sid in subjects and group["item_id"].nunique() >= 80]
            if len(groups) > pair_cap:
                rng = np.random.default_rng(_stable_seed(2026, directory, "pair_sample"))
                groups = [groups[i] for i in np.sort(rng.choice(len(groups), pair_cap, replace=False))]
            for sid, group in groups:
                item_ids = pd.unique(group["item_id"]).copy()
                rng = np.random.default_rng(_stable_seed(0, directory, sid))
                rng.shuffle(item_ids)
                eval_ids = item_ids[len(item_ids)//2:len(item_ids)//2+item_cap]
                evaluation = group[group["item_id"].isin(eval_ids)]
                if evaluation.empty:
                    continue
                x = _vectors(subjects[sid], items, evaluation, metadata.get("feature_mode", "all"))
                p = forward_batch(x, weights).astype(float)
                y = evaluation["response"].to_numpy(dtype=float)
                p_mean, y_mean = float(p.mean()), float(y.mean())
                p_var, y_var = float(p.var()), float(y.var())
                cov = float(np.mean((p-p_mean)*(y-y_mean)))
                rows.append({"benchmark": directory, "n": len(y), "p_mean": p_mean,
                             "y_mean": y_mean, "bias2": (p_mean-y_mean)**2,
                             "p_var": p_var, "y_var": y_var, "cov": cov,
                             "brier": float(np.mean((p-y)**2)),
                             "centered_item_gain": 2*cov-p_var,
                             "confidence": float(np.mean(np.abs(p-0.5))),
                             "p_low": float(np.mean(p < 0.1)),
                             "p_high": float(np.mean(p > 0.9))})
            print(f"diagnosed {directory}: {len(groups)} pairs", flush=True)
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        group = [row for row in rows if row["benchmark"] == benchmark]
        by_benchmark[benchmark] = {
            "pairs": len(group),
            **{key: float(np.mean([row[key] for row in group])) for key in
               ("p_mean", "y_mean", "bias2", "p_var", "y_var", "cov", "brier",
                "centered_item_gain", "confidence", "p_low", "p_high")}}
    result = {"scope": "historical post-hoc diagnostic; eval labels used only for error decomposition",
              "pairs": len(rows), "benchmarks": len(by_benchmark),
              "pair_equal": {key: float(np.mean([row[key] for row in rows])) for key in
                             ("p_mean", "y_mean", "bias2", "p_var", "y_var", "cov", "brier",
                              "centered_item_gain", "confidence", "p_low", "p_high")},
              "by_benchmark": by_benchmark}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.splits) != len(args.weights):
        parser.error("one weights file per split required")
    result = diagnose(args.data_root, args.splits, args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["pair_equal"], indent=2))


if __name__ == "__main__":
    main()
