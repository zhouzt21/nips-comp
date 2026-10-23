"""Fit a portable Beta-Binomial pair prior from public training benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import ALC_WEIGHTS, BUDGETS


def cache_pair_rates(data_root: Path, directories: list[str]) -> dict:
    manifest = json.loads((data_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    if set(directories) - set(manifest["eligible_dirs"]):
        raise ValueError("unrecognized benchmark directory")
    result = {"source_revision": manifest["revision"], "by_benchmark": {}}
    for directory in directories:
        rows = pd.read_parquet(data_root / directory / "response.parquet",
                               columns=["subject_id", "item_id", "response"])
        rows = rows[rows["response"].isin((0, 1))].dropna(subset=["subject_id", "item_id"])
        grouped = rows.groupby("subject_id").agg(
            item_count=("item_id", "nunique"), response_count=("response", "size"),
            success_rate=("response", "mean"))
        grouped = grouped[grouped["item_count"] >= 80]
        sampled = rows.sample(n=min(400, len(rows)), random_state=2026)
        sampled_counts = sampled.groupby("subject_id").size()
        scoreable_ids = set(grouped.index)
        result["by_benchmark"][directory] = {
            "pair_rates": grouped["success_rate"].astype(float).tolist(),
            "pair_response_counts": grouped["response_count"].astype(int).tolist(),
            "sampled_rows": int(len(sampled)),
            "sampled_pairs": int(len(sampled_counts)),
            "sampled_scoreable_pairs": int(sum(sid in scoreable_ids for sid in sampled_counts.index)),
            "sampled_scoreable_rows": int(sampled["subject_id"].isin(scoreable_ids).sum()),
            "sampled_pair_median_rows": float(sampled_counts.median()) if len(sampled_counts) else 0.0}
        print(f"cached {directory}: {len(grouped)} scoreable pairs", flush=True)
    return result


def select_prior(cache: dict, train_directories: list[str],
                 alphas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)) -> dict:
    rates = np.array([rate for directory in train_directories
                      for rate in cache["by_benchmark"][directory]["pair_rates"]], dtype=float)
    if len(rates) < 1:
        raise ValueError("No scoreable training pairs")
    mu = float(rates.mean())
    def proxy_loss(prior: float, alpha: float) -> float:
        losses = []
        for n in BUDGETS:
            # Exchangeable Bernoulli approximation, using full training-pair rate q.
            # It is a fitting proxy only, never an evaluation score.
            q = rates
            risk = q*(1-q) + (alpha*alpha*(prior-q)**2 + n*q*(1-q))/(alpha+n)**2
            losses.append(float(risk.mean()))
        return float(np.dot(losses, ALC_WEIGHTS))
    fit = {"public_prior": mu, "smoothed_05": 0.5}
    selected = {}
    for key, prior in fit.items():
        proxy = {str(alpha): proxy_loss(prior, alpha) for alpha in alphas}
        winner = min(alphas, key=lambda a: proxy[str(a)])
        selected[key] = {"prior": prior, "alpha": winner, "proxy_by_alpha": proxy}
    return {"source_revision": cache["source_revision"], "train_benchmarks": train_directories,
            "training_pairs": len(rates), "training_pair_rate_mean": mu,
            "selection_method": "training-only exchangeable Binomial risk proxy",
            "selected": selected, "priors": {}, "global_mean": mu}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent-split", type=Path, required=True)
    parser.add_argument("--fold-splits", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.parent_split.read_text(encoding="utf-8"))
    cache = cache_pair_rates(args.data_root, parent["train_benchmarks"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pair_rates.json").write_text(json.dumps(cache) + "\n", encoding="utf-8")
    for path in args.fold_splits:
        split = json.loads(path.read_text(encoding="utf-8"))
        if set(split["train_benchmarks"]) & set(split["development_benchmarks"]):
            raise ValueError("fold train/development overlap")
        prior = select_prior(cache, split["train_benchmarks"])
        out = args.out_dir / f"fold{split['cv_fold']}_prior.json"
        out.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
        print(f"{out}: {prior['selected']}", flush=True)


if __name__ == "__main__":
    main()
