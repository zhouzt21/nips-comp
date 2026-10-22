"""Estimate empirical-mean Brier ALC on public, held-out benchmark items.

This is a *local proxy*, not the private Codabench score.  It reproduces the
published 50/50 item split, six nested budgets and equal pair weighting.  One
random recorded trial is acquired per selected item; every evaluation trial is
scored.  Run from the repository root after preparing a public payload:

    python tools/evaluate_mean.py --data-root payload/data --seeds 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)


def _seed(seed: int, benchmark: str, subject: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}\0{benchmark}\0{subject}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def _score_pair(group: pd.DataFrame, seed: int, benchmark: str, subject: str):
    item_ids = group["item_id"].unique()
    if len(item_ids) < 80:
        return None
    rng = np.random.default_rng(_seed(seed, benchmark, subject))
    rng.shuffle(item_ids)
    cut = len(item_ids) // 2
    acquisition, evaluation = item_ids[:cut], item_ids[cut:]
    acquired = group[group["item_id"].isin(acquisition)]
    evaluated = group[group["item_id"].isin(evaluation)]
    # Sampling a single observed response per item follows the interface's
    # one-revealed-response semantics while keeping all trials of an item in
    # the same pool.
    chosen = acquired.groupby("item_id", sort=False).sample(n=1, random_state=int(rng.integers(2**31)))
    labels = chosen.set_index("item_id").loc[acquisition, "response"].to_numpy(dtype=float)
    outcomes = evaluated["response"].to_numpy(dtype=float)
    if len(labels) < 31 or len(outcomes) == 0:
        return None
    scores = []
    for budget in BUDGETS:
        probability = float(labels[:budget].mean()) if budget else 0.5
        scores.append(float(np.mean((probability - outcomes) ** 2)))
    return scores, len(outcomes)


def evaluate(data_root: Path, seeds: int) -> dict:
    manifest = json.loads((data_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    sums = np.zeros((seeds, len(BUDGETS)), dtype=float)
    counts = np.zeros(seeds, dtype=int)
    evaluated_responses = np.zeros(seeds, dtype=int)
    directories_used = 0
    for directory in manifest["eligible_dirs"]:
        path = data_root / directory / "response.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["subject_id", "item_id", "response"])
        frame = frame[frame["response"].isin((0, 1))].dropna(subset=["subject_id", "item_id"])
        if frame.empty:
            continue
        directories_used += 1
        for subject, group in frame.groupby("subject_id", sort=False):
            if group["item_id"].nunique() < 80:
                continue
            for seed in range(seeds):
                result = _score_pair(group, seed, directory, str(subject))
                if result is None:
                    continue
                scores, n_responses = result
                sums[seed] += scores
                counts[seed] += 1
                evaluated_responses[seed] += n_responses
    if not np.all(counts):
        raise ValueError("No eligible subject–benchmark pairs with at least 80 distinct items")
    by_seed = sums / counts[:, None]
    alc_by_seed = by_seed @ np.asarray(ALC_WEIGHTS)
    return {
        "source_revision": manifest.get("revision"),
        "method": "empirical_mean_random_acquisition_public_proxy",
        "budgets": list(BUDGETS),
        "alc_weights": list(ALC_WEIGHTS),
        "seeds": seeds,
        "benchmark_directories_used": directories_used,
        "pairs_per_seed": counts.tolist(),
        "evaluation_responses_per_seed": evaluated_responses.tolist(),
        "brier_by_seed": by_seed.tolist(),
        "brier_mean": by_seed.mean(axis=0).tolist(),
        "brier_alc_by_seed": alc_by_seed.tolist(),
        "brier_alc_mean": float(alc_by_seed.mean()),
        "brier_alc_std": float(alc_by_seed.std(ddof=1)) if seeds > 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("payload/data"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    result = evaluate(args.data_root, args.seeds)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
