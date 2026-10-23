"""Create benchmark-grouped CV folds inside the frozen training pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def make_folds(parent: dict, *, seed: int = 2027, folds: int = 3,
               pair_cap: int = 100) -> list[dict]:
    if folds < 2 or pair_cap < 1:
        raise ValueError("Need at least two folds and a positive pair cap")
    counts = parent["scoreable_pair_counts"]
    train = parent["train_benchmarks"]
    scoreable = [d for d in train if counts[d] > 0]
    if len(scoreable) < folds * 3:
        raise ValueError("Too few scoreable training benchmarks for grouped CV")
    rng = np.random.default_rng(seed)
    rng.shuffle(scoreable)
    scoreable.sort(key=lambda d: -min(counts[d], pair_cap))
    assignments = [[] for _ in range(folds)]
    loads = [0] * folds
    for directory in scoreable:
        index = min(range(folds), key=lambda i: (loads[i], len(assignments[i]), i))
        assignments[index].append(directory)
        loads[index] += min(counts[directory], pair_cap)
    original_heldout = parent["development_benchmarks"] + parent["test_benchmarks"]
    result = []
    for fold, validation in enumerate(assignments):
        training = sorted(set(train) - set(validation))
        entry = {
            "source_revision": parent["source_revision"],
            "seed": seed, "cv_fold": fold, "cv_folds": folds,
            "cv_pair_cap": pair_cap,
            "scoreable_pair_counts": counts,
            "train_benchmarks": training,
            "development_benchmarks": sorted(validation),
            "test_benchmarks": sorted(original_heldout),
        }
        assert not set(training) & set(validation)
        assert not set(original_heldout) & set(training + validation)
        result.append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--pair-cap", type=int, default=100)
    args = parser.parse_args()
    parent_bytes = args.parent_split.read_bytes()
    parent = json.loads(parent_bytes)
    entries = make_folds(parent, seed=args.seed, folds=args.folds,
                         pair_cap=args.pair_cap)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        entry["cv_parent_split_sha256"] = hashlib.sha256(parent_bytes).hexdigest()
        path = args.out_dir / f"fold{entry['cv_fold']}_split.json"
        path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        print(f"{path}: {len(entry['development_benchmarks'])} benchmarks, "
              f"{sum(min(parent['scoreable_pair_counts'][d], args.pair_cap) for d in entry['development_benchmarks'])} capped pairs")


if __name__ == "__main__":
    main()
