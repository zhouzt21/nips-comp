"""Freeze benchmark-disjoint train, development, and untouched test pools.

The three largest scoreable benchmarks are assigned to different pools before
any model is fitted. Remaining scoreable benchmarks are shuffled by a fixed
seed; benchmarks without an 80-item subject pair stay in training only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_split(data_root: Path, seed: int = 2026) -> dict:
    manifest = json.loads((data_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    pair_counts = {}
    for directory in manifest["eligible_dirs"]:
        path = data_root / directory / "response.parquet"
        frame = pd.read_parquet(path, columns=["subject_id", "item_id", "response"])
        frame = frame[frame["response"].isin((0, 1))].dropna(subset=["subject_id", "item_id"])
        pair_counts[directory] = int((frame.groupby("subject_id")["item_id"].nunique() >= 80).sum())

    ranked = sorted((d for d, n in pair_counts.items() if n), key=lambda d: (-pair_counts[d], d))
    if len(ranked) < 15:
        raise ValueError("Need at least 15 scoreable benchmarks for a three-way split")
    train = [ranked[0]]
    development = [ranked[1]]
    test = [ranked[2]]
    remainder = np.asarray(ranked[3:], dtype=object)
    np.random.default_rng(seed).shuffle(remainder)
    n_development = max(1, round(len(ranked) * 0.15))
    n_test = max(1, round(len(ranked) * 0.15))
    development += remainder[:n_development - 1].tolist()
    test += remainder[n_development - 1:n_development + n_test - 2].tolist()
    train += remainder[n_development + n_test - 2:].tolist()
    train += [d for d, n in pair_counts.items() if n == 0]
    assert set(train).isdisjoint(development) and set(train).isdisjoint(test)
    assert set(development).isdisjoint(test)
    assert set(train + development + test) == set(manifest["eligible_dirs"])
    return {
        "source_revision": manifest["revision"], "seed": seed,
        "split_rule": "top 3 by scoreable pair count to train/development/test; seeded shuffle of remainder; zero-scoreable to train",
        "scoreable_pair_counts": pair_counts,
        "train_benchmarks": sorted(train),
        "development_benchmarks": sorted(development),
        "test_benchmarks": sorted(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    split = make_split(args.data_root, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"benchmarks": len(split[k]), "scoreable_pairs": sum(split["scoreable_pair_counts"][d] for d in split[k])}
                      for k in ("train_benchmarks", "development_benchmarks", "test_benchmarks")}, indent=2))


if __name__ == "__main__":
    main()
