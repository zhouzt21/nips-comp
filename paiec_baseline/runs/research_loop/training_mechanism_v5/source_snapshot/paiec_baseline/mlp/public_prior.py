"""Fit a simple subject-name prior from training benchmarks only.

Each subject–benchmark mean contributes one observation, matching the
competition's pair weighting more closely than pooling all response rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import _string


def fit(data_root: Path, split_file: Path, strength: float = 3.0) -> dict:
    manifest = json.loads((data_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    split = json.loads(split_file.read_text(encoding="utf-8"))
    if split["source_revision"] != manifest["revision"]:
        raise ValueError("split and data manifest refer to different revisions")
    records = []
    for directory in split["train_benchmarks"]:
        root = data_root / directory
        responses = pd.read_parquet(root / "response.parquet", columns=["subject_id", "response"])
        responses = responses[responses["response"].isin((0, 1))].dropna(subset=["subject_id"])
        if responses.empty:
            continue
        means = responses.groupby("subject_id", sort=False)["response"].mean()
        subjects = pd.read_parquet(root / "subjects.parquet", columns=["subject_id", "normalized_name"])
        names = subjects.drop_duplicates("subject_id").set_index("subject_id")["normalized_name"]
        for subject_id, mean in means.items():
            if subject_id in names:
                name = _string(names.loc[subject_id])
                if name:
                    records.append((name, float(mean)))
    if not records:
        raise ValueError("No training subject–benchmark means available")
    frame = pd.DataFrame(records, columns=["normalized_name", "pair_mean"])
    global_mean = float(frame["pair_mean"].mean())
    grouped = frame.groupby("normalized_name")["pair_mean"].agg(["sum", "count"])
    priors = {name: float((row["sum"] + strength * global_mean) / (row["count"] + strength))
              for name, row in grouped.iterrows()}
    return {
        "source_revision": manifest["revision"],
        "split_file": str(split_file),
        "train_benchmarks": split["train_benchmarks"],
        "strength": strength,
        "global_mean": global_mean,
        "training_pairs": len(frame),
        "subject_names": len(priors),
        "priors": priors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.strength < 0:
        parser.error("strength must be nonnegative")
    result = fit(args.data_root, args.split_file, args.strength)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("global_mean", "training_pairs", "subject_names")}, indent=2))


if __name__ == "__main__":
    main()
