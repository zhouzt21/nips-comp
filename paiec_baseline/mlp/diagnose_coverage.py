"""Count exact subject-name transfer across benchmark-disjoint pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import _string


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--public-prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    prior = json.loads(args.public_prior.read_text(encoding="utf-8"))
    known = set(prior["priors"])
    result = {"source_revision": prior["source_revision"], "known_training_names": len(known)}
    for role in ("development", "test"):
        by_benchmark = {}
        for directory in split[f"{role}_benchmarks"]:
            root = args.data_root / directory
            response = pd.read_parquet(root / "response.parquet",
                                       columns=["subject_id", "item_id", "response"])
            response = response[response["response"].isin((0, 1))]
            eligible = set(response.groupby("subject_id")["item_id"].nunique().loc[lambda x: x >= 80].index)
            subjects = pd.read_parquet(root / "subjects.parquet",
                                       columns=["subject_id", "normalized_name"])
            names = [_string(name) for sid, name in subjects[["subject_id", "normalized_name"]]
                     .drop_duplicates("subject_id").itertuples(index=False, name=None) if sid in eligible]
            by_benchmark[directory] = {"scoreable_pairs": len(names),
                                       "known_subject_names": sum(name in known for name in names),
                                       "nonempty_subject_names": sum(bool(name) for name in names)}
        result[role] = by_benchmark
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({role: {
        "scoreable_pairs": sum(x["scoreable_pairs"] for x in result[role].values()),
        "known_subject_names": sum(x["known_subject_names"] for x in result[role].values()),
        "nonempty_subject_names": sum(x["nonempty_subject_names"] for x in result[role].values()),
    } for role in ("development", "test")}, indent=2))


if __name__ == "__main__":
    main()
