"""Replay a bounded public-data proxy of PAIEC's six-budget Brier ALC.

Benchmark directories are disjoint across train, development, and test. For
each selected subject–benchmark pair, all recorded responses of an item stay
together in the 50/50 acquisition/evaluation split. One acquisition response
per item is revealed. Evaluation may sample a fixed maximum number of items
per pair for runtime; this approximation is recorded in the output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import FEATURE_DIM, HIDDEN_DIMS, _ITEM_FIELDS, _SUBJECT_FIELDS, _string, encode_mode, forward_batch


BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = np.asarray((0.1, 0.2, 0.2, 0.2, 0.2, 0.1))


def _stable_seed(seed: int, directory: str, subject_id: object) -> int:
    digest = hashlib.blake2b(f"{seed}\0{directory}\0{subject_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _load_weights(path: Path) -> tuple[np.ndarray, ...]:
    expected = (
        ("W1", (FEATURE_DIM, HIDDEN_DIMS[0])), ("b1", (HIDDEN_DIMS[0],)),
        ("W2", (HIDDEN_DIMS[0], HIDDEN_DIMS[1])), ("b2", (HIDDEN_DIMS[1],)),
        ("W3", (HIDDEN_DIMS[1], 1)), ("b3", (1,)),
    )
    with np.load(path, allow_pickle=False) as artifact:
        if "W" in artifact:
            expected = (("W", (FEATURE_DIM, 1)), ("b", (1,)))
        arrays = tuple(np.asarray(artifact[name], dtype=np.float32) for name, _ in expected)
    for array, (_, shape) in zip(arrays, expected):
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid MLP weights in {path}")
    return arrays


def _load_directory(data_root: Path, directory: str):
    table_dir = data_root / directory
    responses = pd.read_parquet(table_dir / "response.parquet", columns=[
        "subject_id", "item_id", "response", "interactors"
    ])
    responses = responses[responses["response"].isin((0, 1))].dropna(subset=["subject_id", "item_id"])
    subjects = pd.read_parquet(table_dir / "subjects.parquet", columns=["subject_id", *_SUBJECT_FIELDS])
    items = pd.read_parquet(table_dir / "items.parquet", columns=["item_id", "content", "item_features"])
    subjects = subjects.drop_duplicates("subject_id").set_index("subject_id").to_dict("index")
    items = items.drop_duplicates("item_id").set_index("item_id").to_dict("index")
    return responses, subjects, items


def _vectors(subject: dict, items: dict, rows: pd.DataFrame,
             feature_mode: str = "all") -> np.ndarray:
    feature_cache = {}
    vectors = []
    for item_id, interactors in rows[["item_id", "interactors"]].itertuples(index=False, name=None):
        key = (item_id, str(interactors))
        if key not in feature_cache:
            item = items[item_id]
            feature_cache[key] = encode_mode([subject, {
                "item_content": item["content"],
                "item_features": item["item_features"],
                "interactors": interactors,
            }], feature_mode)
        vectors.append(feature_cache[key])
    return np.stack(vectors)


def _score_group(group: pd.DataFrame, subject: dict, items: dict, directory: str,
                 subject_id: object, seed: int, max_eval_items: int,
                 weights: dict[str, tuple[np.ndarray, ...]], alphas: tuple[float, ...],
                 public_prior: float | None = None,
                 weight_modes: dict[str, str] | None = None,
                 mixes: tuple[float, ...] = (1.0,),
                 eb_table: np.ndarray | None = None):
    item_ids = pd.unique(group["item_id"]).copy()
    if len(item_ids) < 80:
        return None
    rng = np.random.default_rng(_stable_seed(seed, directory, subject_id))
    rng.shuffle(item_ids)
    cut = len(item_ids) // 2
    acquisition_ids = item_ids[:31]
    evaluation_ids = item_ids[cut:cut + max_eval_items] if max_eval_items else item_ids[cut:]
    acquisition = group[group["item_id"].isin(acquisition_ids)]
    acquisition = acquisition.sample(frac=1, random_state=int(rng.integers(2**31)))
    acquisition = acquisition.drop_duplicates("item_id").set_index("item_id").loc[acquisition_ids].reset_index()
    evaluation = group[group["item_id"].isin(evaluation_ids)]
    if len(acquisition) != 31 or evaluation.empty:
        return None
    rows = pd.concat((acquisition, evaluation), ignore_index=True)
    feature_cache = {}
    acquired_y = acquisition["response"].to_numpy(dtype=np.float32)
    evaluation_y = evaluation["response"].to_numpy(dtype=np.float32)
    scores = {}
    for budget in BUDGETS:
        p = float(acquired_y[:budget].mean()) if budget else 0.5
        scores.setdefault("empirical_mean", []).append(float(np.mean((p - evaluation_y) ** 2)))
        scores.setdefault("constant_0.5", []).append(float(np.mean((0.5 - evaluation_y) ** 2)))
        for alpha in alphas:
            smoothed = (alpha * 0.5 + float(acquired_y[:budget].sum())) / (alpha + budget)
            scores.setdefault(f"smoothed_mean_{alpha:g}", []).append(
                float(np.mean((smoothed - evaluation_y) ** 2)))
        if public_prior is not None:
            scores.setdefault("public_prior:prior", []).append(
                float(np.mean((public_prior - evaluation_y) ** 2)))
            for alpha in alphas:
                smoothed = (alpha * public_prior + float(acquired_y[:budget].sum())) / (alpha + budget)
                scores.setdefault(f"public_prior:smoothed_{alpha:g}", []).append(
                    float(np.mean((smoothed - evaluation_y) ** 2)))
        if eb_table is not None:
            posterior = float(eb_table[budget, int(acquired_y[:budget].sum())])
            scores.setdefault("empirical_bayes", []).append(
                float(np.mean((posterior - evaluation_y) ** 2)))
    for name, params in weights.items():
        mode = (weight_modes or {}).get(name, "all")
        if mode not in feature_cache:
            feature_cache[mode] = _vectors(subject, items, rows, mode)
        predictions = forward_batch(feature_cache[mode], params)
        acquired_prior, evaluation_prior = predictions[:31], predictions[31:]
        prior_brier = float(np.mean((evaluation_prior - evaluation_y) ** 2))
        scores[f"{name}:prior"] = [prior_brier] * len(BUDGETS)
        residuals = np.cumsum(acquired_y - acquired_prior)
        for alpha in alphas:
            values = []
            shrink_values = []
            for budget in BUDGETS:
                correction = float(residuals[budget - 1]) / (budget + alpha) if budget else 0.0
                adjusted = np.clip(evaluation_prior + correction, 0.0, 1.0)
                values.append(float(np.mean((adjusted - evaluation_y) ** 2)))
                local = (alpha * evaluation_prior + float(acquired_y[:budget].sum())) / (alpha + budget)
                shrink_values.append(float(np.mean((local - evaluation_y) ** 2)))
            scores[f"{name}:residual_{alpha:g}"] = values
            scores[f"{name}:shrink_{alpha:g}"] = shrink_values
        for mix in mixes:
            if mix == 1.0:
                continue
            acquired_mixed = 0.5 + mix * (acquired_prior - 0.5)
            evaluation_mixed = 0.5 + mix * (evaluation_prior - 0.5)
            mixed_residuals = np.cumsum(acquired_y - acquired_mixed)
            for alpha in alphas:
                values = []
                for budget in BUDGETS:
                    correction = float(mixed_residuals[budget - 1]) / (budget + alpha) if budget else 0.0
                    adjusted = np.clip(evaluation_mixed + correction, 0.0, 1.0)
                    values.append(float(np.mean((adjusted - evaluation_y) ** 2)))
                scores[f"{name}:mix_{mix:g}:residual_{alpha:g}"] = values
    return scores, len(evaluation)


def evaluate(data_root: Path, split_file: Path, role: str,
             weight_files: dict[str, Path], *, seeds: int = 1,
             max_eval_items: int = 64, max_pairs_per_benchmark: int = 0,
             alphas: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0),
             public_prior_file: Path | None = None,
             mixes: tuple[float, ...] = (1.0,),
             eb_table_file: Path | None = None) -> dict:
    manifest = json.loads((data_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    split = json.loads(split_file.read_text(encoding="utf-8"))
    if split["source_revision"] != manifest["revision"]:
        raise ValueError("split and data manifest refer to different revisions")
    directories = split[f"{role}_benchmarks"]
    if any(d not in manifest["eligible_dirs"] for d in directories):
        raise ValueError("split contains a directory absent from the data manifest")
    weight_modes = {}
    for name, path in weight_files.items():
        metadata_path = path.with_suffix(".json")
        if role == "test" and not metadata_path.is_file():
            raise ValueError(f"Test evaluation requires training metadata beside {path}")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            weight_modes[name] = metadata.get("feature_mode", "all")
            if weight_modes[name] not in {"all", "subject", "block"}:
                raise ValueError(f"Unknown feature mode in {metadata_path}")
            forbidden = set(metadata["train_benchmarks"])
            if role == "test":
                forbidden.update(metadata["validation_benchmarks"])
            if set(directories) & forbidden:
                raise ValueError(f"{name} weights were fitted on a {role} benchmark")
    weights = {name: _load_weights(path) for name, path in weight_files.items()}
    public_prior = None
    if public_prior_file is not None:
        public_prior = json.loads(public_prior_file.read_text(encoding="utf-8"))
        if public_prior["source_revision"] != manifest["revision"]:
            raise ValueError("public prior and data manifest refer to different revisions")
        if set(directories) & set(public_prior["train_benchmarks"]):
            raise ValueError("public prior has seen an evaluated benchmark")
    eb_table = None
    if eb_table_file is not None:
        eb_metadata = json.loads(eb_table_file.read_text(encoding="utf-8"))
        if eb_metadata["source_revision"] != manifest["revision"]:
            raise ValueError("empirical Bayes prior and data revision disagree")
        if set(directories) & set(eb_metadata["train_benchmarks"]):
            raise ValueError("empirical Bayes prior has seen an evaluated benchmark")
        eb_table = np.asarray(eb_metadata["posterior_lookup"], dtype=float)
        if eb_table.shape != (32, 32) or not np.isfinite(eb_table).all() or np.any((eb_table < 0) | (eb_table > 1)):
            raise ValueError("invalid empirical Bayes posterior lookup")
    sums = defaultdict(lambda: np.zeros((seeds, len(BUDGETS)), dtype=float))
    benchmark_sums = defaultdict(lambda: defaultdict(lambda: np.zeros((seeds, len(BUDGETS)), dtype=float)))
    benchmark_counts = defaultdict(lambda: np.zeros(seeds, dtype=int))
    counts = np.zeros(seeds, dtype=int)
    response_counts = np.zeros(seeds, dtype=int)
    prior_name_hits = np.zeros(seeds, dtype=int)
    used_dirs = []
    for directory in directories:
        responses, subjects, items = _load_directory(data_root, directory)
        groups = [(subject_id, group) for subject_id, group in responses.groupby("subject_id", sort=False)
                  if group["item_id"].nunique() >= 80 and subject_id in subjects]
        if max_pairs_per_benchmark and len(groups) > max_pairs_per_benchmark:
            rng = np.random.default_rng(_stable_seed(2026, directory, "pair_sample"))
            chosen = np.sort(rng.choice(len(groups), size=max_pairs_per_benchmark, replace=False))
            groups = [groups[i] for i in chosen]
        if groups:
            used_dirs.append(directory)
        for seed in range(seeds):
            for subject_id, group in groups:
                prior_value = None
                subject_name = None
                if public_prior is not None:
                    subject_name = _string(subjects[subject_id].get("normalized_name"))
                    prior_value = public_prior["priors"].get(subject_name, public_prior["global_mean"])
                result = _score_group(group, subjects[subject_id], items, directory, subject_id,
                                      seed, max_eval_items, weights, alphas, prior_value,
                                      weight_modes, mixes, eb_table)
                if result is None:
                    continue
                group_scores, n_responses = result
                for name, values in group_scores.items():
                    sums[name][seed] += values
                    benchmark_sums[directory][name][seed] += values
                counts[seed] += 1
                if public_prior is not None and subject_name in public_prior["priors"]:
                    prior_name_hits[seed] += 1
                benchmark_counts[directory][seed] += 1
                response_counts[seed] += n_responses
        print(f"evaluated {directory}: {len(groups)} pairs", flush=True)
    if not np.all(counts):
        raise ValueError("No scoreable subject–benchmark pairs")
    results = {}
    for name, total in sums.items():
        by_seed = total / counts[:, None]
        alc = by_seed @ ALC_WEIGHTS
        results[name] = {
            "brier_by_seed": by_seed.tolist(),
            "brier_mean": by_seed.mean(axis=0).tolist(),
            "alc_by_seed": alc.tolist(),
            "alc_mean": float(alc.mean()),
            "alc_std": float(alc.std(ddof=1)) if seeds > 1 else None,
        }
    by_benchmark = {}
    for directory, method_sums in benchmark_sums.items():
        n = benchmark_counts[directory]
        by_benchmark[directory] = {
            "pairs_per_seed": n.tolist(),
            "alc_mean": {name: float(((total / n[:, None]) @ ALC_WEIGHTS).mean())
                         for name, total in method_sums.items()},
            "brier_mean": {name: (total / n[:, None]).mean(axis=0).tolist()
                           for name, total in method_sums.items()},
        }
    return {
        "source_revision": manifest["revision"],
        "split_file": str(split_file), "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "role": role, "budgets": list(BUDGETS), "alc_weights": ALC_WEIGHTS.tolist(),
        "seeds": seeds, "max_evaluation_items_per_pair": max_eval_items,
        "max_pairs_per_benchmark": max_pairs_per_benchmark,
        "prior_mixes": list(mixes),
        "benchmarks_used": used_dirs, "pairs_per_seed": counts.tolist(),
        "evaluation_responses_per_seed": response_counts.tolist(),
        "public_prior_file": str(public_prior_file) if public_prior_file else None,
        "eb_table_file": str(eb_table_file) if eb_table_file else None,
        "public_prior_name_hits_per_seed": prior_name_hits.tolist() if public_prior else None,
        "by_benchmark": by_benchmark,
        "weight_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in weight_files.items()},
        "weight_feature_modes": weight_modes,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "test"), required=True)
    parser.add_argument("--weights", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--public-prior", type=Path)
    parser.add_argument("--eb-table", type=Path)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--max-evaluation-items", type=int, default=64)
    parser.add_argument("--max-pairs-per-benchmark", type=int, default=0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--prior-mixes", type=float, nargs="+", default=[1.0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.max_evaluation_items < 1 or args.max_pairs_per_benchmark < 0:
        parser.error("seeds and evaluation items must be positive; max pairs must be nonnegative")
    if any(alpha <= 0 for alpha in args.alphas):
        parser.error("all correction strengths must be positive")
    if any(not 0 <= mix <= 1 for mix in args.prior_mixes):
        parser.error("prior mixes must be in [0,1]")
    try:
        weight_files = dict(entry.split("=", 1) for entry in args.weights)
    except ValueError:
        parser.error("each --weights argument must be NAME=PATH")
    if len(weight_files) != len(args.weights):
        parser.error("weight names must be unique")
    result = evaluate(args.data_root, args.split_file, args.role,
                      {name: Path(path) for name, path in weight_files.items()},
                      seeds=args.seeds, max_eval_items=args.max_evaluation_items,
                      max_pairs_per_benchmark=args.max_pairs_per_benchmark,
                      alphas=tuple(args.alphas), public_prior_file=args.public_prior,
                      mixes=tuple(args.prior_mixes), eb_table_file=args.eb_table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: values["alc_mean"] for name, values in result["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
