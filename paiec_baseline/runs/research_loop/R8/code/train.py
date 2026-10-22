"""Fit the offline MLP on an already prepared public PAIEC payload.

This script never downloads data. It requires PyTorch only for local training;
the exported ``weights.npz`` is consumed by the NumPy-only ``model.py``.

    python -m mlp.train --data-root payload/data --out mlp/weights.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import FEATURE_DIM, HIDDEN_DIMS, _ITEM_FIELDS, _SUBJECT_FIELDS, encode_mode, forward_batch


def _sample_scoreable_pairs(responses: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Allocate the same row budget across scoreable pairs before row sampling.

    This experimental sampler avoids making a one-response sampled pair equal
    in training weight to a well-covered pair. The default sampler is retained.
    """
    counts = responses.groupby("subject_id")["item_id"].nunique()
    eligible = counts.index[counts >= 80].to_numpy()
    if not len(eligible):
        return responses.iloc[:0].copy()
    rng = np.random.default_rng(seed)
    n_pairs = min(len(eligible), max(1, max_rows // 8))
    chosen = rng.choice(eligible, size=n_pairs, replace=False)
    per_pair, extra = divmod(max_rows, n_pairs)
    groups = responses.groupby("subject_id", sort=False)
    selections = []
    for index, subject_id in enumerate(chosen):
        group = groups.get_group(subject_id)
        count = min(len(group), per_pair + int(index < extra))
        selections.append(group.sample(n=count, random_state=int(rng.integers(2**31))))
    return pd.concat(selections, ignore_index=True)


def _load_examples(data_root: Path, directories: list[str], max_rows: int, seed: int,
                   feature_mode: str = "all", sampling: str = "response"):
    features = []
    labels = []
    groups = []
    for directory in directories:
        table_dir = data_root / directory
        response_path = table_dir / "response.parquet"
        if not response_path.is_file():
            continue
        responses = pd.read_parquet(response_path, columns=[
            "subject_id", "item_id", "response", "interactors"
        ])
        responses = responses[responses["response"].isin((0, 1))].dropna(
            subset=["subject_id", "item_id"]
        )
        if responses.empty:
            continue
        if sampling == "scoreable_pair_balanced":
            responses = _sample_scoreable_pairs(responses, max_rows, seed)
            if responses.empty:
                continue
        elif len(responses) > max_rows:
            responses = responses.sample(n=max_rows, random_state=seed)
        subjects = pd.read_parquet(table_dir / "subjects.parquet", columns=[
            "subject_id", *_SUBJECT_FIELDS
        ]).drop_duplicates("subject_id")
        items = pd.read_parquet(table_dir / "items.parquet", columns=[
            "item_id", "content", "item_features"
        ]).drop_duplicates("item_id")
        rows = responses.merge(subjects, on="subject_id", how="inner", validate="many_to_one")
        rows = rows.merge(items, on="item_id", how="inner", validate="many_to_one")
        if feature_mode in {"semantic_block", "retained_block"}:
            from .semantic_variant import prepare
            prepare(rows["content"].tolist())
        for row in rows.itertuples(index=False):
            subject = {field: getattr(row, field) for field in _SUBJECT_FIELDS}
            item = {
                "item_content": row.content,
                "item_features": row.item_features,
                "interactors": row.interactors,
            }
            features.append(encode_mode([subject, item], feature_mode))
            labels.append(float(row.response))
            groups.append(f"{directory}\0{row.subject_id}")
    if not features:
        raise ValueError("No joined binary response rows found in selected benchmarks")
    return np.stack(features), np.asarray(labels, np.float32), np.asarray(groups)


def _pair_weights(groups: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float32)
    return (weights / weights.mean()).astype(np.float32)


def _pair_brier(predictions: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    errors = (predictions - labels) ** 2
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    pair_sum = np.bincount(inverse, weights=errors)
    return float(np.mean(pair_sum / counts))


def train(data_root: Path, out: Path, *, seed: int = 2026, max_rows: int = 400,
          val_fraction: float = 0.2, epochs: int = 30, batch_size: int = 256,
          split_file: Path | None = None, learning_rate: float = 1e-3,
          weight_decay: float = 1e-4, feature_mode: str = "all",
          internal_validation: bool = False,
          fixed_epochs: int | None = None,
          sampling: str = "response", network_kind: str = "mlp") -> dict:
    if feature_mode not in {"all", "subject", "block", "structured", "semantic_block", "retained_block"}:
        raise ValueError("unsupported feature_mode")
    if network_kind not in {"mlp", "linear"}:
        raise ValueError("unsupported network_kind")
    if sampling not in {"response", "scoreable_pair_balanced"}:
        raise ValueError("unsupported sampling mode")
    if fixed_epochs is not None and (fixed_epochs < 1 or split_file is None or internal_validation):
        raise ValueError("fixed_epochs requires --split-file, a positive count, and no internal fold")
    manifest = json.loads((data_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    directories = list(manifest["eligible_dirs"])
    if len(directories) < 2:
        raise ValueError("Need at least two benchmark directories for cold-start validation")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if split_file is not None:
        split = json.loads(split_file.read_text(encoding="utf-8"))
        if split["source_revision"] != manifest["revision"]:
            raise ValueError("split source revision does not match the data manifest")
        train_dirs = list(split["train_benchmarks"])
        val_dirs = list(split["development_benchmarks"])
        test_dirs = set(split["test_benchmarks"])
        if not set(train_dirs + val_dirs + list(test_dirs)) == set(directories):
            raise ValueError("split does not cover the manifest benchmark directories")
        if set(train_dirs) & (set(val_dirs) | test_dirs) or set(val_dirs) & test_dirs:
            raise ValueError("train, development, and test benchmarks must be disjoint")
        development_dirs = list(val_dirs)
        if internal_validation:
            ranked = sorted(
                (d for d in train_dirs if split["scoreable_pair_counts"].get(d, 0)),
                key=lambda d: (-split["scoreable_pair_counts"][d], d),
            )
            if len(ranked) < 3:
                raise ValueError("Need at least three scoreable training benchmarks")
            # Keep the largest training benchmark in fit; include the next
            # largest in early stopping so this fold has meaningful pair count.
            remainder = ranked[2:]
            np.random.default_rng(seed).shuffle(remainder)
            n_internal = max(2, round(len(ranked) * 0.15))
            val_dirs = [ranked[1], *remainder[:n_internal - 1]]
            train_dirs = [d for d in train_dirs if d not in val_dirs]
        if fixed_epochs is not None:
            train_dirs = directories
            val_dirs = []
    else:
        if internal_validation:
            raise ValueError("--internal-validation requires --split-file")
        rng = np.random.default_rng(seed)
        rng.shuffle(directories)
        n_val = max(1, min(len(directories) - 1, round(len(directories) * val_fraction)))
        val_dirs, train_dirs = directories[:n_val], directories[n_val:]
        development_dirs = []
    x_train, y_train, g_train = _load_examples(data_root, train_dirs, max_rows, seed, feature_mode, sampling)
    if fixed_epochs is None:
        x_val, y_val, g_val = _load_examples(data_root, val_dirs, max_rows, seed + 1, feature_mode, sampling)
    else:
        x_val = np.empty((0, FEATURE_DIM), np.float32)
        y_val = np.empty(0, np.float32)
        g_val = np.empty(0, dtype=str)
    weights = _pair_weights(g_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = (nn.Sequential(
        nn.Linear(FEATURE_DIM, HIDDEN_DIMS[0]), nn.ReLU(),
        nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]), nn.ReLU(),
        nn.Linear(HIDDEN_DIMS[1], 1),
    ) if network_kind == "mlp" else nn.Linear(FEATURE_DIM, 1)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train)
    w_tensor = torch.from_numpy(weights)
    val_tensor = torch.from_numpy(x_val).to(device)
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    validation_history = []
    for epoch in range(1, (fixed_epochs if fixed_epochs is not None else epochs) + 1):
        net.train()
        order = torch.randperm(len(x_tensor))
        for indices in order.split(batch_size):
            features = x_tensor[indices].to(device)
            labels = y_tensor[indices].to(device)
            row_weights = w_tensor[indices].to(device)
            predictions = torch.sigmoid(net(features).flatten())
            loss = (row_weights * (predictions - labels).square()).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if fixed_epochs is not None:
            print(f"epoch={epoch} fixed_refit_complete", flush=True)
            continue
        net.eval()
        with torch.no_grad():
            val_pred = np.concatenate([
                torch.sigmoid(net(part).flatten()).cpu().numpy()
                for part in val_tensor.split(batch_size)
            ])
        score = _pair_brier(val_pred, y_val, g_val)
        validation_history.append({"epoch": epoch, "pair_brier": score})
        print(f"epoch={epoch} validation_pair_brier={score:.6f}", flush=True)
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    if fixed_epochs is not None:
        best_epoch = fixed_epochs
        best_score = None
        best_state = {name: tensor.detach().cpu().clone()
                      for name, tensor in net.state_dict().items()}
    assert best_state is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "W1": best_state["0.weight"].numpy().T.copy(),
        "b1": best_state["0.bias"].numpy(),
        "W2": best_state["2.weight"].numpy().T.copy(),
        "b2": best_state["2.bias"].numpy(),
        "W3": best_state["4.weight"].numpy().T.copy(),
        "b3": best_state["4.bias"].numpy(),
    } if network_kind == "mlp" else {
        "W": best_state["weight"].numpy().T.copy(),
        "b": best_state["bias"].numpy(),
    }
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        reference = torch.sigmoid(net(x_tensor[:32].to(device)).flatten()).cpu().numpy()
    exported = forward_batch(x_train[:32], tuple(arrays.values()))
    export_error = float(np.max(np.abs(reference - exported)))
    if export_error > 2e-6:
        raise ValueError(f"Training/export mismatch: {export_error}")
    np.savez_compressed(out, **arrays)
    report = {
        "source_revision": manifest.get("revision"), "seed": seed,
        "train_benchmarks": train_dirs, "validation_benchmarks": val_dirs,
        "development_benchmarks_untouched": development_dirs if internal_validation else [],
        "early_stopping_pool": ("none_fixed_refit" if fixed_epochs is not None else
                                "training" if internal_validation else "development_or_random"),
        "fixed_epochs": fixed_epochs,
        "train_rows": len(y_train), "validation_rows": len(y_val),
        "train_pairs": len(np.unique(g_train)), "validation_pairs": len(np.unique(g_val)),
        "max_rows_per_benchmark": max_rows, "sampling": sampling, "best_epoch": best_epoch,
        "validation_pair_brier_at_budget_0": best_score,
        "validation_history": validation_history,
        "split_file": str(split_file) if split_file else None,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "feature_mode": feature_mode,
        "network_kind": network_kind, "batch_size": batch_size,
        "export_max_abs_error": export_error,
        "weight_file": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("payload/data"))
    parser.add_argument("--out", type=Path, default=Path("mlp/weights.npz"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-rows-per-benchmark", type=int, default=400)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-mode", choices=("all", "subject", "block", "structured", "semantic_block", "retained_block"), default="all")
    parser.add_argument("--network-kind", choices=("mlp", "linear"), default="mlp")
    parser.add_argument("--sampling", choices=("response", "scoreable_pair_balanced"),
                        default="response")
    parser.add_argument("--internal-validation", action="store_true",
                        help="select the early-stopping fold within training benchmarks")
    parser.add_argument("--fixed-epochs", type=int,
                        help="refit on all public benchmarks for exactly this many epochs")
    args = parser.parse_args()
    if args.max_rows_per_benchmark < 1 or args.batch_size < 1 or args.epochs < 1:
        parser.error("row limit, batch size, and epochs must be positive")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be in (0,1)")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    print(json.dumps(train(args.data_root, args.out, seed=args.seed,
                           max_rows=args.max_rows_per_benchmark,
                           val_fraction=args.val_fraction,
                           epochs=args.epochs, batch_size=args.batch_size,
                           split_file=args.split_file,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           feature_mode=args.feature_mode,
                           internal_validation=args.internal_validation,
                           fixed_epochs=args.fixed_epochs, sampling=args.sampling,
                           network_kind=args.network_kind), indent=2))


if __name__ == "__main__":
    main()
