"""Research adapter: local item slopes around a smoothed pair prevalence.

No fitted global prediction or benchmark identity enters the kernel. All slope
information comes from the labels supplied to this call. R3 controls select
the regularizer using outer-training episodes only.
"""
from __future__ import annotations
import numpy as np
from .model import encode, _string


def item_vector(item: dict) -> np.ndarray:
    vector = encode([{}, item]) - encode([{}, {}], include_item=False)
    return vector.astype(np.float64) / max(float(np.linalg.norm(vector)), 1e-12)


def kernel_predict(support_kernel: np.ndarray, query_kernel: np.ndarray,
                   labels: np.ndarray, ridge: float | None, alpha: float = 2.) -> np.ndarray:
    """Labels are ONLY the current support prefix; kernels carry no labels."""
    labels = np.asarray(labels, dtype=np.float64)
    n = len(labels)
    local = (alpha * .5 + labels.sum()) / (alpha + n)
    if ridge is None or n < 2:
        return np.full(len(query_kernel), local)
    if ridge <= 0 or alpha <= 0:
        raise ValueError("regularizers must be positive")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    kernel = np.asarray(support_kernel, np.float64)[:n, :n]
    cross = np.asarray(query_kernel, np.float64)[:, :n]
    centered_kernel = kernel - kernel.mean(0)[None, :] - kernel.mean(1)[:, None] + kernel.mean()
    centered_cross = cross - cross.mean(1)[:, None] - kernel.mean(0)[None, :] + kernel.mean()
    coefficients = np.linalg.solve(centered_kernel + ridge*np.eye(n), labels - labels.mean())
    return np.clip(local + centered_cross @ coefficients, 0, 1)


def predict(input: list[dict], labeled=None, *, ridge: float | None = 1.) -> float:
    subject, item = input
    matched = []
    benchmark = _string(item.get("benchmark_id"))
    for (other_subject, other_item), label in (labeled or []):
        if subject == other_subject and benchmark and _string(other_item.get("benchmark_id")) == benchmark:
            if label not in (0, 1):
                raise ValueError("labels must be binary")
            matched.append((other_item, label))
    if not matched:
        return .5
    support = np.stack([item_vector(x) for x, y in matched])
    query = item_vector(item)[None, :]
    return float(kernel_predict(support @ support.T, query @ support.T,
                                np.array([y for x, y in matched]), ridge)[0])
