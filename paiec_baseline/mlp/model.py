"""Offline MLP predictor for PAIEC.

The weights are fitted before submission and saved as ``weights.npz`` beside
this file.  This module needs only NumPy at inference time.  No benchmark ID is
encoded as a model feature: hidden benchmarks have new anonymous IDs.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re

import numpy as np


FEATURE_DIM = 2048
HIDDEN_DIMS = (128, 64)
PRIOR_STRENGTH = 4.0
_WEIGHTS_PATH = Path(__file__).with_name("weights.npz")
_SUBJECT_FIELDS = (
    "normalized_name", "provider", "release_date", "access_date", "harness",
    "harness_version", "reasoning_effort", "subject_features_extra",
)
_ITEM_FIELDS = ("item_content", "item_features", "interactors")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _string(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return "" if text in {"nan", "none", "<na>", "nat"} else text


def _add(vector: np.ndarray, namespace: str, token: str, weight: float = 1.0,
         *, partitioned: bool = False) -> None:
    digest = hashlib.blake2b(f"{namespace}\0{token}".encode("utf-8"), digest_size=8).digest()
    code = int.from_bytes(digest, "little")
    offset, width = 0, FEATURE_DIM
    if partitioned:
        if namespace.startswith("subject:"):
            offset, width = 0, 512
        elif namespace.startswith("item:item_content"):
            offset, width = 512, 1024
        elif namespace.startswith("item:item_features"):
            offset, width = 1536, 256
        else:
            offset, width = 1792, 256
    vector[offset + code % width] += weight if code & (1 << 63) else -weight


def encode(input: list[dict], *, include_item: bool = True,
           partitioned: bool = False) -> np.ndarray:
    """Map the public subject/item attributes to a stable fixed-width vector.

    Hashing avoids fitting a vocabulary from hidden data.  The anonymous
    ``benchmark_id`` is deliberately excluded; it is used only to select
    same-group revealed labels in ``predict``.
    """
    if len(input) != 2 or not all(isinstance(part, dict) for part in input):
        raise ValueError("input must be [subject, item]")
    subject, item = input
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)

    def add(vector, namespace, token, weight=1.0):
        _add(vector, namespace, token, weight, partitioned=partitioned)

    for field in _SUBJECT_FIELDS:
        value = _string(subject.get(field, ""))
        add(vector, f"subject:{field}:exact", value or "<missing>")
        if value:
            tokens = _TOKEN_RE.findall(value)[:32]
            scale = 0.5 / math.sqrt(len(tokens))
            for token in tokens:
                add(vector, f"subject:{field}:word", token, scale)

    for field in (_ITEM_FIELDS if include_item else ()):
        value = _string(item.get(field, ""))
        if not value:
            add(vector, f"item:{field}", "<missing>")
            continue
        snippet = value if len(value) <= 20_000 else value[:16_000] + value[-4_000:]
        tokens = _TOKEN_RE.findall(snippet)
        # The front and end of a long task often contain the instructions and
        # the question.  Bound work for large code/agent benchmark items.
        selected = tokens if len(tokens) <= 640 else tokens[:512] + tokens[-128:]
        scale = 1.0 / math.sqrt(max(len(selected), 1))
        for token in selected:
            add(vector, f"item:{field}:word", token, scale)
        for left, right in zip(selected, selected[1:]):
            add(vector, f"item:{field}:bigram", left + "\0" + right, 0.35 * scale)
        add(vector, f"item:{field}:length", str(min(len(value) // 200, 40)), 0.5)
        if field == "item_content":
            # Word tokenization alone treats a Chinese sentence as one token.
            compact = re.sub(r"\s+", "", snippet)
            compact = compact if len(compact) <= 600 else compact[:480] + compact[-120:]
            han = re.findall(r"[\u3400-\u9fff]{2,}", compact)
            grams = [span[i:i + 2] for span in han for i in range(len(span) - 1)]
            if grams:
                han_scale = 0.5 / math.sqrt(len(grams))
                for gram in grams:
                    add(vector, "item:item_content:han2", gram, han_scale)
    return vector


def encode_mode(input: list[dict], mode: str = "all") -> np.ndarray:
    if mode in {"semantic_block", "retained_block"}:
        from .semantic_variant import encode_input
        return encode_input(input, mode)
    if mode not in {"all", "subject", "block", "structured"}:
        raise ValueError(f"Unknown feature mode: {mode}")
    if mode == "structured":
        from .structured_text import extract_text
        subject, item = input
        input = [subject, {**item, "item_content": extract_text(item.get("item_content", ""))}]
    return encode(input, include_item=mode != "subject", partitioned=mode == "block")


@lru_cache(maxsize=1)
def _load_weights() -> tuple[np.ndarray, ...]:
    if not _WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing fitted MLP weights: {_WEIGHTS_PATH}. Train and package weights.npz first."
        )
    expected = (
        ("W1", (FEATURE_DIM, HIDDEN_DIMS[0])), ("b1", (HIDDEN_DIMS[0],)),
        ("W2", (HIDDEN_DIMS[0], HIDDEN_DIMS[1])), ("b2", (HIDDEN_DIMS[1],)),
        ("W3", (HIDDEN_DIMS[1], 1)), ("b3", (1,)),
    )
    with np.load(_WEIGHTS_PATH, allow_pickle=False) as artifact:
        weights = []
        for name, shape in expected:
            if name not in artifact or artifact[name].shape != shape:
                raise ValueError(f"weights.npz has no valid {name} array of shape {shape}")
            array = np.asarray(artifact[name], dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"weights.npz contains nonfinite {name} values")
            weights.append(array)
    return tuple(weights)


def forward(features: np.ndarray, weights: tuple[np.ndarray, ...] | None = None) -> float:
    """Return the pretrained MLP's calibrated base probability for one input."""
    params = _load_weights() if weights is None else weights
    x = np.asarray(features, dtype=np.float32)
    if x.shape != (FEATURE_DIM,):
        raise ValueError(f"features must have shape ({FEATURE_DIM},)")
    if len(params) == 2:
        return float(forward_batch(x[None, :], params)[0])
    w1, b1, w2, b2, w3, b3 = params
    h1 = np.maximum(x @ w1 + b1, 0.0)
    h2 = np.maximum(h1 @ w2 + b2, 0.0)
    logit = float((h2 @ w3 + b3).item())
    if not math.isfinite(logit):
        raise ValueError("MLP produced a nonfinite logit")
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit)))))


def forward_batch(features: np.ndarray, weights: tuple[np.ndarray, ...] | None = None) -> np.ndarray:
    """Return the same MLP probabilities for a two-dimensional feature batch."""
    params = _load_weights() if weights is None else weights
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must have shape (batch, {FEATURE_DIM})")
    if len(params) == 2:
        w, b = params
        logits = (x @ w + b).reshape(-1)
    else:
        w1, b1, w2, b2, w3, b3 = params
        h1 = np.maximum(x @ w1 + b1, 0.0)
        h2 = np.maximum(h1 @ w2 + b2, 0.0)
        logits = (h2 @ w3 + b3).reshape(-1)
    if not np.isfinite(logits).all():
        raise ValueError("MLP produced a nonfinite logit")
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(np.float32)


def predict(input: list[dict], labeled: list | None = None) -> float:
    """Predict P(success), correcting the MLP prior with revealed group labels.

    For matching subject–benchmark observations, the correction is the mean
    residual ``label - MLP(observed item)`` shrunk by ``PRIOR_STRENGTH``.
    Every call is independent and uses only the supplied budget-limited labels.
    """
    subject, item = input
    prior = forward(encode(input))
    benchmark = _string(item.get("benchmark_id", ""))
    if not labeled:
        return prior
    if not benchmark:
        raise ValueError("benchmark_id is required when acquired labels are supplied")
    residual = 0.0
    count = 0
    for entry in labeled:
        (observed_subject, observed_item), label = entry
        if observed_subject != subject or _string(observed_item.get("benchmark_id")) != benchmark:
            continue
        if label not in (0, 1):
            raise ValueError("acquired response must be binary (0 or 1)")
        residual += int(label) - forward(encode([observed_subject, observed_item]))
        count += 1
    result = prior + residual / (count + PRIOR_STRENGTH)
    return float(min(1.0, max(0.0, result)))
