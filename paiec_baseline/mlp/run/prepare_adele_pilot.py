"""Prepare offline ADeLe annotation inputs from frozen PAIEC training items.

This command has no API client, model loading, training, or submission path.
The upstream prompt builders are imported unchanged from the pinned author
checkout. Only source-cache metadata and public item_id/content are decoded.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from ..semantic_features import SemanticFeatures

UPSTREAM_COMMIT = "816321f327dfe66910b72b514156aaa296b15f00"
DATA_REVISION = "54074453da7b9911be0f48e4c560be547bc84d21"
DEMANDS = ("AS", "MCr", "QLl", "KNf")
PENDING_MODEL = "MODEL_SNAPSHOT_PENDING__DRY_RUN_ONLY"
BASELINE_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_KEYS = {
    "answer", "answers", "gold", "gold_answer", "groundtruth", "ground_truth",
    "correct", "correct_answer", "correctness", "label", "labels", "target",
    "targets", "solution", "solutions", "reference_answer", "reference_solution",
    "response", "responses", "model_response", "hidden", "hidden_fields",
    "private", "private_metadata", "score", "scores", "expected_answer",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def load_author_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prohibited_content_key(value):
    """Reject whole structured items; never rewrite their public task text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if isinstance(value, dict):
        if str(value.get("role", "")).lower() == "assistant":
            return "assistant_role"
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized in FORBIDDEN_KEYS or normalized.startswith(("hidden_", "private_")):
                return normalized
            found = prohibited_content_key(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = prohibited_content_key(child)
            if found:
                return found
    return None


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(args):
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output}")
    upstream = args.upstream.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected upstream commit: {commit}")
    prompt_path = upstream / "src/adele/annotation/prompts.py"
    catalog_path = upstream / "src/adele/rubrics/catalog.py"
    rubric_paths = [upstream / f"src/adele/rubrics/data_v1/{d}.txt" for d in DEMANDS]
    checked = [prompt_path, catalog_path, *rubric_paths]
    subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", "HEAD", "--",
                    *[str(p.relative_to(upstream)) for p in checked]], check=True,
                   stdout=subprocess.DEVNULL)
    prompts = load_author_module("_r53_author_prompts", prompt_path)
    catalog_module = load_author_module("_r53_author_catalog", catalog_path)
    catalog = catalog_module.RubricsCatalog()
    rubrics = {d: catalog[d] for d in DEMANDS}

    source_info_path = BASELINE_ROOT / "runs/research_loop/reproduction_v4/source_fold0.json"
    frozen_split_path = BASELINE_ROOT / "runs/research_loop/R25/fold0_selection.json"
    corpus_path = args.data_root / "corpus_manifest.json"
    info = json.loads(source_info_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_split_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if corpus["revision"] != DATA_REVISION:
        raise ValueError("Unexpected PAIEC revision")
    for field in ("train_benchmarks", "heldout_benchmarks"):
        if info[field] != frozen[field]:
            raise ValueError(f"Source cache split differs from R25: {field}")
    training = sorted(info["train_benchmarks"])
    if set(training) & set(info["heldout_benchmarks"]):
        raise ValueError("Training and heldout benchmark overlap")
    if not set(training).issubset(corpus["eligible_dirs"]):
        raise ValueError("Training benchmark absent from public corpus manifest")

    queues, sources, exclusions = {}, [], Counter()
    raw_candidates = 0
    for benchmark in training:
        cache_key = text_hash(f"{benchmark}|73|25|{info['cache_fingerprint']}")
        cache_path = args.cache_root / "source" / (cache_key + ".npz")
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        # NPZ arrays are lazy: this is the only member accessed. Never load
        # labels, counts, successes, support/query features or subject vectors.
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
        if len(metadata) > 25:
            raise ValueError("Source pair cap differs from frozen cap25")
        candidates = {}
        for episode_index, episode in enumerate(metadata):
            if episode["benchmark"] != benchmark or episode["seed"] != 73:
                raise ValueError("Unexpected source episode benchmark or seed")
            if set(episode["query_item_ids"]) & set(episode["support_item_ids"]):
                raise ValueError("Source support/query item overlap")
            for item_id in episode["query_item_ids"]:
                candidates.setdefault(str(item_id), episode_index)
        raw_candidates += len(candidates)
        source = {"benchmark": benchmark, "source_cache": source_record(cache_path),
                  "decoded_npz_members": ["metadata"], "episode_count": len(metadata),
                  "candidate_query_items": len(candidates)}
        if not candidates:
            sources.append(source)
            continue
        item_path = args.data_root / benchmark / "items.parquet"
        parquet = pq.ParquetFile(item_path)
        if not {"item_id", "content"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"Missing required public columns: {item_path}")
        contents = {}
        # Project only public fields; response.parquet is never opened.
        for batch in parquet.iter_batches(batch_size=256, columns=["item_id", "content"]):
            for row in batch.to_pylist():
                item_id = str(row["item_id"])
                if item_id in candidates and item_id not in contents:
                    contents[item_id] = row["content"]
            if len(contents) == len(candidates):
                break
        source.update({"items_parquet": source_record(item_path),
                       "parquet_column_names": parquet.schema_arrow.names,
                       "decoded_parquet_columns": ["item_id", "content"]})
        sources.append(source)
        queue = deque()
        for item_id, episode_index in candidates.items():
            if item_id not in contents:
                raise ValueError(f"Query item absent from public items: {benchmark}/{item_id}")
            content = contents[item_id]
            forbidden = prohibited_content_key(content)
            if forbidden:
                exclusions[f"structured_field:{forbidden}"] += 1
                continue
            canonical = SemanticFeatures.canonical(content)
            if not canonical.strip() or canonical == "<empty_structure>":
                exclusions["empty_canonical_text"] += 1
                continue
            queue.append({"benchmark": benchmark, "item_id": item_id,
                          "source_episode_index": episode_index, "seed": 73,
                          "source_cache_sha256": source["source_cache"]["sha256"],
                          "items_parquet_sha256": source["items_parquet"]["sha256"],
                          "public_content_sha256": text_hash("" if content is None else str(content)),
                          "canonical_text_sha256": text_hash(canonical),
                          "canonical_characters": len(canonical), "text": canonical})
        queues[benchmark] = queue

    selected, seen = [], set()
    while len(selected) < 128:
        progressed = False
        for benchmark in training:
            queue = queues.get(benchmark)
            while queue:
                item = queue.popleft()
                if item["canonical_text_sha256"] in seen:
                    exclusions["duplicate_canonical_text"] += 1
                    continue
                seen.add(item["canonical_text_sha256"])
                selected.append(item)
                progressed = True
                break
            if len(selected) == 128:
                break
        if not progressed:
            raise ValueError(f"Only {len(selected)} eligible distinct content items")

    output.mkdir(parents=True, exist_ok=True)
    request_stats, item_rows, request_ids = [], [], set()
    request_path = output / "requests.jsonl"
    with request_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, item in enumerate(selected):
            item_key = f"paiec-r53-{index:03d}-{item['canonical_text_sha256'][:16]}"
            item_row = {key: value for key, value in item.items() if key != "text"}
            item_row["custom_item_id"] = item_key
            item_rows.append(item_row)
            for demand in DEMANDS:
                rubric = rubrics[demand]
                prompt = prompts.build_annotation_prompt(rubric.full_name, rubric.content, item["text"])
                request = prompts.build_batch_request(item_key, demand, prompt,
                                                      model=PENDING_MODEL, max_completion_tokens=1000)
                if request["custom_id"] in request_ids:
                    raise ValueError("Duplicate request ID")
                request_ids.add(request["custom_id"])
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")
                request_stats.append({"custom_id": request["custom_id"], "demand": demand,
                                      "input_characters": len(prompt), "input_utf8_bytes": len(prompt.encode("utf-8")),
                                      "prompt_sha256": text_hash(prompt)})
    items_path, statistics_path = output / "items.json", output / "request_statistics.json"
    write_json(items_path, item_rows)
    write_json(statistics_path, request_stats)
    with request_path.open(encoding="utf-8") as handle:
        observed = [json.loads(line) for line in handle]
    assert len(observed) == 512 and len(selected) == len(seen) == 128
    assert Counter(row["custom_id"].rsplit("__", 1)[1] for row in observed) == Counter({d: 128 for d in DEMANDS})
    assert all(row["body"]["model"] == PENDING_MODEL and row["body"]["messages"][0]["role"] == "user"
               and len(row["body"]["messages"]) == 1 for row in observed)
    input_lengths = [row["input_characters"] for row in request_stats]
    manifest = {
        "status": "dry_run_inputs_prepared_annotation_pending", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_elapsed_seconds": time.perf_counter() - timer,
        "model_snapshot": None, "request_model_placeholder": PENDING_MODEL,
        "api_authorization": "not_granted_by_preparation", "api_calls": 0,
        "paid_usage": 0, "model_downloads": 0, "labels_read": False,
        "effectiveness_experiment": False, "data_revision": DATA_REVISION,
        "outer_fold": 0, "source_seed": 73, "source_pair_cap": 25,
        "cache_fingerprint": info["cache_fingerprint"], "requested_items": 128,
        "distinct_items": len(selected), "requests": len(observed), "demands": list(DEMANDS),
        "selection": "sorted training benchmarks round-robin; original cached query order; global canonical hash deduplication",
        "benchmark_counts": dict(sorted(Counter(x["benchmark"] for x in selected).items())),
        "candidate_query_items": raw_candidates, "exclusions": dict(exclusions),
        "input_character_statistics": {"total": sum(input_lengths), "min": min(input_lengths),
                                       "max": max(input_lengths), "mean": sum(input_lengths) / len(input_lengths)},
        "billing_note": "Character counts are not token counts, a price estimate, or measured API usage.",
        "canonicalization": "SemanticFeatures.canonical unchanged; extracted public content capped at 20000 characters, without MiniLM token window truncation",
        "upstream_commit": commit, "upstream_repository": "https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation",
        "author_sources": [source_record(p) for p in checked],
        "rubrics": {d: {"full_name": rubrics[d].full_name,
                        "content_sha256": text_hash(rubrics[d].content)} for d in DEMANDS},
        "input_sources": {"source_fold": source_record(source_info_path),
                          "r25_split": source_record(frozen_split_path),
                          "corpus_manifest": source_record(corpus_path), "benchmarks": sources},
        "implementation_sources": [source_record(Path(__file__)),
                                   source_record(BASELINE_ROOT / "mlp/semantic_features.py"),
                                   source_record(BASELINE_ROOT / "mlp/structured_text.py")],
        "artifacts": [source_record(p) for p in (request_path, items_path, statistics_path)],
        "structural_checks": {"128_distinct_text_hashes": True, "512_unique_requests": True,
                              "128_requests_per_demand": True, "only_fold0_training_benchmarks": True,
                              "no_response_parquet_opened": True, "npz_metadata_only": True,
                              "author_prompt_builders_unchanged": True, "pending_model_in_every_request": True},
        "next_condition": "Obtain an authorized annotator snapshot and budget, measure parsing/cost/latency on a bounded sample; then decide whether to freeze a PAIEC effectiveness protocol.",
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "distinct_items", "requests", "benchmark_counts",
                                                     "input_character_statistics", "preparation_elapsed_seconds")}, ensure_ascii=False))
    print(str(output / "manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("D:/nips-comp/paiec-dataset/reproduction_sources/adele/upstream"))
    parser.add_argument("--data-root", type=Path, default=Path("D:/paiec_payload/data"))
    parser.add_argument("--cache-root", type=Path, default=Path("D:/paiec_research_cache/reproduction_v4"))
    parser.add_argument("--output", type=Path, default=Path("D:/nips-comp/paiec-dataset/reproduction_v4/adele_pilot"))
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
