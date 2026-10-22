"""
prepare_data.py — Build the offline databank payload for the PAEC submission.

The competition sandbox has no HuggingFace hub access at runtime, so the
official public training corpus and item embeddings must be shipped: either
bundled inside the submission zip (payload copied into data/ and
embeddings/ at its root) or published as a public HF repo that the platform
pre-downloads via models.txt.

Run on a development machine with internet access and the dependencies from
ble/requirements.txt installed. A new build resolves the current main branches
of measurement-db and measurement-db-embed; an interrupted build can resume
those recorded revisions with --resume. It does not generate item embeddings.

    python tools/prepare_data.py --out payload                # build payload/
    python tools/prepare_data.py --out payload --no-embeddings
    python tools/prepare_data.py --out payload --no-embeddings --no-traces
    python tools/prepare_data.py --out payload --no-embeddings --no-traces --resume

The payload layout:
    data/<benchmark_dir>/{benchmarks,subjects,items,response[,traces]}.parquet
    data/corpus_manifest.json
    embeddings/manifest.json + embeddings/items/<benchmark_dir>.parquet
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

# Canonical repository behind the competition site's public-training link.
PUBLIC_DATA_REPO = "aims-foundations/measurement-db"
PUBLIC_EMBEDDINGS_REPO = "aims-foundations/measurement-db-embed"
# These published vectors match the original text-embedding-3-small index.
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIMENSIONS = 1536
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORPUS_SCOPE = "paec-public-training"

_TABLES = ("benchmarks", "subjects", "items", "response", "traces")


def _fetch(filename, revision):
    return hf_hub_download(
        PUBLIC_DATA_REPO,
        filename,
        repo_type="dataset",
        revision=revision,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "payload"))
    ap.add_argument(
        "--no-embeddings",
        action="store_true",
        help="skip measurement-db-embed; prepare only the seven keyword/table tools",
    )
    ap.add_argument(
        "--no-traces",
        action="store_true",
        help="skip large traces.parquet tables; sufficient for mean/MLP experiments",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted build using its recorded Hugging Face revisions",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="concurrent metadata and file requests (default: 16)",
    )
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be at least 1")
    out = os.path.abspath(args.out)
    state_path = os.path.join(out, ".prepare_state.json")
    previous = None
    nonempty = False
    if os.path.exists(out):
        with os.scandir(out) as entries:
            nonempty = next(entries, None) is not None
    if nonempty:
        if not args.resume or not os.path.isfile(state_path):
            raise RuntimeError("--out must be new/empty, or an interrupted build with --resume")
        with open(state_path, encoding="utf-8") as handle:
            previous = json.load(handle)
        if (
            previous.get("repo") != PUBLIC_DATA_REPO
            or previous.get("no_embeddings") != args.no_embeddings
            or previous.get("no_traces") != args.no_traces
        ):
            raise RuntimeError("--resume options do not match the interrupted build")

    # A new build resolves main; an interrupted build resumes its exact snapshot.
    print(f"Listing public training release {PUBLIC_DATA_REPO} ...")
    api = HfApi()
    data_info = api.repo_info(
        PUBLIC_DATA_REPO, repo_type="dataset",
        revision=previous["data_revision"] if previous else "main",
    )
    data_revision = data_info.sha
    files = {entry.rfilename for entry in data_info.siblings}
    embedding_info = None
    if not args.no_embeddings:
        embedding_info = api.repo_info(
            PUBLIC_EMBEDDINGS_REPO, repo_type="dataset",
            revision=previous["embedding_revision"] if previous else "main",
        )
    if previous is None:
        os.makedirs(out, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump({
                "repo": PUBLIC_DATA_REPO,
                "data_revision": data_revision,
                "embedding_revision": embedding_info.sha if embedding_info else None,
                "no_embeddings": args.no_embeddings,
                "no_traces": args.no_traces,
            }, handle, indent=2)
    data_out = os.path.join(out, "data")
    os.makedirs(data_out, exist_ok=True)
    dirs = sorted(
        {f.rsplit("/", 1)[0] for f in files if f.endswith("/benchmarks.parquet")}
    )
    print(f"  {len(dirs)} benchmark dirs")

    # Competition-eligible tables have binary, item-level responses.
    def _elig(d):
        benchmark = pd.read_parquet(
            _fetch(f"{d}/benchmarks.parquet", data_revision)
        ).iloc[0]
        return d, bool(
            benchmark["response_type"] == "binary"
            and benchmark["granularity"] == "item"
        )

    eligible = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for d, ok in ex.map(_elig, dirs):
            if ok:
                eligible.append(d)
    eligible = sorted(eligible)
    if not eligible:
        raise ValueError("the selected revision contains no eligible benchmarks")
    print(f"  {len(eligible)} eligible (binary + item granularity)")

    # Download and copy every available public table per eligible directory.
    manifest_files = []

    def _one(d):
        os.makedirs(os.path.join(data_out, d), exist_ok=True)
        got = []
        for t in _TABLES:
            if args.no_traces and t == "traces":
                continue
            source_name = f"{d}/{t}.parquet"
            # The current release uses both spellings. Keep one local schema
            # so the retrieval code can always load response.parquet.
            if t == "response" and source_name not in files:
                source_name = f"{d}/responses.parquet"
            if source_name not in files:
                continue
            src = _fetch(source_name, data_revision)
            dst = os.path.join(data_out, d, f"{t}.parquet")
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                partial = dst + ".part"
                shutil.copyfile(src, partial)
                os.replace(partial, dst)
            got.append(f"{d}/{t}.parquet")
        return got

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, got in enumerate(ex.map(_one, eligible), 1):
            manifest_files.extend(got)
            if i % 25 == 0 or i == len(eligible):
                print(f"  {i}/{len(eligible)} dirs ({time.time() - t0:.0f}s)")

    with open(os.path.join(data_out, "corpus_manifest.json"), "w") as f:
        json.dump(
            {
                "db_repo": PUBLIC_DATA_REPO,
                "corpus_scope": _CORPUS_SCOPE,
                "revision": data_revision,
                "generated_unix": int(time.time()),
                "eligible_dirs": eligible,
                "files": sorted(manifest_files),
            },
            f,
            indent=2,
        )

    # The embedding repo stores <benchmark>.parquet at its root. Package only
    # eligible benchmarks, checking IDs and content against this data snapshot.
    if embedding_info is not None:
        embedding_files = {entry.rfilename for entry in embedding_info.siblings}
        selected = [d for d in eligible if f"{d}.parquet" in embedding_files]
        if not selected:
            raise ValueError("the embeddings contain no eligible public benchmarks")
        emb_out = os.path.join(out, "embeddings", "items")
        os.makedirs(emb_out, exist_ok=True)
        benchmarks = {}
        for directory in selected:
            src = hf_hub_download(
                PUBLIC_EMBEDDINGS_REPO,
                f"{directory}.parquet",
                repo_type="dataset",
                revision=embedding_info.sha,
            )
            vectors = pd.read_parquet(
                src, columns=["item_id", "content_sha1", "embedding"]
            )
            items = pd.read_parquet(
                os.path.join(data_out, directory, "items.parquet"),
                columns=["item_id", "content"],
            )
            # This hash verifies the text embedded offline, never a test label.
            content_hashes = {
                str(row.item_id): hashlib.sha1(
                    str(row.content or "").encode("utf-8")
                ).hexdigest()
                for row in items.itertuples()
            }
            if any(
                content_hashes.get(str(row.item_id)) != row.content_sha1
                for row in vectors.itertuples()
            ):
                raise ValueError(f"{directory}: embeddings do not match current items")
            matrix = np.stack(vectors["embedding"].to_numpy())
            if (
                matrix.shape != (len(vectors), _EMBEDDING_DIMENSIONS)
                or not np.isfinite(matrix).all()
            ):
                raise ValueError(f"{directory}: invalid embedding vectors")
            shutil.copyfile(src, os.path.join(emb_out, f"{directory}.parquet"))
            benchmarks[directory] = {"n_items": len(vectors)}
        embedding_manifest = {
            "model": _EMBEDDING_MODEL,
            "dimensions": _EMBEDDING_DIMENSIONS,
            "benchmarks": benchmarks,
            "corpus_scope": _CORPUS_SCOPE,
            "data_revision": data_revision,
            "embeddings_repo": PUBLIC_EMBEDDINGS_REPO,
            "revision": embedding_info.sha,
        }
        with open(os.path.join(out, "embeddings", "manifest.json"), "w") as handle:
            json.dump(embedding_manifest, handle, indent=2)
        print(
            f"  embeddings: {len(selected)}/{len(eligible)} benchmark vector files copied"
        )
    else:
        print("  keyword/table retrieval only (no embeddings requested)")

    def _du(path):
        total = 0
        for cur, _, fs in os.walk(path):
            total += sum(os.path.getsize(os.path.join(cur, x)) for x in fs)
        return total / 1e9

    print(f"\nPayload ready at {out}")
    print(f"  data:       {_du(data_out):.2f} GB")
    emb_dir = os.path.join(out, "embeddings")
    if os.path.isdir(emb_dir):
        print(f"  embeddings: {_du(emb_dir):.2f} GB")
    os.remove(state_path)


if __name__ == "__main__":
    main()
