---
license: cc-by-sa-4.0
pretty_name: The AI Measurement Data Bank
extra_gated_prompt: "Please provide the following information so we can review your access request."
extra_gated_fields:
  Full name: text
  Organization or affiliation: text
  User category:
    type: select
    options:
      - Student
      - Academic researcher or faculty
      - Industry researcher or practitioner
      - Government or nonprofit researcher
      - Independent researcher
      - label: Other
        value: other
  Please briefly describe how you intend to use the Measurement Data Bank: text
configs:
- config_name: benchmarks
  default: true
  features:
  - name: benchmark_id
    dtype: string
  - name: name
    dtype: string
  - name: version
    dtype: string
  - name: license
    dtype: string
  - name: source_url
    dtype: string
  - name: description
    dtype: string
  - name: one_line_description
    dtype: string
  - name: modality
    list: string
  - name: domain
    list: string
  - name: multi_single_turn
    dtype: string
  - name: response_type
    dtype: string
  - name: response_scale
    dtype: string
  - name: categorical
    dtype: bool
  - name: paper_url
    dtype: string
  - name: release_date
    dtype: string
  - name: granularity
    dtype: string
  - name: release
    dtype: string
  - name: n_response_values
    dtype: int64
  - name: benchmark_features
    dtype: string
  - name: n_subjects
    dtype: int64
  - name: n_items
    dtype: int64
  - name: n_responses
    dtype: int64
  - name: max_trial
    dtype: int64
  - name: coverage
    dtype: float64
  - name: has_reference_answer
    dtype: bool
  data_files:
  - split: train
    path:
    - matharena/formatted_tables/benchmarks.parquet
    - mmdocrag/formatted_tables/benchmarks.parquet
    - multi_swebench/formatted_tables/benchmarks.parquet
    - real_webagents/formatted_tables/benchmarks.parquet
    - researchcodebench/formatted_tables/benchmarks.parquet
    - swe_rebench/formatted_tables/benchmarks.parquet
- config_name: items
  features:
  - name: item_id
    dtype: string
  - name: benchmark_id
    dtype: string
  - name: raw_item_id
    dtype: string
  - name: content
    dtype: string
  - name: asset_manifest
    dtype: string
  - name: grading_criterion
    dtype: string
  - name: verifier
    dtype: string
  - name: content_hash
    dtype: string
  - name: item_features
    dtype: string
  data_files:
  - split: matharena
    path: matharena/formatted_tables/items.parquet
  - split: mmdocrag
    path: mmdocrag/formatted_tables/items.parquet
  - split: multi_swebench
    path: multi_swebench/formatted_tables/items.parquet
  - split: real_webagents
    path: real_webagents/formatted_tables/items.parquet
  - split: researchcodebench
    path: researchcodebench/formatted_tables/items.parquet
  - split: swe_rebench
    path: swe_rebench/formatted_tables/items.parquet
- config_name: responses
  features:
  - name: subject_id
    dtype: string
  - name: item_id
    dtype: string
  - name: response_id
    dtype: string
  - name: benchmark_id
    dtype: string
  - name: trial
    dtype: int64
  - name: test_condition
    dtype: string
  - name: interactors
    dtype: string
  - name: response
    dtype: float64
  data_files:
  - split: matharena
    path: matharena/formatted_tables/responses.parquet
  - split: mmdocrag
    path: mmdocrag/formatted_tables/responses.parquet
  - split: multi_swebench
    path: multi_swebench/formatted_tables/responses.parquet
  - split: real_webagents
    path: real_webagents/formatted_tables/responses.parquet
  - split: researchcodebench
    path: researchcodebench/formatted_tables/responses.parquet
  - split: swe_rebench
    path: swe_rebench/formatted_tables/responses.parquet
- config_name: subjects
  features:
  - name: subject_id
    dtype: string
  - name: display_name
    dtype: string
  - name: normalized_name
    dtype: string
  - name: provider
    dtype: string
  - name: release_date
    dtype: string
  - name: access_date
    dtype: string
  - name: harness
    dtype: string
  - name: reasoning_effort
    dtype: string
  - name: harness_version
    dtype: string
  - name: subject_features_extra
    dtype: string
  data_files:
  - split: matharena
    path: matharena/formatted_tables/subjects.parquet
  - split: mmdocrag
    path: mmdocrag/formatted_tables/subjects.parquet
  - split: multi_swebench
    path: multi_swebench/formatted_tables/subjects.parquet
  - split: real_webagents
    path: real_webagents/formatted_tables/subjects.parquet
  - split: researchcodebench
    path: researchcodebench/formatted_tables/subjects.parquet
  - split: swe_rebench
    path: swe_rebench/formatted_tables/subjects.parquet
- config_name: traces
  features:
  - name: response_id
    dtype: string
  - name: subject_id
    dtype: string
  - name: item_id
    dtype: string
  - name: benchmark_id
    dtype: string
  - name: trial
    dtype: int64
  - name: test_condition
    dtype: string
  - name: interactors
    dtype: string
  - name: trace
    dtype: string
  data_files:
  - split: matharena
    path: matharena/formatted_tables/traces.parquet
  - split: mmdocrag
    path: mmdocrag/formatted_tables/traces.parquet
  - split: multi_swebench
    path: multi_swebench/formatted_tables/traces.parquet
  - split: real_webagents
    path: real_webagents/formatted_tables/traces.parquet
  - split: swe_rebench
    path: swe_rebench/formatted_tables/traces.parquet
---

## Source reproducibility holds (2026-10-04)

ImageNet-Hard (`imagenet_hard`), VisualMemory (`visual_memory`), and SWE-Together
(`swe_together`) are temporarily **withheld from the active public release**.
Their `metadata.yaml` files and `formatted_tables/benchmarks.parquet` catalog
records now agree on `release: withheld`. ImageNet-Hard and VisualMemory require
author-hosted image archives that currently return HTTP 404, with no verified
author-endorsed replacements. SWE-Together's live result file has changed, and
no immutable author snapshot matching the pinned version has been verified.

These holds concern source reproducibility and do not change the recorded
licenses. Existing measurement tables and raw snapshots remain as historical
archives. Filter catalog records to `release == "public"` for active releases.
The dataset-viewer configurations already exclude these three benchmarks.
Restore public release only after verified author-source recovery and strict
table reproduction, or a separately reviewed version update using immutable
author inputs. The [release-update record](release_holds_20261004.json) preserves
the reasons, previous catalog hashes, and source revision for this decision.

## Tabular-builder migration checkpoint

This development branch archives the data for **249 validated benchmark migrations** at [GitHub revision 2874959](https://github.com/aims-foundations/measurement-db/tree/287495967fc276275c3b466dfede2cc3534e830c). [Upload status](checkpoint_status.json) records whether the complete checkpoint has been verified.

Each added benchmark has `raw/` for captured upstream inputs, `formatted_tables/` for the validated Parquet tables, and its metadata and characterization. `data_manifest.json` records file sizes and SHA-256 hashes. [The checkpoint inventory](tabular_checkpoint.json) lists all 249 benchmarks and their table sizes. The six benchmarks inherited from the parent branch now also store their tables in `formatted_tables/`.

These records preserve the released observations, including missing grades and benchmark-specific scales; they do not represent fresh model evaluations. Original upstream licenses and the limitations documented in each benchmark's metadata continue to apply.


Data-Juicer 2.0 archives its derived timing tables and seven upstream protocol files. Full paper captures remain local; the builder downloads the pinned HTML, PDF and TeX source from arXiv. These records describe processing workloads, not LLM correctness.

# The AI Measurement Data Bank

Measurement Data Bank is a curated collection of standardized, item-level AI
evaluation results for measurement-science analysis.

## Schema and versioned downloads

These six benchmark datasets use [schema version 3](https://github.com/aims-foundations/measurement-db/blob/287495967fc276275c3b466dfede2cc3534e830c/parquet_schemas.yaml). The [GitHub repository](https://github.com/aims-foundations/measurement-db) contains the corresponding builders, source manifests, tests, and curation records.

Grading criteria are stored in `items.parquet`; traces link to observations through `response_id`. All tables are stored in each benchmark’s `formatted_tables/` directory; the response table is named `responses.parquet`. Item and dependent response IDs changed in this migration, while all 571,921 recorded observations and available traces were preserved. Download all tables from the same repository revision when joining them; earlier snapshots remain accessible through repository history.

Each benchmark also includes `characterization.yaml` alongside `metadata.yaml`. The [characterization schema](characterization_schema.yaml) requires reviewed table counts and logical content hashes, plus a nonempty `source_claims` section. Claims distinguish author-reported values from quantities independently derived from released data, and specify their source, location, scope, and any aggregate tolerance. The benchmark tests check every declared claim. See the [characterization guide](https://github.com/aims-foundations/measurement-db/blob/287495967fc276275c3b466dfede2cc3534e830c/characterization.md) for the fields and review procedure. These validation files do not change the measurements.

## Archived source data

The exact source snapshots used to build these datasets are preserved in each benchmark's `raw/` directory, with its `metadata.yaml` source description alongside it. The metadata files list upstream sources and their known revisions. Current builders fetch the upstream sources declared in metadata and verify their recorded sizes and content hashes; the captured archives remain available for review. Together, the six archives contain 1,191 source files (10.4 GB). Use the [matching builders and schema](https://github.com/aims-foundations/measurement-db/tree/287495967fc276275c3b466dfede2cc3534e830c) to reproduce this release. See the reference-refresh record below for the current table and identity changes.

Browse the sources for [MathArena](https://huggingface.co/datasets/aims-foundations/measurement-db/tree/main/matharena/raw), [MMDocRAG](https://huggingface.co/datasets/aims-foundations/measurement-db/tree/main/mmdocrag/raw), [Multi-SWE-bench](https://huggingface.co/datasets/aims-foundations/measurement-db/tree/main/multi_swebench/raw), [REAL](https://huggingface.co/datasets/aims-foundations/measurement-db/tree/main/real_webagents/raw), [ResearchCodeBench](https://huggingface.co/datasets/aims-foundations/measurement-db/tree/main/researchcodebench/raw), and [SWE-rebench](https://huggingface.co/datasets/aims-foundations/measurement-db/tree/main/swe_rebench/raw). [Restoration instructions](https://github.com/aims-foundations/measurement-db#restoring-archived-sources) show how to download a pinned snapshot and rebuild from cached inputs if upstream links change.

## License

To the extent that AIMS holds copyright or database rights, the original
curation contributions in Measurement Data Bank—including its selection,
organization, standardized schema, metadata, and normalization work—are
licensed under the [Creative Commons Attribution-ShareAlike 4.0 International
License](https://creativecommons.org/licenses/by-sa/4.0/).

Individual benchmarks and other third-party materials are not relicensed under
CC BY-SA 4.0. They retain their original upstream licenses and terms, as
identified in each benchmark's metadata. Those upstream terms govern the
corresponding material.

## Citation

If you use Measurement Data Bank, please cite:

```bibtex
@misc{measurementdb2026,
  title        = {The AI Measurement Data Bank},
  author       = {Truong, Nhi and Truong, Sang T. and Koyejo, Sanmi},
  year         = {2026},
  howpublished = {\url{https://aimslab.stanford.edu/measurement-db}},
  note         = {AIMS Lab, Stanford University}
}
```

## Reproducible reference refresh (2026-10-02)

This checkpoint refreshes 40 benchmark references and stabilizes Alpha-SQL’s derived SQL grading. Every updated benchmark produced identical Parquet files in two builds. Thirty-nine writer refreshes preserve every table value (MathArena also changes row order); Multi-SWE-bench incorporates the already reviewed full-content and grading-protocol corrections. Alpha-SQL preserves all 1,534 grades while recording a fixed SQLite engine, UTC clock, and computation allowance; its grading-protocol IDs change. Raw archives are unchanged. [The refresh record](reference_refresh_20261002.json) lists old and new file hashes and the matching code revision.
