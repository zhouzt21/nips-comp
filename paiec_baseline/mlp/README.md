# PAIEC offline MLP predictor

## Code layout

The package root contains predictor/model components, feature encoders,
`train.py`, and `evaluate.py`. Experiment orchestration lives in subpackages:

| Directory | Responsibility |
| --- | --- |
| `audit/` | Data audits, failure diagnostics, and diagnostic analyses |
| `run/` | Experiment drivers |
| `verify/` | Frozen-parameter checks and replay verification |
| `reporting/` | Result summaries, plots, and report generation |
| `state/` | Research status and checkpoint document updates |
| `data/` | Benchmark holdout and cross-validation splits |

Run modules from `paiec_baseline/`, for example
`python -m mlp.audit.audit_evaluation_items` or
`python -m mlp.run.run_metadata_round`. These commands run experiments;
see the [reproduction guide](../../reports/research_loop_v3_reproduction.md)
for their data dependencies and admission conditions. Historical module names
are mapped in [module_path_migration.json](../../doc/module_path_migration.json).
The competition entry point remains `model.py`.

## Predictor

`model.py` implements the competition's
[`predict(input, labeled) -> float`](https://aims.stanford.edu/competition#task)
interface. It needs NumPy and the adjacent fitted `weights.npz`; inference is
offline. The current weight file is the `C1-internal` candidate trained on
public measurement-db data at revision
`54074453da7b9911be0f48e4c560be547bc84d21`. Its training record is
`weights.json`. The same artifact is preserved under
`runs/mlp_trials/internal_val_cap400/`.

This is a **local research candidate**, not a verified private-test winner.
The full data, split, training, baseline, six-budget comparison, negative
results, and limitations are in
[`reports/mlp_benchmark_holdout_2026-10-22.md`](../../reports/mlp_benchmark_holdout_2026-10-22.md).

## Method

`encode()` uses deterministic signed feature hashing into 2,048 dimensions
for the subject's eight public fields and the item's content, features, and
interactors. It handles blank or missing values, bounds long text, and adds
word/bigram features plus Chinese character bigrams. The anonymous
`benchmark_id` is never a learned feature; it is used only to match the
current target to allowed revealed labels.

`forward()` applies a `2048 -> 128 -> 64 -> 1` ReLU MLP with a sigmoid output.
`predict()` starts with that probability and adds the mean acquired residual
`label - MLP(observed item)` divided by `number of matching labels + 4`, then
clips to `[0, 1]`. It uses only matching subject and benchmark observations
from the current `labeled` argument and retains no cross-call state. Random
acquisition is supplied by the evaluator; no `labeling.py` is packaged.

Training uses PyTorch locally, pair-weighted Brier loss, and a benchmark-level
split. The selected artifact uses a further five-benchmark early-stop fold
inside the training pool; the development pool selects residual strength 4.
The local test pool is benchmark-disjoint. The development ALC is `0.191819`
and the post hoc local test ALC is `0.236854` under the documented 64-item
proxy. See the report for all six budgets and fair simple-method comparisons.

## Reproduce and package (PowerShell, repository root)

```powershell
python tools/prepare_data.py --out 'D:\paiec_payload' --no-embeddings --no-traces
python -m mlp.data.make_split --data-root 'D:\paiec_payload\data' --seed 2026 --out 'runs\mlp_split_2026.json'
$env:OPENBLAS_NUM_THREADS='1'
python -m mlp.train --data-root 'D:\paiec_payload\data' --split-file 'runs\mlp_split_2026.json' --out 'runs\mlp_trials\internal_val_cap400\weights.npz' --seed 2026 --max-rows-per-benchmark 400 --epochs 12 --batch-size 512 --learning-rate 0.001 --weight-decay 0.0001 --feature-mode all --internal-validation
python -m mlp.evaluate --data-root 'D:\paiec_payload\data' --split-file 'runs\mlp_split_2026.json' --role development --weights 'internal=runs\mlp_trials\internal_val_cap400\weights.npz' --seeds 1 --max-evaluation-items 64 --alphas 2 4 8 16 --out 'runs\mlp_trials\development_replay.json'
Copy-Item 'runs\mlp_trials\internal_val_cap400\weights.npz' 'mlp\weights.npz'
python tools/build_mlp_zip.py --out 'dist\mlp_candidate_internal_alpha4.zip'
python check_submission_zip.py 'dist\mlp_candidate_internal_alpha4.zip'
python tools/smoke_mlp_zip.py 'dist\mlp_candidate_internal_alpha4.zip'
```

`--no-traces` omits large trace files unused by this method while preserving
the four core tables. The original official `--no-embeddings` preparation
also downloads traces. The local `D:\paiec_payload_full` trace-inclusive
attempt is incomplete and must not be used without its final manifest.
Downloaded data, model weights, and generated ZIPs are local ignored
artifacts. Do not upload the archive or publish data without the appropriate
competition submission steps and data-license review.

## 2026-10-22 failure follow-up

The grouped-CV failure diagnosis and three targeted experiments are in
`reports/mlp_failure_mechanisms_2026-10-22.md`, with machine-readable results
under `runs/mlp_failure_2026_10_22/`. None demonstrated stable improvement
over `smoothed_mean_strength2.zip`; it remains the current submission fallback.
`--sampling scoreable_pair_balanced` and `--eb-table` are experimental options,
not the selected predictor. The old development and local test results above
are historical observations and should not be used as blind validation for
new method selection.
