# Exploratory MLP candidate (2026-10-22)

This historical record documents an initial implementation and real-data smoke test. It
does **not** establish an improvement over the empirical-mean baseline or a
private-test result.

The exploratory weight was later copied to
`runs/mlp_trials/exploratory_pre_split/weights.npz` before `mlp/weights.npz`
was replaced by the benchmark-split candidate. The exploratory ZIP remains
`dist/mlp_exploratory.zip`; current results are in
`reports/mlp_benchmark_holdout_2026-10-22.md`.

## Input and implementation

- Public data: `D:\paiec_payload`, Measurement Data Bank revision
  `54074453da7b9911be0f48e4c560be547bc84d21`.
- Model: [`mlp/model.py`](../paiec_baseline/mlp/model.py), deterministic 2,048-dimensional
  feature hashing; `2048 -> 128 -> 64 -> 1` ReLU MLP with sigmoid output;
  matching revealed-label residual correction with prior strength 8.
- Training command:
  `python -u -m mlp.train --data-root 'D:\paiec_payload\data' --out 'mlp\weights.npz' --epochs 8 --max-rows-per-benchmark 400 --batch-size 256`.
- Training used PyTorch 2.9.1 CPU; submitted inference uses NumPy only. The
  split seed was 2026. `mlp/weights.json` records all training and validation
  benchmark directories and counts.

## Observed checks

- Training: 17,971 sampled response rows from 1,943 subject–benchmark pairs.
  Validation: 4,800 sampled rows from 372 pairs, with benchmark directories
  disjoint from training.
- Best epoch: 2 of at most 8. The zero-label, pair-weighted validation Brier
  at that epoch was **0.231831**. A constant 0.5 predictor has Brier 0.25 on
  binary responses, but this validation set was also used for epoch selection.
- A real-data smoke call on a `financebench` validation example returned
  `predict(..., []) = 0.444789` and
  `predict(..., [one matching label]) = 0.396253`; both are finite
  probabilities. These numbers illustrate the interface, not performance.
- `python tools/build_mlp_zip.py` produced `dist/mlp_exploratory.zip`.
  `python check_submission_zip.py dist/mlp_exploratory.zip` passed locally.
- `mlp/weights.npz` SHA-256:
  `411003F48B45DCB71051F8FEB3C7B92AEE1F16A9F30D4F6A3FDD404E318DCE36`.

## Limits and next evaluation

The default trainer samples at most 400 response rows per benchmark. This is
useful for a quick cross-benchmark smoke test but gives some subject groups
very few observations. The reported validation score is used to choose an
epoch and therefore is not an untouched held-out test estimate. It is also a
budget-0 score on sampled rows, whereas the competition metric is pair-equal
Brier ALC across six label budgets. Before any performance claim, freeze a
separate benchmark-level test split, replay the six acquisition budgets on the
same test groups for empirical mean, MLP prior, and MLP plus correction, and
report variation across random seeds and calibration diagnostics. Tune the
label-correction strength only on training/validation benchmarks.
