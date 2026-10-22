# Empirical mean baseline: public-data proxy (2026-10-22)

This is a **local simulation on public training benchmarks**, not a Codabench
hidden-test result. The raw machine-readable output is
`runs/empirical_mean_public.json`; the downloaded data and Hugging Face cache
are at `D:\paiec_payload` and `D:\paiec_hf_cache` respectively.

## Reproduction contract

- Dataset: `aims-foundations/measurement-db`, exact revision
  `54074453da7b9911be0f48e4c560be547bc84d21`.
- Preparation: `python tools/prepare_data.py --out 'D:\paiec_payload' --no-embeddings --no-traces`.
  This checkout added `--no-traces`; all required response, subject, item and
  benchmark tables were downloaded. The manifest lists 61 binary,
  item-granularity directories and 244 files; all 244 files were verified
  present. The local payload is about 0.796 GB.
- Evaluator: `python tools/evaluate_mean.py --data-root 'D:\paiec_payload\data' --seeds 5 --out runs/empirical_mean_public.json`.
- For each subject–benchmark group with at least 80 distinct items, the
  evaluator randomly splits items 50/50. All trials of an item remain in one
  pool. It reveals one randomly selected recorded response per acquisition
  item, with nested budgets `0, 1, 3, 7, 15, 31`; all evaluation responses
  are scored. The predictor returns `0.5` at budget 0 and the unsmoothed mean
  of acquired labels at positive budgets.
- Each subject–benchmark pair contributes equally to each budget's Brier
  score. Brier ALC uses weights `0.1, 0.2, 0.2, 0.2, 0.2, 0.1`.

## Results

| Acquired labels | Mean Brier over 5 seeds | Constant 0.5 reference |
| ---: | ---: | ---: |
| 0 | 0.250000 | 0.250000 |
| 1 | 0.342977 | 0.250000 |
| 3 | 0.228291 | 0.250000 |
| 7 | 0.195800 | 0.250000 |
| 15 | 0.182678 | 0.250000 |
| 31 | 0.176901 | 0.250000 |

**Brier ALC: 0.232639 ± 0.000494** (mean ± sample standard deviation over
five item-split seeds). Each seed scores 6,870 subject–benchmark pairs and
about 8.10 million evaluation responses. The budget-1 predictor always
outputs 0 or 1 from one observed outcome; this high-variance estimate explains
why its observed Brier score is worse than the constant reference at that
budget.

Of the 61 prepared directories, 60 contributed at least some binary rows.
`chartmuseum/formatted_tables/response.parquet` has 162 rows, all with a
missing (`NaN`) response, and therefore contributes no group. Groups with
fewer than 80 distinct items are also excluded from the split simulation.

## Interpretation boundary

The hidden test pool uses different benchmarks and the organizer fixes its
own acquisition/evaluation split and sampled subjects. This public proxy
supports method development and a reproducible report baseline; it does not
estimate the private leaderboard score without uncertainty. The local script
groups public responses by `subject_id` within each benchmark directory,
whereas the submitted `empirical_mean/model.py` matches the complete visible
subject dictionary and the anonymous `benchmark_id` supplied at runtime.
