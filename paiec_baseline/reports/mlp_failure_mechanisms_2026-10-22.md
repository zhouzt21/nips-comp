# PAIEC: MLP failure mechanisms and targeted follow-up (2026-10-22)

**Decision:** retain `dist/smoothed_mean_strength2.zip` as the current
submission fallback. Three targeted public-data experiments did not meet the
predeclared promotion rule in
`reports/plan_1_experiment_protocol_2026-10-22.md`. These are local simulations,
not Codabench or hidden-test scores. The original development and local test
were already inspected; the grouped CV folds here were also previously used.
Thus their reuse is diagnostic and cannot supply a fresh blind performance
claim.

## Protocol and provenance

The [current official rules](https://aims.stanford.edu/competition) specify a
probability for one response, benchmark-level cold start, budgets
`0,1,3,7,15,31`, distinct-item 50/50 acquisition/evaluation pools, nested
random acquisitions without `labeling.py`, and pair-equal Brier ALC with
weights `.1,.2,.2,.2,.2,.1`. The final code and report deadline remains
**2026-10-30 23:59 AoE**. Official scoring reuses one prediction for repeated
responses with the same input and scores all those responses. The local
evaluator preserves that property, but caps evaluation at 64 items and 100
pairs per benchmark in these CV experiments. It is a bounded proxy.

Data: `D:\paiec_payload\data`, measurement-db revision
`54074453da7b9911be0f48e4c560be547bc84d21`, public binary item-level
tables. No additional data was downloaded. The Hugging Face dataset API
reported the same current repository SHA on
2026-10-22, so no newly released revision was available for an untouched
benchmark audit at the time of this run. Folds: the existing three
benchmark-disjoint folds inside the original training pool, seed 2027,
737 sampled scoreable pairs over 36 benchmark directories. Fold models and
their training metadata are in `runs/mlp_cv/fold*/`; the old development and
local test pools were never used to fit any new method in this report. The
same seed-0 item split, acquisition trajectory, evaluated responses and budgets
were used for each paired comparison. Subject configurations may recur across
benchmark folds. Cluster bootstrap resamples **benchmarks** 10,000 times;
confidence intervals describe this small observed benchmark set and are not
hidden-test guarantees.

## Experiment 0: diagnose the old MLP

Historical grouped-CV Brier, recomputed directly from the saved fold JSONs:

| Method | B0 | B1 | B3 | B7 | B15 | B31 | ALC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Empirical mean | .250000 | .353581 | .240047 | .202292 | .190509 | .183069 | .240593 |
| Smoothed mean, 2 | .250000 | .228972 | .211961 | .195988 | .188472 | .182523 | **.208331** |
| Old MLP prior | .275033 | .275033 | .275033 | .275033 | .275033 | .275033 | .275033 |
| Old MLP + residual, 2 | .275033 | .241218 | .217292 | .198772 | .190034 | .183720 | .215339 |

For residual MLP versus smoothed-2, the per-budget Brier differences are
`+.02503,+.01225,+.00533,+.00278,+.00156,+.00120`. Their weighted ALC
contributions are `+.00250,+.00245,+.00107,+.00056,+.00031,+.00012`, summing
to `+.00701`. Thus budget 0 and 1 account for about 71% of the residual
model's deficit. The residual update reduces the prior-only ALC deficit from
`.06670` to `.00701`; adaptation is useful but cannot repair the zero-label
state and does not fully remove the early-budget penalty. Higher-budget errors
converge.

The B0 pair-level decomposition uses
`Brier = Var(y) + (mean(p)-mean(y))² + Var(p) - 2 Cov(p,y)`.
Across 737 held-out pairs, these terms are respectively
`.174568 + .099386 + .001889 - .000810 = .275033`.
The **average squared pair-mean error** is large even though the signed
mean difference across pairs is only about `+.02656`; opposing pair biases
cancel. With oracle centering using evaluation labels solely for diagnosis,
the mean gain from MLP item variation, `2 Cov(p,y)-Var(p)`, is
`-.001079`; it is positive on only 10 of 36 benchmarks. The MLP rarely makes
extreme predictions (`p<.1` and `p>.9` fractions both zero here), so broad
overconfidence is not the dominant observed problem. This evidence argues
against preserving the current hashed item variation merely by correcting
the pair intercept. It does **not** show that item text itself is useless:
a different representation might transfer.

The largest historical residual-2 regressions include `appworld` (`+.0331`,
18 pairs), `bertaqa` (`+.0307`, 6), `emoji_attack` (`+.0304`, 10), and
`bird_sql` (`+.0203`, 100). `abstract_reason` is `+.0334` but has one pair.
Some benchmarks improve, including `ai2d_test` (`-.0041`, 100) and
`cmmlu` (`-.0057`, 22). This heterogeneity prevents treating a few gains as
general transfer.

### Training-sample audit

The old sampler independently draws at most 400 **responses** per benchmark,
then applies inverse-count weighting to the pairs that happened to be drawn.
This is not equivalent to selecting representative scoreable pairs. In the
45-benchmark original training pool, the 16,619 sampled rows cover 1,692
sampled pairs; 14,147 rows and 1,231 sampled pairs come from pairs that have
at least 80 distinct items in the full public data. For
`alignment_faking`, 400 rows spread over 382 of its 4,278 scoreable pairs,
with a median **one** sampled row per pair. `ai2d_test` and `hallusionbench`
have a median two. Such nearly singleton targets receive equal pair weight
despite weak within-pair estimates. This is a concrete sample-efficiency
concern, not by itself proof of the cross-benchmark failure.

The old CV weights selected epochs 2, 5, and 4 in folds 0, 1, and 2,
respectively, using internal training-pool benchmark validation. The claim
that every model stopped at epoch 2 would be inaccurate. Full training-loss
traces were not saved for the old runs, so overfitting versus unstable early
stopping is not separately identified.

## Targeted algorithm tests

All methods below were specified before scoring in the linked protocol.
Negative `ΔALC` means an improvement over frozen smoothed-2.

| ID and mechanism | Pair-equal ALC | ΔALC | Benchmark-equal ALC | Fold ΔALC vs smoothed-2 |
| --- | ---: | ---: | ---: | --- |
| Frozen smoothed-2 | **.208331** | 0 | **.217475** | 0, 0, 0 |
| Training-selected 0.5-prior smoother | .208598 | +.000267 | .216487 | 0, 0, +.000803 |
| E1: training-pair global prior + Beta smoothing | .208429 | +.000098 | .217539 | -.003121, +.002807, +.000611 |
| E2: scoreable-pair balanced MLP sample + residual-2 | .214458 | +.006127 | .227899 | +.005340, +.012090, +.000930 |
| E3: benchmark-balanced empirical Bayes prior | .210467 | +.002136 | .218267 | +.008148, -.001463, -.000288 |

**E1, global prior.** For each outer fold, set `μ` to the mean full-response
success rate across scoreable training pairs, and predict
`(α μ + successes)/(α+n)`. Choose `α∈{1,2,4,8,16}` using only the
exchangeable Binomial risk proxy on those training pairs. The selected
`(μ,α)` values were `(0.4260,2)`, `(0.4211,2)`, and `(0.5130,4)`.
The same training-only selection for a 0.5 prior yielded pooled ALC `.208598`;
E1 beat it by only `.000169` but lost to frozen smoothed-2 by `.000098`.
The benchmark-cluster 95% interval for E1 minus smoothed-2 is
`[-.00456,+.00360]`; 17/36 benchmarks improve and the worst regression
is `+.01966`. The prior mean changes substantially with training composition,
mainly because `alignment_faking` has thousands of pairs. The training-risk
proxy assumes exchangeable item labels; its selected alpha is not a calibrated
estimate of hidden-benchmark evidence strength.

**E2, pair-aware sampling.** At the same per-benchmark 400-row cap, sample
only pairs with at least 80 distinct items and distribute the row budget
across chosen pairs before fitting the unchanged 2048–128–64 MLP. This changes
training-example organization, not the encoder, network, loss, inference, or
residual-2 adaptation. It also removes non-scoreable pairs and yields fewer
total rows (folds: 8,800; 7,915; 8,000) than the old response sampler
(11,104; 10,219; 10,304), so the result cannot isolate pure row allocation
from eligibility and row count. New early-stop epochs were 1, 5, and 1.
Against the old MLP residual-2, E2 changes pooled ALC by `-.000881`, with
fold differences `-.00255,+.00731,-.00743`, a 36-benchmark cluster interval
`[-.00469,+.00430]`, and worse benchmark-equal ALC. Against smoothed-2,
all three folds lose; the paired interval is `[+.00177,+.01153]`. A single
benchmark, `enginemt_qa` (one pair), regresses by `+.16563`. Sampling quality
matters, but this intervention does not resolve the main limitation.

**E3, empirical Bayes adaptation.** Build a discrete prior over scoreable
training-pair success rates, weighting each training benchmark equally and
each pair equally within it. Smooth each training rate using a Jeffreys
half-success and half-failure count. Given `n` local labels with `s`
successes, compute

`p_n = Σ_j w_j q_j^(s+1)(1-q_j)^(n-s) / Σ_j w_j q_j^s(1-q_j)^(n-s)`.

The 32-by-32 lookup is fit entirely in each outer training fold. This tests
whether a non-Beta, potentially multimodal prior over pair prevalence helps
at low budgets. Its Brier sequence is
`.255973,.234883,.213427,.196157,.188560,.182639`, while smoothed-2 has
`.250000,.228972,.211961,.195988,.188472,.182523`.
The largest losses are at budgets 0 and 1; the pooled ALC deficit is `.002136`.
It also loses to the training-selected 0.5-prior smoother by `.001869` ALC.
It improves 20/36 individual benchmarks but loses the pair-equal target,
with cluster interval `[-.00269,+.00910]` and a worst regression `+.02449`.
The prior distribution shape learned from seen benchmarks does not transfer
reliably enough here.

## Ranked mechanism assessment and next decision

1. **Cross-benchmark pair-level bias is the strongest demonstrated bottleneck.**
   Large B0 pair bias, poor old MLP B0, and the near-zero or negative oracle
   centered item gain support this. It could reflect prevalence shift,
   subject-feature relation change, or hash representation failure; those are
   not yet causally separated.
2. **Current local residual adaptation is useful but not enough.** It removes
   most of the prior-only ALC deficit, yet n=0/1 dominate the remaining gap.
   E3 rejects one richer prevalence-prior mechanism as a reliable fix on this
   public fold set. It does not reject all Bayesian or episodic approaches.
3. **Sampling mismatch is real but not a sufficient explanation.** E2 modestly
   improves pooled ALC against the old MLP but has opposing fold effects,
   worse benchmark-equal ALC, and loses to the simple baseline everywhere.
4. **Validation composition and reuse remain important uncertainty.** Original
   development was dominated by `afrieval`; the old local test was inspected
   repeatedly. This follow-up uses 36 old training-pool benchmarks, not a new
   blind set. Confidence intervals are wide for small algorithmic differences.

Do not expand the current hashed MLP or tune prior-mix/residual constants on
the viewed test pool. If continuing research, the smallest informative next
experiment is a **field-aware item representation** (for example, isolate
subject attributes from item text and structured metadata, with a simple
sparse-text control) under a newly predeclared grouped cross-fit protocol.
Keep training recipe and adaptation fixed. Require B0 improvement over both
0.5 and the smoothed-2 prior on most folds, and separately check for positive
within-pair item gain as a post-hoc diagnostic before revisiting complex
adaptation. Abandon the representation if it only improves seen benchmarks or
one dominant public benchmark. A newly released permitted public benchmark,
if available before deadline, should be reserved untouched for final audit;
merely reshuffling these same public benchmarks does not create a new blind
test.

## Engineering status and reproduction

- `mlp/train.py` retains the original `response` sampler by default. The
  experimental `--sampling scoreable_pair_balanced` option and validation
  history recording are additive. `mlp/evaluate.py` retains old defaults and
  adds optional `--eb-table`, with revision and benchmark-overlap checks.
- New analysis/fit modules are `mlp/analyze_failure.py`,
  `mlp/diagnose_prior.py`, `mlp/stat_prior.py`, `mlp/empirical_bayes.py`,
  `mlp/summarize_stat_prior.py`, and `mlp/summarize_candidate.py`.
- Machine-readable outputs live under `runs/mlp_failure_2026_10_22/`.
  `historical_decomposition.json` and `prior_diagnostic.json` are post-hoc
  diagnostics. `stat_prior/summary.json`, `pair_balanced_summary.json`, and
  `empirical_bayes/summary.json` are the paired experimental summaries.
  Each pair-balanced fold has its own `weights.npz`/`weights.json`; folds 1
  and 2 have command logs. Fold 0's validation trajectory was observed in
  terminal output but not saved to a log; its metadata records the selected
  epoch and score. Current `mlp/train.py` SHA-256 is
  `222dcd9135ecdbbb3b926638300eae30e6caab2343f02c7fa6497bb37cc4463e`;
  `mlp/evaluate.py` SHA-256 is
  `fa87d4a75cd35474e6f2542aa7637b4c82ac784f7f5360713981454d5bf3c9ef`.
  Fold-0 training preceded the validation-history-only source edit.
- Python 3.12.7, NumPy 1.26.4, pandas 2.2.2, PyTorch 2.9.1+cpu,
  Windows 11. All 34 local tests passed after code changes (including seven
  training/evaluation regression tests). The unchanged fallback ZIP passed
  `python check_submission_zip.py dist/smoothed_mean_strength2.zip` and has
  SHA-256 `e55ee0b267f3b345685ef072cfa511c3c017d9082256573793961ea20feadcb9`.
  No new ZIP was built because no new method passed the frozen promotion rule.
  No Codabench submission was made.

From the repository root in PowerShell, the exact main commands are:

```powershell
python -m mlp.analyze_failure runs/mlp_cv/fold0/mixes.json runs/mlp_cv/fold1/mixes.json runs/mlp_cv/fold2/mixes.json --out runs/mlp_failure_2026_10_22/historical_decomposition.json
python -m mlp.diagnose_prior --data-root 'D:\paiec_payload\data' --splits runs/mlp_cv/fold0_split.json runs/mlp_cv/fold1_split.json runs/mlp_cv/fold2_split.json --weights runs/mlp_cv/fold0/weights.npz runs/mlp_cv/fold1/weights.npz runs/mlp_cv/fold2/weights.npz --out runs/mlp_failure_2026_10_22/prior_diagnostic.json
python -m mlp.stat_prior --data-root 'D:\paiec_payload\data' --parent-split runs/mlp_split_2026.json --fold-splits runs/mlp_cv/fold0_split.json runs/mlp_cv/fold1_split.json runs/mlp_cv/fold2_split.json --out-dir runs/mlp_failure_2026_10_22/stat_prior
python -m mlp.evaluate --data-root 'D:\paiec_payload\data' --split-file runs/mlp_cv/fold0_split.json --role development --public-prior runs/mlp_failure_2026_10_22/stat_prior/fold0_prior.json --seeds 1 --max-evaluation-items 64 --max-pairs-per-benchmark 100 --alphas 2 4 8 --out runs/mlp_failure_2026_10_22/stat_prior/fold0_evaluation.json
python -m mlp.empirical_bayes --cache runs/mlp_failure_2026_10_22/stat_prior/pair_rates.json --fold-splits runs/mlp_cv/fold0_split.json runs/mlp_cv/fold1_split.json runs/mlp_cv/fold2_split.json --out-dir runs/mlp_failure_2026_10_22/empirical_bayes
python -m mlp.train --data-root 'D:\paiec_payload\data' --split-file runs/mlp_cv/fold0_split.json --out runs/mlp_failure_2026_10_22/pair_balanced_fold0/weights.npz --seed 2026 --max-rows-per-benchmark 400 --sampling scoreable_pair_balanced --internal-validation --epochs 30 --batch-size 512 --learning-rate 0.001 --weight-decay 0.0001
python -m mlp.evaluate --data-root 'D:\paiec_payload\data' --split-file runs/mlp_cv/fold0_split.json --role development --weights old=runs/mlp_cv/fold0/weights.npz --weights balanced=runs/mlp_failure_2026_10_22/pair_balanced_fold0/weights.npz --seeds 1 --max-evaluation-items 64 --max-pairs-per-benchmark 100 --alphas 2 4 --out runs/mlp_failure_2026_10_22/pair_balanced_fold0/evaluation.json
python -m mlp.evaluate --data-root 'D:\paiec_payload\data' --split-file runs/mlp_cv/fold0_split.json --role development --eb-table runs/mlp_failure_2026_10_22/empirical_bayes/fold0_lookup.json --seeds 1 --max-evaluation-items 64 --max-pairs-per-benchmark 100 --alphas 2 4 8 --out runs/mlp_failure_2026_10_22/empirical_bayes/fold0_evaluation.json
python -m mlp.summarize_stat_prior --priors runs/mlp_failure_2026_10_22/stat_prior/fold0_prior.json runs/mlp_failure_2026_10_22/stat_prior/fold1_prior.json runs/mlp_failure_2026_10_22/stat_prior/fold2_prior.json --evaluations runs/mlp_failure_2026_10_22/stat_prior/fold0_evaluation.json runs/mlp_failure_2026_10_22/stat_prior/fold1_evaluation.json runs/mlp_failure_2026_10_22/stat_prior/fold2_evaluation.json --out runs/mlp_failure_2026_10_22/stat_prior/summary.json
python -m mlp.summarize_candidate --files runs/mlp_failure_2026_10_22/pair_balanced_fold0/evaluation.json runs/mlp_failure_2026_10_22/pair_balanced_fold1/evaluation.json runs/mlp_failure_2026_10_22/pair_balanced_fold2/evaluation.json --candidate balanced:residual_2 --baseline smoothed_mean_2 --baseline old:residual_2 --out runs/mlp_failure_2026_10_22/pair_balanced_summary.json
python -m mlp.summarize_candidate --files runs/mlp_failure_2026_10_22/empirical_bayes/fold0_evaluation.json runs/mlp_failure_2026_10_22/empirical_bayes/fold1_evaluation.json runs/mlp_failure_2026_10_22/empirical_bayes/fold2_evaluation.json --candidate empirical_bayes --baseline smoothed_mean_2 --baseline smoothed_mean_4 --baseline empirical_mean --out runs/mlp_failure_2026_10_22/empirical_bayes/summary.json
python -m unittest discover -s tools -p 'test_mlp*.py'
python check_submission_zip.py dist/smoothed_mean_strength2.zip
```

Repeat the shown fold-0 train/evaluation commands with fold IDs 1 and 2, then
use the respective `mlp.summarize_*` scripts to regenerate aggregate JSONs.
The report scores are tied to the saved fold weights and lookup tables; they
are **not** performance estimates for the previously built full-public MLP
refit or any hypothetical full-public statistical retraining.
