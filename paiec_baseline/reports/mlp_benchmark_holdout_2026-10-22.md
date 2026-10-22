# PAIEC MLP: public benchmark holdout experiment (2026-10-22)

This report records **local public-data simulations**, not a Codabench score.
The [competition specification](https://aims.stanford.edu/competition) defines
the prediction interface, six budgets, item-level acquisition split, and equal
subject–benchmark-pair Brier ALC. Raw outputs are in `runs/mlp_trials/`.

## Data and protocol

- Source: `aims-foundations/measurement-db`, revision
  `54074453da7b9911be0f48e4c560be547bc84d21`, prepared at
  `D:\paiec_payload\data` using `--no-embeddings --no-traces`.
  The four core Parquet tables are complete for 61 public, binary,
  item-granularity directories. Traces and embeddings are unused by this model.
- Audit: 16,423,758 response rows; 16,286,209 binary outcomes; 137,549 missing
  outcomes excluded; 1,356,713 item rows; 8,163 subject rows. Within each
  benchmark table, there are no duplicate response/subject/item IDs or orphan
  joins among labeled responses. Subject IDs can recur across benchmarks. There
  are 6,870 scoreable subject–benchmark pairs in 52 directories (at least 80
  distinct items). Full audit: `runs/public_data_audit.json`.
- Frozen split: `runs/mlp_split_2026.json`, seed 2026, SHA-256
  `376ba49beb2447ab68af0584644711883e7b75423d200ae395ae2ad8058d45be`.
  It places the three largest scoreable benchmarks into separate pools, then
  shuffles the remainder: 45 training benchmarks / 5,242 scoreable pairs,
  8 development / 1,061 pairs, 8 local test / 567 pairs. Benchmarks are
  disjoint, but subject configurations can recur.
- Evaluator: `mlp/evaluate.py` splits **distinct items** 50/50 within each pair,
  keeps all trials of an item on one side, randomly reveals one response for
  each of the first 31 acquisition items, and scores every response among up
  to 64 evaluation items. The six budgets use nested labels and the same
  evaluation responses. ALC weights are `0.1, 0.2, 0.2, 0.2, 0.2, 0.1`; pair
  scores are averaged equally. The 64-item cap makes this a bounded proxy,
  and the hidden evaluator's fixed split and sampled subjects may differ.

The standalone official empirical-mean replay, with **all** evaluation items
and five split seeds across all public benchmarks, yielded ALC
**0.232639 ± 0.000494** (6,870 pairs/seed; about 8.10 million evaluation
responses/seed). Its six Brier scores were `0.250000, 0.342977, 0.228291,
0.195800, 0.182678, 0.176901`. See
`reports/empirical_mean_public_2026-10-22.md`. Those numbers have a different
scope from the split-specific, 64-item-capped comparisons below.

## Models and selection

The MLP uses stable signed hashing of eight subject fields and item text,
item features, and interactors into 2,048 dimensions; two ReLU layers have
128 and 64 units. It outputs a sigmoid probability. The submitted predictor
uses only the current call's matching subject and anonymous benchmark labels
to add the shrunken mean residual; it never learns the benchmark ID. The
training loss is pair-weighted Brier, optimized with AdamW. Only NumPy is
needed for inference.

| ID | Fit and early stopping | Development role | Main purpose |
| --- | --- | --- | --- |
| `C0-cap400` | 16,619 sampled training rows; early stop on sampled development rows, epoch 2 | Also used to choose residual strength 8 | Initial candidate; development score is optimistic for selection |
| `C1-internal` | 14,619 sampled fit rows; 2,000 rows in five *training-pool* benchmarks for early stopping, epoch 2 | Entire development pool reserved for choosing adaptation strength 4 | Cleaner MLP candidate |
| `A1-subject-only` | Same internal split and architecture as C1; item fields removed | Development comparison | Isolate the value of item features |

All three use seed 2026, at most 400 response rows per benchmark, batch size
512, learning rate `0.001`, weight decay `0.0001`, and pair-equal sample
weighting. C1's internal early-stop Brier was `0.214975` on 253 sampled pairs.
The training and inference weight check on eight float32 vectors differed by
at most `1.19e-7` in probability. Weight metadata and exact directory lists
are beside each `weights.npz` under `runs/mlp_trials/`.

The public subject-name prior is a separate non-neural control. It shrinks
per-name means of **training** subject–benchmark pairs toward the training
global mean with strength 3. Only 37/1,061 development and 38/567 test pairs
have an exact training name match; missing names use the global prior.

## Development results

One item-split seed, 1,061 pairs, 68,556 scored responses. Adaptation strengths
were compared **here**, not chosen from the local test pool.

| Method | B0 | B1 | B3 | B7 | B15 | B31 | ALC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Empirical mean | .250000 | .349694 | .235062 | .198435 | .184647 | .179378 | .236505 |
| Smoothed mean, strength 2 | .250000 | .227676 | .208835 | .192815 | .183303 | .178859 | .205411 |
| C1 MLP prior, no labels | .208058 | .208058 | .208058 | .208058 | .208058 | .208058 | .208058 |
| A1 subject-only + residual, strength 2 | .242628 | .224305 | .207498 | .192293 | .183077 | .178806 | .203578 |
| C1 full MLP + residual, strength 4 | .208058 | .202257 | .195825 | .187336 | .180943 | .177405 | **.191819** |
| C0 full MLP + residual, strength 8 | .193839 | .191918 | .189514 | .185365 | .181566 | .178897 | .186946 |

The public subject-name prior alone scored `.289751`; combined with two
local labels of prior strength it scored `.214882` ALC. Item features lower
C1's development ALC by `.011759` compared with the matched subject-only
network. At budget 31 the gap shrinks to about `.0014`, consistent with
local evidence becoming more informative. This is a development-set
observation; it does not prove item-feature transfer to all new benchmarks.

Development is dominated by `afrieval` (948/1,061 pairs). C1 residual-4
beats smoothed-2 on 5/8 development benchmarks; its benchmark-unweighted
mean ALC is `.218330` versus `.220529`. The alternative MLP-to-local-mean
shrinkage rule gave `.192261` pair-equal ALC and `.215949` benchmark-unweighted
ALC at strength 4. These secondary checks expose sensitivity to benchmark
composition; they were not used to change the frozen C1 predictor.

## Local test results and interpretation

The first local test replay evaluated the frozen C0/strength-8 candidate.
After seeing its poor transfer, C1 was trained using an internal early-stop
fold and selected on development. **C1's test replay is a post hoc second
look**, so it is useful for diagnosis but not an untouched estimate for
further method selection. Both replays used the same 567 pairs, three item
split seeds, and 64-item cap; each seed scored roughly 89,000 responses.

| Method | B0 | B1 | B3 | B7 | B15 | B31 | ALC ± seed SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Empirical mean | .250000 | .427660 | .284020 | .243360 | .227560 | .219970 | .283518 ± .000275 |
| Smoothed mean, dev-selected strength 2 | .250000 | .253664 | .245433 | .233666 | .225054 | .219396 | .238503 ± .000287 |
| Smoothed mean, strength 8 (diagnostic) | .250000 | .245049 | .238635 | .230816 | .224118 | .219265 | **.234650 ± .000351** |
| C0 full MLP + residual, strength 8 | .267877 | .259190 | .248360 | .237110 | .228000 | .221690 | .243491 ± .000555 |
| C1 MLP prior, no labels | .253891 | .253891 | .253891 | .253891 | .253891 | .253891 | .253891 ± .000416 |
| C1 full MLP + residual, dev-selected strength 4 | .253891 | .248128 | .240888 | .232427 | .225609 | .220542 | **.236854 ± .000432** |
| C1 MLP-to-local-mean shrinkage, strength 4 (diagnostic) | .253891 | .247638 | .240025 | .231280 | .224309 | .219203 | .235960 ± .000425 |

C1 improves on the **development-selected** smoothed-2 comparator by
`.001649` ALC in this replay, but it is worse than the test-favored
smoothed-8 comparator by `.002204`. The latter strength was not selected on
development, so it must not be treated as an unbiased winner. C1's zero-label
prior is worse than constant 0.5 on this test, whereas residual correction
helps at budgets 1–7. At budgets 15–31 the smoothed-2 comparator is slightly
better. C1 beats smoothed-2 on 5/8 test benchmarks, including `bridging_gap`
(`.2491` versus `.2521`), but is much worse on `crow` (`.1404` versus `.1035`).
The benchmark-unweighted means are `.208483` and `.208930`, respectively;
the small difference and eight-benchmark sample do not support a broad
superiority claim.

`afrieval` supplies 89% of development pairs and has no nonempty subject
names, while `bridging_gap` supplies 74% of test pairs and only 3/417 names
match the training-name prior. The observed transfer gap could reflect
benchmark content, label prevalence, feature semantics, or sampling. Those
causes have not been isolated experimentally. No hidden Codabench feedback
was used.

## Three-fold grouped robustness check

After the first local test exposed weak transfer, we made three additional
benchmark-disjoint folds **inside the original training pool**. Each fold
trained a fresh MLP with an internal early-stop fold and evaluated on its
untouched benchmark fold. Original development and local test benchmarks were
excluded from every CV fit. The folds contain 246, 246, and 245 sampled pairs,
with at most 100 pairs per benchmark and 64 evaluation items per pair. The
fold assignment seed is 2027. This is a post hoc diagnostic; the fold cap and
changing fit composition differ from the official scoring distribution.

| Method | Fold 0 | Fold 1 | Fold 2 | Pair-equal pooled ALC | Benchmark-equal ALC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoothed mean, strength 2 | .177406 | .239913 | .207670 | **.208331** | **.217475** |
| MLP prior, no labels | .277683 | .266130 | .281313 | .275033 | .272307 |
| MLP residual, strength 2 | .185293 | .244697 | .216028 | .215339 | .223657 |
| MLP prior mix 0.25 + residual, strength 2 | .178704 | .239776 | .208166 | .208883 | .217749 |
| MLP prior mix 0.5 + residual, strength 2 | .180455 | .240530 | .209728 | .210238 | .218875 |

The pooled and benchmark-equal values come from
`runs/mlp_cv/summary_mixes.json`.
Mix 0 reduces exactly to smoothed mean. Among the evaluated mixes
`0, 0.25, 0.5, 0.75, 1` and strengths `2, 4, 8`, every nonzero MLP mix had a
higher pooled ALC than smoothed-2. The result contradicts a general claim that
the current MLP improves prediction across held-out public benchmarks.

**Current evidence-based submission fallback:**
`dist/smoothed_mean_strength2.zip`. Keep the MLP ZIP as an executable research
candidate and do not present its small post hoc local-test advantage over
smoothed-2 as stable superiority. A fresh independent benchmark pool is needed
to judge any new feature or calibration design without reuse of the examined
local test pool.

## Engineering acceptance and next decision

- `dist/mlp_candidate_internal_alpha4.zip` contains `model.py`, fitted
  `weights.npz`, README, and license at the ZIP root. The local organizer
  `check_submission_zip.py` and `tools/smoke_mlp_zip.py` passed; the latter
  imports the archive in an isolated Python process and checks missing fields,
  unseen benchmark, six budgets, bounded probabilities, and call isolation.
  Weight SHA-256: `19870c69f37e8612d0835a732f2b14c8763515a549d7705d90a77b45c5781085`.
  ZIP SHA-256: `97a6553666f457545d400f065e8ad79c15b43950308adfe1bf8319a9e45219ee`.
- `dist/smoothed_mean_strength2.zip` is a second, dependency-free ZIP with
  the development- and CV-supported statistical fallback. It also passes the
  local ZIP validator. ZIP SHA-256:
  `e55ee0b267f3b345685ef072cfa511c3c017d9082256573793961ea20feadcb9`.
- `runs/mlp_trials/full_public_epoch2/weights.npz` refits the fixed C1
  architecture for two epochs across **all 61 public core directories**
  (one has no usable binary labels), with
  22,771 sampled responses from 2,322 sampled groups. Its separate
  `dist/mlp_full_public_epoch2_alpha4.zip` passed the validator and isolated
  smoke test. Weight SHA-256:
  `c334e5fc596cba96a7bbee038c2a9da19ee3f4054b860cfa1aaf2b3e606c3c2`;
  ZIP SHA-256:
  `e174ba66edc1ab7d51d08f980e972c63e1c3dc9c4161b9fa9c14f50555f9f591`.
  This refit uses the former development and local test benchmarks for
  training, so the held-out scores above **do not estimate these exact
  weights**. It is an experimental full-data artifact, not the recommended
  evidence-based fallback.
- All 32 local unit tests passed on Windows. The BLE packaging tests use the
  Python packager directly on Windows because Git Bash does not resolve
  native temporary paths passed by `subprocess`.
- Environment: Windows 11, Python 3.12.7, PyTorch 2.9.1 CPU for training,
  NumPy 1.26.4, pandas 2.2.2, PyArrow 16.1.0. No model API or network access
  is required at inference.
- The official starting-kit `run_smoke_test.py` was not available in this
  checkout; the local baseline validator and an isolated archive smoke test
  were run. No Codabench submission or NeurIPS report upload occurred.

The next high-value experiment is a targeted item-feature redesign and
calibration check on **new benchmark-grouped folds**, testing whether the
zero-label prior can beat the simple 0.5 prior in most folds. The existing
three-fold result already rejects the easiest fix, uniform shrinkage of the
current MLP prior toward 0.5. A new frozen benchmark pool is needed for an
unbiased claim about any revised method; the current local test has been
examined.

## Reproduction from the baseline repository root (PowerShell)

```powershell
python tools/prepare_data.py --out 'D:\paiec_payload' --no-embeddings --no-traces
python tools/audit_public_data.py --data-root 'D:\paiec_payload\data' --out 'runs\public_data_audit.json'
python tools/evaluate_mean.py --data-root 'D:\paiec_payload\data' --seeds 5 --out 'runs\empirical_mean_public.json'
python -m mlp.make_split --data-root 'D:\paiec_payload\data' --seed 2026 --out 'runs\mlp_split_2026.json'
$env:OPENBLAS_NUM_THREADS='1'
python -m mlp.train --data-root 'D:\paiec_payload\data' --split-file 'runs\mlp_split_2026.json' --out 'runs\mlp_trials\internal_val_cap400\weights.npz' --seed 2026 --max-rows-per-benchmark 400 --epochs 12 --batch-size 512 --learning-rate 0.001 --weight-decay 0.0001 --feature-mode all --internal-validation
python -m mlp.evaluate --data-root 'D:\paiec_payload\data' --split-file 'runs\mlp_split_2026.json' --role development --weights 'internal=runs\mlp_trials\internal_val_cap400\weights.npz' --seeds 1 --max-evaluation-items 64 --alphas 2 4 8 16 --out 'runs\mlp_trials\development_replay.json'
Copy-Item 'runs\mlp_trials\internal_val_cap400\weights.npz' 'mlp\weights.npz'
python tools/build_mlp_zip.py --out 'dist\mlp_candidate_internal_alpha4.zip'
python check_submission_zip.py 'dist\mlp_candidate_internal_alpha4.zip'
python tools/smoke_mlp_zip.py 'dist\mlp_candidate_internal_alpha4.zip'
$env:PYTHONPATH='.;tools'
python -m unittest discover -s tools -p 'test_*.py'
```

The additional grouped-CV and full-public refit commands are:

```powershell
python -m mlp.make_cv_splits --parent-split 'runs\mlp_split_2026.json' --out-dir 'runs\mlp_cv' --seed 2027 --folds 3 --pair-cap 100
$env:OPENBLAS_NUM_THREADS='1'
for ($i=0; $i -lt 3; $i++) {
  $split = "runs\mlp_cv\fold${i}_split.json"
  $weights = "runs\mlp_cv\fold${i}\weights.npz"
  $scores = "runs\mlp_cv\fold${i}\mixes.json"
  python -m mlp.train --data-root 'D:\paiec_payload\data' --split-file $split --out $weights --seed 2026 --max-rows-per-benchmark 400 --epochs 12 --batch-size 512 --learning-rate 0.001 --weight-decay 0.0001 --feature-mode all --internal-validation
  python -m mlp.evaluate --data-root 'D:\paiec_payload\data' --split-file $split --role development --weights "fold${i}=$weights" --seeds 1 --max-evaluation-items 64 --max-pairs-per-benchmark 100 --alphas 2 4 8 --prior-mixes 0 0.25 0.5 0.75 1 --out $scores
}
python -m mlp.summarize_cv --input 'runs\mlp_cv\fold0\mixes.json' --input 'runs\mlp_cv\fold1\mixes.json' --input 'runs\mlp_cv\fold2\mixes.json' --out 'runs\mlp_cv\summary_mixes.json'
python -m mlp.train --data-root 'D:\paiec_payload\data' --split-file 'runs\mlp_split_2026.json' --out 'runs\mlp_trials\full_public_epoch2\weights.npz' --seed 2026 --max-rows-per-benchmark 400 --fixed-epochs 2 --batch-size 512 --learning-rate 0.001 --weight-decay 0.0001 --feature-mode all
python tools/build_mlp_zip.py --out 'dist\mlp_full_public_epoch2_alpha4.zip' --weights 'runs\mlp_trials\full_public_epoch2\weights.npz' --readme 'mlp\README_full_public.md'
python check_submission_zip.py 'dist\mlp_full_public_epoch2_alpha4.zip'
python tools/smoke_mlp_zip.py 'dist\mlp_full_public_epoch2_alpha4.zip'
python tools/build_smoothed_zip.py
python check_submission_zip.py 'dist\smoothed_mean_strength2.zip'
```

The local test replays used these options; the current evaluator may also
emit additional diagnostic methods added afterward. Do not use their
outcomes to retune the same test pool:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python -m mlp.evaluate --data-root 'D:\paiec_payload\data' --split-file 'runs\mlp_split_2026.json' --role test --weights 'cap400=runs\mlp_trials\cap400\weights.npz' --seeds 3 --max-evaluation-items 64 --alphas 2 8 --out 'runs\mlp_trials\test_frozen_v1.json'
python -m mlp.evaluate --data-root 'D:\paiec_payload\data' --split-file 'runs\mlp_split_2026.json' --role test --weights 'internal=runs\mlp_trials\internal_val_cap400\weights.npz' --seeds 3 --max-evaluation-items 64 --alphas 2 4 8 --out 'runs\mlp_trials\test_internal_val_posthoc.json'
```
