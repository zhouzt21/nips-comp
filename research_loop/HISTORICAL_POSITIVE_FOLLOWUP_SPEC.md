# R62–R64 execution contract

Status: frozen before new outer scoring, 2026-10-22. User specification is the full, unchanged `doc/plan_4.md`; this file supplies implementation choices and does not replace or abbreviate that prompt.

## Question and falsifiable claims

Exploratory empirical follow-up: do historical, source-selected signals improve strong R25 under the original complete pair distribution? H62: low-budget prior segmentation or the already fixed first-label gate improves outer pair-equal ALC. H63: source-selected conditional corrections survive full outer pair weighting and query-specific correction adds information beyond episode-only. H64: replacing the lexical component of the fixed R46 average with the original R42 support-template removal contributes residual information beyond R25. Negative paired effects, heterogeneous fold signs, or absent residual information refute these claims at the current public-development scope.

Non-goals: new architectures, grids, hidden-test claims, fresh blind-test claims, automatic deployment or submission. All public Benchmarks have been reused adaptively; cluster bootstrap does not correct selection bias.

## Data, metric, randomness, isolation

- Corpus revision `54074453da7b9911be0f48e4c560be547bc84d21`; MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Existing migrated data and caches.
- Original R25 folds and fixed semantic/metadata ridge; 36 eligible outer Benchmarks, 5,242 unique Benchmark–subject pairs. Seeds 0,1,2 are item-split seeds, not training seeds. Six budgets 0/1/3/7/15/31, ALC weights .1/.2/.2/.2/.2/.1.
- Same historical shuffle, support acquisition, query items (up to64) and repeat responses. MLP interactor variants retain their individual counts and outcomes; R25 item predictions broadcast to those variants. Accuracy threshold >=.5, counted over responses and averaged over pairs; Brier ALC is the official optimization proxy, not accuracy.
- Before predictions, assert target Benchmarks disjoint from fit, normalization, calibration and checkpoint-selection sources; validate inner-half union and disjointness. Freeze selection JSONs and model hashes. No outer score enters fitting or selection.
- Selection epochs already selected exclusively on the outer-training sources: R54 [1,2,1], R56 [1,2,1], R43 query [0,0,20], R55 resampled_query [0,0,17], R55 resampled_episode [1,20,17]. No reselection from outer data.
- R54/R56: refit unchanged network/AdamW/data sampler/risk on all eligible outer-training Benchmarks; fixed epoch from historical inner validation. Training monitoring is not checkpoint selection. Label `foldwise refit of historical candidate`.
- R43: reuse archived full-source fold-specific weights only after matching both archive hashes and current selection metadata; epoch0 is a fold-level source-selected R25 fallback. R55: fixed fit-only normalization and historical seed5501…5520 episode banks; necessary full-source refits for nonzero epochs; epoch0 R25 fallback.
- Predictions are computed without query outcomes; store identifiers/counts/outcomes separately for reproducible scoring. Stop on isolation mismatch, baseline replay error >1e-7, nonfinite probabilities, response mismatch or changed frozen contract.

## Frozen candidates and sequence

R62: R54_segment, R56_segment use MLP residual-strength2 predictions at B0/B1 and R25 elsewhere. R61_support1_gate uses R25 at B0 and B3+, at B1 selects R56 iff its *unadapted first-support* squared error is strictly <.25, otherwise R25. Full R54 and R56 are references. Baselines: R25, smoothed_mean_2, training_selected_smoothed (source alpha [2,2,4]).

R63: original R43 query_best20, R55 resampled_query, R55 resampled_episode. All B0/B1 equal R25. Reuse identical outer trajectories and compare paired identities to R62. Corrector p=clip(b+g*d,0,1), unchanged features, loss, normalization and initialization.

R64: one intervention, replacing raw lexical prediction by template-removed lexical prediction inside the already fixed .5 semantic/.5 lexical R46 average. Compare R25, raw lexical, template lexical, raw average and template average. Templates use only the current n support inputs, no query labels or future support inputs. Original R42 tokenization/window/template and ridge are unchanged. Evaluate seed73, at most25 source pairs/Benchmark in each historical source pool. Source pools overlap; do not call them three independent trials.

R64 diagnostic: exact squared-loss decomposition relative to R25 and raw average; report residual alignment and squared displacement. Negative control: four fixed query-index permutations of template lexical residual (seeds6401–6404), preserving its episode-level mean, not changing labels. This is diagnostic only, never a deployable predictor.

R64 outer entry (frozen before pilot): template average improves R25 pair-equal ALC in at least2/3 source pools and pooled pair-weighted ALC< R25; pooled Benchmark-equal delta<=0; template average also beats raw average in pooled pair ALC. This deliberately permits one non-improving source and does not require CI<0. Require legal template inputs and nonzero residual-alignment benefit. If not met, stop R64 after pilot. If met, evaluate that one frozen average under full original outer protocol without tuning.

## Reporting and decision gates

Report pair-equal/Benchmark-equal ALC, per-budget Brier and accuracy, each fold and seed, per-Benchmark effects, alignment_faking weighted contribution and all others, worst regressions, paired Benchmark-cluster bootstrap (10,000 resamples, seed20261022, weighted numerator/denominator, same draws for all comparisons).

Research Success Gate stays unchanged: >=.005 absolute ALC improvement over frozen smoothed2 and strong simple baseline training-selected smoothed, all three folds improve, Benchmark-equal improves, cluster CI upper95%<0, no unexplained severe regression and mechanism support. Also report comparisons to R25 explicitly. Historical severe numeric flag is .03 Benchmark ALC; list .005-or-larger regressions as warnings as well. No promotion solely on favorable point estimates; no official submission or fallback overwrite. A small robust gain can merit a separate submission-feasibility review.

## Compute and provenance

Existing Python3.12 CPU runtime, Torch4threads, serial fits. Six short MLP fits; four nonzero R55 refits; three R43 weights reused. Existing train routine has1800s/fit ceiling. Each full scoring stage bounded to2h initially; if approached, inspect bottleneck and persist progress, never silently reduce pairs/seeds. Cache per-benchmark outputs to resume unchanged contracts. No costly new embedding training.

Save original code snapshots, commit and dirty state, exact commands, all model/data/selection IDs and SHA256, actual elapsed time, per-pair predictions and raw scores in new R62/R63/R64 directories. Reports must give each stage's historical positive signal, why stopped, execution/isolation audit, new evaluation, mechanism/risk, and decision. Check source and fallback hashes at completion.

## Primary-source grounding

DomainBed (Gulrajani & Lopez-Paz, ICLR2021; arXiv2007.01434), §3–3.1: model selection is part of the learning algorithm; distinguish training-domain validation from test-domain oracle. Author repository `facebookresearch/DomainBed`, pinned commit `b93c22a1cfc3b2428398272c1a116c8de1f4139e`, `domainbed/model_selection.py` IIDAccuracySelectionMethod/LeaveOneOutSelectionMethod/OracleSelectionMethod. Local full text and code: `D:/nips-comp/paiec-dataset/training_mechanism_sources/`. AnySearch academic discovery request `5ccca24a-a226-4970-b508-e5c4f5c0f67a` in this execution. PAIEC uses historical Brier and its own networks; this is reproduction of the selection/isolation principle, not the full DomainBed benchmark.

R54–R56 and R43/R42 are project historical implementations. In particular R42 template removal is a project intervention, not attributed to an external author algorithm. Existing primary-source ledgers `literature/training_mechanism_v5.md` and `literature/low_budget_prior_composition.md` retain original-paper and repository provenance for previous implementation choices.
