# R15b / V2-06: source-only relative capability and a matched peer correction

Registered2026-10-22 after R15a input audit, before response aggregation, fitting or held-out prediction. Across737targetpairs,464have a matching(name,provider,reasoning_effort) source entry,213match at least2source benchmarks. Full visible configurations have zero cross-benchmark overlap. This is potential coverage; binary support remains to be checked.

## High-level hypothesis and boundary

Some relative subject ability may be transferable even when a new benchmark's absolute prevalence is not. Rival: rank relations change with benchmark/harness or sparse source co-occurrence makes capability estimates unreliable. The key is a **relative, benchmark-centered** effect, not a new global success-rate lookup.

Observational Scaling Laws v2 methods3.2/3.4 use80models on8benchmarks with<1%missing, PCA and target-task regression; our matrix is much sparser. Do not import those scores, assume a shared PCA space, or claim replication. The paper motivates a structure check only.

## Source formation and solver

For each original-training-pool directory, aggregate only binary responses, retain all response trials, require80distinct items for a subject pair. Cache counts/rates, not sampled labels. For each outer fold fit using **only its train_benchmarks**. Held-out dictionaries cannot enter the fit.

Fit least squares `q_bs = d_b + a_key` with per-subject-pair equal weight and ridge penalty `sum_key a_key^2` (fixed lambda1, not tuned). Source benchmark intercepts are nuisance parameters, never used on target. Alternating exact updates to convergence; zero-centered ridge anchors the ability scale. Use inferred ability only for keys with at least2source benchmarks; unknown/single-benchmark keys yield no correction. No private benchmark identity enters prediction features.

## Downward tests

1. Diagnostic: within held-out benchmark, compare inferred ability differences to differences in evaluation response means among supported subjects. Report sign agreement and centered covariance by benchmark. Evaluation outcomes are diagnostic only; no ability refit.
2. Candidate: retain R12 alpha/kappa and cohort4 fixedseed193, itemseeds0/1/2. Modify peer prior numerator by `sum_peer_count*(a_target-a_peer)` **only when both keys are supported**. Clip the corrected peer prior to[0,1]; then the exact same own-label shrinkage. No peers/unsupported keys give the original peer predictor.
3. Baselines: exact R12 peer and same-evidence smoothed2. No semantic combination until this isolated mean component shows value.

Interpretation: positive rankings but no Brier gain suggests scale/calibration/limited coverage, not absence of relative signal. Rank reversals and negative Brier reject this scalar capability assumption. Low usable coverage directs investigation to feature transfer rather than an identity table. Any gain remains model-known/benchmark-unseen evidence, not subject cold-start evidence.

No optional parameter search, no hidden test feedback, no new external competition training data. Save source manifest, coverage, convergence, coefficients, perbudget/fold/BM metrics and actual runtime.
