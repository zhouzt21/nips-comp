# R27: budget-dependent statistical baseline audit

Frozen before scoring, 2026-10-22. Operator: re-express the estimation target -> derive a parameter-free control -> test transport of its assumptions.

R26 online variance acquisition improves only 0.00008950 over random (CI crosses zero; one fold worsens). Stop that selector. R25 gains remain far below the 0.005 gate. Before increasing complexity, test whether constant prior strength is a poor baseline across the six budgets.

Candidate: for n > 0 use p = (sum(y) + sqrt(n)/2)/(n + sqrt(n)); for n = 0 use 0.5. No fitted parameters. Primary source: https://arxiv.org/html/2009.11413v2, introduction equation (1), already retrieved via AnySearch and read locally. This is the classical minimax estimator of an iid Bernoulli parameter under squared parameter error. Its risk is 1/[4(sqrt(n)+1)^2]. It is NOT a minimax predictor under full Brier loss (which adds q(1-q)), nor a theorem for heterogeneous finite PAIEC items. Do not claim novelty.

Use the unchanged three outer folds, seeds 0/1/2, 100 pairs per benchmark, fixed acquisition/evaluation records, 64 evaluation items retaining all responses, isolated-pair evidence. Compare empirical mean, smoothed2, the previously training-selected smoothed mean, and frozen R25. Reproduce historical smoothed2 per-benchmark scores to 1e-7 before comparison. Average seeds within benchmark before paired cluster bootstrap; report all budgets, folds, worst benchmark and both aggregation units. No selection on outer labels. Reused public benchmarks are not blind data.

Accept as a stronger baseline only if it improves all folds and its paired CI upper bound is below zero against smoothed2; the unchanged 0.005 research gate remains separate. If it fails, do not sweep exponents or coefficients on outer scores. If useful, a separate experiment may test replacement of the mean while freezing item correction. Save negative results and mathematical checks.
