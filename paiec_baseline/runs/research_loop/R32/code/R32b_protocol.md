# R32b: stratified full-evaluation-item contrast

Frozen after count census, before selecting/scoring new outcomes. 5224/5242pairs have >64queryitems; full2179077 versus capped335247items/seed. R31 gains are zero at0/1 and77%from15/31. Census itself does not identify a noise floor.

Selection by counts only: within each benchmark, stratify evaluation pool sizes<=128,129-512,>512 crossed with any repeated-item response. Uniformly select min(2,N) pairs per stratum, except the largest benchmark by pair count (alignment_faking), min(32,N). Fixed seed3201 with stable benchmark/stratum hash. Preserve inclusion probabilities N/n. This is a representative diagnostic, not full5242-pair evaluation.

For seeds0/1/2 replay the original support draw and nested budgets. Compute full evaluation-pool predictions using unchanged R25/R7, smoothed2, and frozen training-selected smoothing. Aggregate item counts/successes analytically to retain EVERYresponse without materializing duplicate feature rows. Original64queryitems are a subset of the same pool. Additionally draw16uniform64-item subsets with fixed support/predictions, to describe conditional sampling variation, not independent experiments or predictor selection.

Record exact empirical identity: Brier(p)=weighted mean (p-item_response_mean)^2 + weighted within-item response variance. The latter is shared by all predictors; it cancels in model comparisons. It is not a population Bayes noise estimate, particularly for singleton/no-iid inputs. If interactors vary, group diagnostics explicitly or do not interpret as identical-input noise.

Report matched sample results, per-stratum/benchmark contrasts, and separately design-weighted EXTRAPOLATIONS to5242pairs. For cap-removal effect on the candidate-minus-baseline difference, use finite-population stratified sampling variance after averaging seeds within pair. Query-subset variation is conditional on current support and is not cross-benchmark uncertainty. If the shift is comparable with the0.00143656remaining gap or uncertainty covers it, expand actual evaluation before algorithm claims. Never count a metric correction as algorithm improvement or change frozen promotion criteria.

Resources: offline existing MiniLM only; cache additions onD; stream onepairquery at a time. Check number of missing vectors before encoding. No corpus download or global dependency changes.
