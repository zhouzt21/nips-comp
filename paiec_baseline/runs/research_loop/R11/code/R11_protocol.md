# R11 / V2-02: is peer heterogeneity an estimable source of pooling failure?

Registered2026-10-22 before selection/scoring. Conditional shared-evidence branch only; frozen isolated-pair success gate unchanged.

## Before / upward diagnosis

R10 supports a small item-discrimination component, mostly at higher budgets. R9's low-budget peer signal has larger pooled gains but reverses in fold1. A shared benchmark does not imply equal subject abilities. The new scientific object is the target subject prevalence conditional on currently observed peer counts, with uncertainty about between-subject spread. It is not generic architecture capacity.

R9's plug-in prior treats all peer trials as estimates of a common success rate and uses fixed own strength2. Competing explanations: (A) ignoring between-subject heterogeneity causes negative transfer; (B) few peer groups make heterogeneity too weakly identified; (C) cohort sampling/benchmark composition, not the estimator, dominates. Do not assume A.

## Operator sequence

Audit exchangeability assumption -> distinguish population mean from subject rates -> condition on the current counts -> integrate uncertain heterogeneity -> compare with fixed heterogeneity on identical evidence. The hierarchy only follows the already observed conditional peer signal; it is not combined with semantic ridge in this round. User operator references sections3/10 and Stan binary pooling methods support formulation, not predicted PAIEC efficacy.

## Model

`theta ~ Beta(kappa/2,kappa/2)`, `q_s | theta,c ~ Beta(c*theta,c*(1-theta))`, `S_s | q_s ~ Binomial(n_s,q_s)`.

Fixed control c=4. Adaptive candidate c has uniform discrete prior over {1,4,16}. Marginal likelihood for each group is `B(S+c*theta,n-S+c*(1-theta))/B(c*theta,c*(1-theta))`; integrate theta with128 Gauss-Legendre nodes and sum over c. Binomial coefficients cancel in posterior weights. Prediction integrates `(S_target+c*theta)/(n_target+c)`; includes all allowed evidence, never evaluation outcomes. With <=186observations and integer kappa2/8, the unnormalized integrand is a polynomial within128-node quadrature degree; numerical convergence checked independently at256nodes.

When only one distinct visible subject contributes, use exact smoothed2. This is an explicit hybrid fallback, not the pure hierarchical posterior. Thus size1 must exactly reproduce baseline. Normalize groups only by complete visible subject dict and same benchmark association.

## Selection and matching

Select kappa2 or8 separately for fixed/adaptive using R9's outer-training benchmarks, seed73, max25pairs/BM, cohort4. Four configurations total. All fold selections saved before outer evaluation. Compare R9 frozen plug-in, same-evidence smoothed2, training-selected smooth. Same targetpairs, labels, cohortseed193 and evaluationresponses; cohort4 primary,1/2/6 stress with frozen choices. No unseen held-out labels used for selection.

## Predeclared interpretation

- Adaptive beats fixed and plug-in across folds: support heterogeneity integration, subject to CI/effect-size gates.
- Fixed/adaptive both lose plugin: reject this hierarchy/priors, not all Bayesian inference. Inspect whether prior mean/concentration constraints dominate before further search.
- Adaptive only helps n15/31: heterogeneity needs more observations; cannot claim solution to early-budget bottleneck.
- Both regress on same benchmarks: retains cohort/composition/conditional-dependence explanation; avoid repeatedly enlarging concentration grids.
- Nopeer fallback, complement symmetry, permutation invariance and128/256quadrature checks are required before scoring.

Exact hosted cohort distribution is unverified. This round cannot authorize promotion even if conditional score threshold passes. Report worst benchmark, fold directions, perbudget, posterior concentration, code/data hashes, runtime; do not equate posterior concentration with recovered true ability variance.
