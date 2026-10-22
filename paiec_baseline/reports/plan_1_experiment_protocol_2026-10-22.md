# PAIEC failure follow-up: frozen decision protocol (2026-10-22)

This protocol was written before the new statistical-prior experiment was scored.
The original development, local test, and three grouped CV folds have already
been inspected. Reusing those public benchmarks gives a paired robustness
diagnostic, **not an untouched estimate of hidden-test performance**.

## Main hypothesis and minimal intervention

H1: the current MLP's zero-label prior is poorly calibrated across new
benchmarks. Compare its historical out-of-fold B0 with constant 0.5 and inspect
within-pair covariance, probability variance, and mean bias. An item-ranking
gain after oracle pair centering is diagnostic only; it uses evaluation labels.

H2: a training-only pair-weighted global prevalence can improve the prior of a
Beta-Binomial smoother. Fit the prior mean using complete binary response rates
of scoreable pairs in each outer fold's training benchmarks. Choose the prior
strength from {1, 2, 4, 8, 16} using only an exchangeable Binomial risk proxy on
those training pairs; apply the selected prior and strength to the held-out
fold. Fit the same strength grid with prior 0.5 as the stronger statistical
control. Do not choose based on the held-out fold outcomes.

The outer folds, pairs, seed 0 item splits, random acquisition trajectories,
64-item evaluation cap, six budgets, and pair-equal ALC match the historical
MLP grouped CV. This is a bounded simulation, not the full official evaluator.
MLP weights are the existing fold-specific weights; no new MLP is trained here.

## Frozen promotion criterion

Promote an algorithm as a new submission candidate only if its paired pooled
pair-equal ALC is at least **0.005 below** both frozen smoothed-2 and the
training-selected 0.5-prior smoother, all three folds improve, benchmark-equal
ALC also improves, and the benchmark-cluster paired bootstrap 95% upper bound
for the ALC difference is below zero. Report maximum benchmark regression.
This intentionally conservative criterion reflects the previously observed
0.0070 CV deficit and the limited number of independent benchmarks. Failure
does not prove the mechanism useless; it means the existing smoothed-2 ZIP
remains the evidence-backed submission fallback.

The next MLP training experiment, if warranted, will change one variable:
response-level capped sampling to pair-aware sampling at the same per-benchmark
row cap. It enters only if a sampling-coverage audit shows material loss of
scoreable pairs or the prior diagnostic shows useful within-pair signal.

## Third low-cost mechanism check, frozen before scoring

H3: fixed-strength Beta smoothing may be suboptimal if the distribution of
pair success rates is multimodal. Form a benchmark-balanced discrete prior from
training-pair success rates, with Jeffreys smoothing for finite rate estimates.
Use its exact Bernoulli posterior mean for every count (successes, labels) up
to 31. The entire lookup is fit on the outer fold's training benchmarks; no
held-out responses enter it. Compare paired against smoothed-2, the
training-selected 0.5-prior smoother, and the training-selected global-pair
prior. Use the same conservative promotion criterion above. This is a
distribution-shape adaptation test, not another fixed MLP mix.
