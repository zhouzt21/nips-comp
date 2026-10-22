# Smoothed empirical mean control

This self-contained predictor implements a beta-style `0.5` prior with two
pseudo-observations for the current subject–benchmark pair. It follows the
same random acquisition protocol as `empirical_mean/` and uses no training
data, network call, or runtime dependency beyond Python's standard library.

Strength 2 was selected on the public development benchmarks, not the local
test pool. The six-budget results and caveats are in
`reports/mlp_benchmark_holdout_2026-10-22.md` in the source repository.
At budget zero it returns 0.5; with `n` matching labels it returns
`(1 + successes) / (2 + n)`.
