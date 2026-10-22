# PAIEC MLP: all-public-data refit

This archive contains the same offline MLP predictor and residual correction
strength 4 as the benchmark-held-out `C1-internal` research candidate. Its
weights were refitted for exactly two epochs on the complete eligible public
measurement-db core release, revision
`54074453da7b9911be0f48e4c560be547bc84d21`, with at most 400 sampled
responses per benchmark. There are 22,771 sampled training responses from
2,322 sampled subject–benchmark groups. This fixed epoch count came from the
internal-validation candidate; this refit itself had no early stopping.

Because **all public benchmarks** were used for training, no local
benchmark-held-out score applies to these exact weights. The source repository
contains the separate held-out experiments and their limitations in
`reports/mlp_benchmark_holdout_2026-10-22.md`. Those experiments show that the
current MLP is not consistently better than a strength-2 smoothed mean across
three additional benchmark-grouped folds. Treat this full-data archive as an
experimental submission candidate, not as a proven stronger method.

Inference uses only NumPy, the adjacent `weights.npz`, supplied subject/item
attributes, and the allowed matching revealed labels. No runtime network
access, local training data, or credential is required.
