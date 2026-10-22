# R10 / V2-01: identify what the local predictor actually improves

Registered 2026-10-22 before running diagnostic scores. Authority: root loop_1.md. This replaces the unimplemented suggestion of a hierarchical R10; no hierarchical experiment had been run or registered.

## High-level state before

R7 semantic local ridge improves three-seed ALC by 0.001165 versus smoothed2, all three folds; its incremental gain over matched-window hash remains uncertain. R8 semantic MLP residual2 is 0.212226, worse than smoothed2 in all folds. R7 and R8 are not an architecture-controlled factorial comparison: their predictors, learning data, priors and fitting procedures differ. The evidence supports investigating information use, not declaring semantic features untransferable or MLP structurally incapable.

## Competing mechanisms and operators

H-A: allowed local labels align item geometry with correctness; gains should vanish when label-to-support correspondence is destroyed and should include a positive within-pair discrimination term.
H-B: scores improve mainly through feature-induced changes in the pair mean or clipping; item discrimination may be zero/negative despite lower total ALC.
H-C: gains are limited to a few tasks/seed choices; pooled improvement is not a broadly portable relation.

Sequence: Audit the claim `ALC gain means item discrimination` -> exact error decomposition -> correspondence-breaking control -> update which probability component should be modeled. Decomposition must precede differentiated treatment; no new network or pooling model until this result is interpreted.

## Frozen computation

Use R7's frozen per-fold ridge choices, original 737 pairs/36 benchmarks, seeds0/1/2, both semantic and same-window text-hash geometry. No refitting, no new parameter selection. Reuse saved episode kernels, labels and responses. Support permutations use 16 deterministic repetitions within each current budget prefix; they preserve label counts and geometry and never move a label across pairs or beyond its budget. These are controlled corrupted-evidence diagnostics, not acquisition strategies.

For each response-weighted evaluation set, let b be smoothed2, m=mean(p), q=mean(y):

`B(b)-B(p) = [(b-q)^2-(m-q)^2] + [2cov(p,y)-var(p)]`.

First term is pair-mean correction gain; second is centered item discrimination gain. Report the two terms and verify the identity to numerical tolerance. Also report expected Brier after randomly permuting query predictions: `mean(p^2)-2mean(p)mean(y)+mean(y^2)`. This is a response-level mathematical covariance diagnostic, not a valid item-level predictor or deployable intervention; repeated responses retain their original score weights. The collapsed prediction m is likewise diagnostic only, not an interface-compliant candidate because it uses the evaluation feature collection.

## Interpretation fixed before output

- Positive discrimination with benchmark-cluster CI upper/lower appropriate to gain and a loss under support permutation supports H-A locally; it does not prove causal semantics or universal difficulty geometry.
- Positive mean gain but negligible/negative discrimination supports H-B and redirects attention to prevalence estimation before richer item heads.
- Fold reversals or a handful of dominant benchmarks support H-C; preserve that boundary.
- If replay differs from R7_verify by >1e-7 or decomposition residual >1e-10, stop interpretation and fix implementation.
- No parameter, model or success-gate promotion may be based on this diagnostic. Original minimum improvement0.005 and all other gates remain frozen.

## Evidence and alternative-explanation review

The user references distinguish representation, access and actual use; A09 and MN-P32 motivate inspecting information source. The Transformer paper is an analogy, not evidence that NumPy ridge implements Transformer ICL. A numerical identity supplies the exact measurement here. A separate self-review must challenge clipping, unequal response multiplicity, seed pseudoreplication and the uncontrolled R7/R8 contrast; it is not external expert feedback.

Expected output: per-pair/seed/budget CSV, per-benchmark components, paired benchmark bootstrap after averaging seeds, code/data hashes and an updated recognition graph.
