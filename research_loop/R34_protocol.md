# R34: support-only evidence for local correction magnitude

Frozen after R33, before any R34 source selection or outer scores. R33 max4 oracle gain=0.00938759 versus actual0.00356344; attenuation oracle gain=0.00415460. Oracle is optimistic. The question is whether allowed support labels can identify useful amplification without access to evaluation outcomes.

**Operator:** Reframe evidence use -> Adapt by Conditioning -> matched-coefficient control. The new object is the magnitude of the existing local direction, not a new encoder/feature/ridge grid. Preserve R25 feature blocks and source-selected penalties.

For n>=3 support rows, form the ordinary ridge smoother with an unpenalized intercept: Xc=X-mean(X), K=Xc Xc^T, H=11^T/n+K(K+I)^-1. Exact leave-one-out standard ridge prediction is y_i-(y_i-(Hy)_i)/(1-H_ii). Remove its leave-one-out raw label mean and add the Beta(1,1) smoothed leave-one-out mean b_i=(1+sum(y)-y_i)/(n+1). Clip, then subtract b_i to obtain d_i. Define

`c = clip((sum d_i*(y_i-b_i) + tau)/(sum d_i^2 + tau), 0, 4)`.

This minimizes an un-clipped one-dimensional squared loss plus tau*(c-1)^2, then clips the coefficient. Final prediction is clip(b+c*(p_R25-b),0,1). n=0/1 retains R25 exactly. tau=infinity returns c=1; constant labels/zero corrections cannot create fictitious directional evidence. The finite-support mean correction is derived here, not claimed as a result of the retrieved ridge-CV paper. Verify against explicit leave-one-out refits before real scoring.

**Source-only selection:** per outer fold, use the unchanged R25 source benchmarks, seed73, max25pairs/BM, historical64queryitems. Select tau in [infinity,0.1,1,10] by pair-equal source ALC. Fixed-scale control selects c in [0.5,1,2,4] on exactly those source episodes. Freeze all three folds' selections before outer scoring. No outer coefficient/oracle is reused.

**Outer paired experiment:** all5242eligiblepairs, seeds0/1/2, exact historical64queryitems/all responses, same support draws. Methods: adaptive scale (primary), source-selected fixed scale, R25, smoothed2, training-selected smoothing, empirical mean. Mechanism control: use four deterministic permutations of support label-feature correspondence only to estimate c, while leaving the actual R25 query direction and label counts fixed; average their losses, not their predictions. Report the six budgets, all folds, benchmark-equal, paired benchmark-cluster CI, worst regressions, and scale distributions. Replay R25/baselines against R31.

**Decision:** require the unchanged research gate and inspect adaptive-vs-fixed to attribute value to local scale estimation. If tau=infinity, or adaptive does not beat fixed, do not claim an adaptive mechanism. Do not sweep extra tau/coefficient values after outer results. If a candidate appears to pass the gate on64items, perform full-query validation before freezing a submission; R32 cap sensitivity of old R25 does not automatically transfer to a new amplified predictor. If adaptive fails and fixed also fails, return upward to the gap between oracle and support identifiability, not a bigger coefficient grid.
