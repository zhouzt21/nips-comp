# PAIEC-LOOP-R4: frozen MLP representation with a new local head

Registered2026-10-22 before R4 scoring.

R3 three-seed ΔALC=-0.000862807 vs smoothed2, two folds improve, third+0.0000399. Benefit concentrated at15/31; gate not met. R2 completed0.213186511 versus0.208330903. Global prior preprocessing alone is insufficient.

Question: can an MLP whose output prior transfers poorly still provide useful hidden features for support-conditioned prediction? Motivated by L4 and L5 (Chen et al., arXiv1904.04232 sections3.1-3.2), not a claim that image few-shot results transfer to PAIEC.

Primary: unit-normalized64-dimensional penultimate activations of the original fold-specific MLP; same centered local ridge and smoothed2 intercept as R3. Matched control: unit-normalized full2048 input vectors including subject features, same local algorithm and selection. This matched comparison isolates the learned nonlinear map. R3 item-only model is an additional historical anchor, not the only control.

Use only the original MLP's internal validation benchmarks to select ridge: these benchmarks did not fit gradients, though they were used for early stopping. Inner seed73,max25pairs/BM,64eval items. Ridge grid off,10,1,0.1,0.01, ties favor earlier option. Both control and hidden use exactly this grid. This pool can be small; report counts and do not claim broad inner coverage.

Finish all3 inner selections before outer scoring. Outer seed0,existing100pairs/BM,64evalitems; assert baseline replay equality. Weights trained/early-stopped on no outer evaluation benchmark. No retraining, no global prior mixture. New candidate evaluated against frozen and training-selected means, matched raw-full local model, R3 and old original residual.

Gate unchanged. If hidden fails relative to matched raw, stop reusing that supervised embedding as the presumed solution; the next high-information alternative is directly learning a small representation against the episodic adaptation loss. If hidden succeeds only at31, do not claim early-budget value.
