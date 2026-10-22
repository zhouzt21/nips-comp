# R17 / V2-08: an oracle bound on repairing the existing predictor by shrinkage

Registered2026-10-22 before oracle scores. R10 local geometry has genuine but small discrimination; R14's incremental gain over strong peer estimation remains~0.0012. R15/R16 scalar capability and direction filtering fail. Before another coefficient change, determine whether even an unrealistically informed shrinkage rule could meet the0.005 isolated-pair gate.

Frozen isolated R7 predictors, original folds, seeds0/1/2. Let b=smoothed2 and d=p_R7-b. For eachpair/budget define U=mean[d(y-b)], V=mean[d²]. Within the family `p_gamma=b+gamma*d, gamma in[0,1]`, the oracle using **evaluation labels** chooses gamma=clip(U/V,0,1), with gamma0 ifV=0. Brier is `Bbase-2gamma*U+gamma²V`. Predictions remain valid probabilities by convexity.

Also compute a single oracle gamma perbenchmark/budget from pair-equal averaged U,V. Neither oracle is deployable or eligible for model selection. Perpair oracle is an exact empirical lower risk bound for this specific convex-shrinkage family, not an information-theoretic bound on all predictors, alternative kernels, amplification or nonlinear calibration. It can rule out coefficient attenuation as sufficient for the frozen gate on these data.

If even perpair oracle gain<0.005, stop attempts to obtain the missing effect through shrinkage of current R7 predictions; investigate new discriminative information or task-conditioned representation. If oracle gain is large, investigate whether reliability can be estimated from permitted support labels without an evaluation-label oracle. Verify algebra against direct predictions and replay R7; bootstrap uses benchmarks afterseed averaging. Do not report the oracle as a candidate score.
