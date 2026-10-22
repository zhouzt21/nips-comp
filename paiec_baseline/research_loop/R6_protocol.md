# PAIEC-LOOP-R6: item covariance that can operate with one label

Registered2026-10-22 before R6 scoring. R5 ALC0.207678705 versus fixed-metric0.207694244; difference-0.0000155, inconsistent by fold and CI crosses0. No evidence for larger hash metric learning. R3's persistent benefit is predominantly at15/31. Its centered slope is identically0 for n=1.

L6 (ALPaCA, arXiv1807.08912 sections2-4) motivates separating an uncertain prior from a conditional local covariance. This experiment is an approximate kernel regression diagnostic, **not an exact Bernoulli Bayesian posterior and not an ALPaCA reproduction**.

Retain R3 unit-normalized item-only hash vectors and the same cached episodes. Let R=exp(-(1-cosine)/0.25), a fixed RBF kernel (length scale0.5 on unit vectors). Define K=11'+eta R, query cross-kernel k=1+eta Rquery, observation regularizer2I. Predict `clip(0.5 + k(K+2I)^(-1)(y-0.5),0,1)`. At eta0 it exactly equals smoothed mean strength2. At one label, item similarity can change the evidence weight; no new global success prior is fitted.

Select eta only from[0,0.25,1,4] using the same R3 outer-training episodes (seed73,25pairs/BM,64queryitems), minimizing pair-equal ALC. All3 selections precede outer scoring. No kernel-width tuning.

Matched controls: (a) the same selected eta with an item-independent covariance (support noise eta I and zero query cross-covariance), exactly smoothed mean strength(2+eta); (b) independently training-selected smoothing from the same eta-derived strengths; (c) frozen strength2 and previous training-selected means. This distinguishes content-related covariance from merely changing the amount of shrinkage.

Outer same seed0,737pairs,36benchmarks,64queryitems. Gate unchanged. Report B1 explicitly. If covariance is disabled or fails the matched shrinkage control, stop regarding raw lexical covariance as a sufficient early-budget fix; reconsider semantic representation or the cross-benchmark prior rather than searching more fixed mixes.
