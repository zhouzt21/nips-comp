# PAIEC-LOOP-R5: directly learn a regularized episodic metric

Registered2026-10-22 before training or outer scoring. R4 inner selection disabled hidden slopes in folds0/2 (fold1 ridge10); its all-fold superiority gate is therefore already impossible. Complete R4 output nonetheless.

R3 established a small reproducible local-slope benefit mainly at15/31; fixed supervised hidden features have not supplied a reliable next step. Mechanism from L4: optimize features through the adaptation solver against held-out episode queries. Start with a diagonal metric rather than a larger MLP.

`z(x)=normalize(x * exp(log_weight/2))`, 2048 trainable log weights, initialized0. Use same full input hash features as R4 matched control. Center support, ridge1, smoothed2 intercept; all six-budget weights fixed. B0 andB1 have zero metric gradient. Loss is official budget-weighted query Brier plus `0.01*mean(log_weight^2)`; clip log weights to[-2,2] after steps.

Outer training and inner-validation directories follow each old fold weight's metadata. Use no outer benchmark for gradient, early stopping or selection. Training episode seed73,max25pairs/BM,31support and up to64query items, all responses preserved. Each eligible sampled pair equal in optimization. All episode labels originate in legal public training benchmarks. This is more data than old cap400 training; do not attribute old-MLP differences solely to the objective.

Adam lr0.01, gradient accumulation16episodes, seed2026, max20epochs, inner ALC patience4. Include epoch0 as an eligible checkpoint. Fix all settings; no hyperparameter sweep. Export NumPy metric, check torch/NumPy solver prediction agreement on an inner episode. Matched control: identical solver and features with log weights fixed0 and ridge1. This comparison specifically tests whether learning metric weights improves an existing local method.

Outer seed0,max100pairs/BM,64queryitems. Compare to matched fixed metric, R3, smoothed2, training/inner-selected statistical baselines and original MLP. Gate unchanged. If learning fails, stop expanding this hash-based metric and investigate missing semantic information or the irreducible early-budget limitation; do not add more layers by default.
