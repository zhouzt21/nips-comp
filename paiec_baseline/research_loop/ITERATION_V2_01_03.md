# v2 iterations01–03: actual upward/downward records

## V2-01 / R10

**Before:** R7 slightly improves ALC, R8 supervised semantic MLP fails. Possible explanations include item relations, mean correction and clipping. The R7/R8 contrast changes multiple factors.

**Hypotheses / downward:** separate pair mean from centered item error exactly; break support correspondence while preserving current-budget counts and geometry. Frozen16 permutations, three seeds, original folds.

**Observed:** R7 gain0.001164626 consists of mean0.000074945 and item0.001089680. All folds have positive item gain. Support permutation ALC0.208339 vs true0.206244. Replay passes and maximum identity residual3.1e-16.

**Upward:** reject the interpretation that all gain is a mean/calibration artifact. Do not infer global zero-label impossibility or universal causal semantics. Local evidence can make item geometry useful.

**AnySearch:** inspected methods/results of Chan et al. arXiv2210.05675; source distinction is an explanatory analogy only. Wu et al. arXiv1903.01689 constrains claims that representation alignment suffices; its assumptions differ from this task. Exact decomposition is algebra, not a borrowed theorem.

**Operator / next:** audit -> measurement -> correspondence intervention. Next target: low-budget prevalence, where local centered slopes have no n=1 effect.

**Gate:** controlled local mechanism supported; total gain<0.005; independent evidence absent.

## V2-02 / R11

**Before:** R9 peer pooling helps low budgets but regresses in one fold. Competing explanations are subject heterogeneity, uncertain peer mean and cohort composition.

**Downward:** Beta population/subject hierarchy, fixedc4 versus a discrete concentration prior1/4/16; train-only kappa2/8; nopeer smoothed2 fallback. Same inputs as R9.128/256quadrature, complement symmetry, ordering and label-scope checks passed.

**Observed:** cohort4 three-seed adaptive0.204838 vs smoothed2 0.208503. Adaptive vs fixed improves0.000221 but CI crosses0. Exact outcome of original preregistered first run and subsequent frozen-seed verification are preserved separately.

**Upward:** conditional shared evidence is promising; adaptive heterogeneity is not established. Alternate explanation: changing own-prior strength from2 to4 may explain R9-to-R11 improvement.

**Literature:** official Stan case study (Carpenter2016) supplies the hierarchical likelihood and warns about weakly constrained population spread. It cannot establish PAIEC gains.

**Operator / next:** separate probability levels -> condition -> integrate; then audit attribution before adding more machinery.

**Gate:** conditional gain0.003665<0.005. Hosted cohorts unverified; isolated fallback identical to baseline.

## V2-03 / R12

**Before:** hierarchical explanation confounded by coefficient change.

**Downward:** same kappa/evidence/three seeds; compare fixedc4 hierarchy with alpha4 plug-in; add training-only alpha selection as stronger simple control. All three inner folds select4.

**Observed:** fixed hierarchy0.205059; matched plug-in0.204959; adaptive hierarchy0.204838. Matched-control differences are small and cluster intervals cross0. Additional Bayesian machinery has no demonstrated practical advantage.

**Upward:** change the claimed mechanism from hierarchical uncertainty integration to peer evidence plus adequate shrinkage. Keep the lighter estimator. This is a negative attribution result despite positive absolute scores.

**Operator / next:** audit -> matched control -> simplify -> factorial composition (R13). Item slope and prevalence have separate evidence; test their interaction without adding tuned mixing coefficients.

**Gate:** no promoted method. Reused public folds and unverified sharing assumptions remain explicit.

## Alternate-explanation self-review

This is internal adversarial reasoning, not feedback from an external expert. (1) R7/R8 are not a causal factorial: retain that limit. (2) R10 mean-only/query permutation use the evaluation feature/prediction distribution: mark diagnostics, not submissions. (3) Response multiplicity remains in scores. (4) Seeds/permutations are averaged within benchmarks before cluster bootstrap, not treated as independent tasks. (5) R11 concentration is a model posterior, not recovered true subject variance. (6) R12 shows the importance of a strength-matched simple baseline. (7) All current shared runs fix cohort seed193; changing item-split seeds does not test cohort-composition uncertainty. R13 must therefore not claim robustness to all sharing patterns.
