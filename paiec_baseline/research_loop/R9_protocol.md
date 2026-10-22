# PAIEC-LOOP-R9: conditional shared-evidence protocol diagnostic

Registered2026-10-22 before scores. This is an **additional evidence-availability scenario**, not a replacement for the frozen isolated-pair gate.

Official page says `labeled` contains entries across sampled subject–benchmark pairs, budgets are per pair, and formative feedback samples at most1000unique subject-item pairs. Existing replay intentionally supplies only the current pair. The starting kit could not be retrieved: AnySearch extraction failed and direct Codabench API access had an SSL EOF; no runtime implementation has been verified. Therefore exact hosted sharing/cohort distribution is unresolved.

Input audit also found visible subject-configuration duplicates in ceval, devbench and emoji_attack (35excess configurations across original training pool). Interface matching must use the complete visible subject dictionary; private subject_id must not be used by predict to separate indistinguishable inputs.

## Scenario construction

Keep original outer folds, targetpairs, acquisition order, six budgets and evaluation responses. Within each benchmark, partition sampled pairs into cohorts independently of outcomes using fixed seed193. Primary cohort size4; stress sizes1,2,6. Each pair receives its own n-label prefix and the same-budget prefixes of other cohort pairs. No other benchmark labels or evaluation responses enter predictors. Max6pairs each with at most128distinct acquisition/evaluation items gives≤768subject-item pairs, within the announced1000 cap for this controlled scenario. This does not claim the organizer samples these cohorts.

All methods receive identical evidence. Baseline means match full visible configuration and benchmark, so identical public configurations can legally contribute more than nmatching labels. Recompute baseline scores under this scenario; do not compare a pooled candidate only to an artificially evidence-deprived baseline. Size1 must reproduce original smoothed2 scores.

## Candidate and selection

For same-benchmark evidence, separate matching-subject labels `(s_own,n_own)` and other-visible-subject labels `(s_peer,n_peer)`. Define `q_peer=(kappa/2+s_peer)/(kappa+n_peer)` and `p=(2*q_peer+s_own)/(2+n_own)`. This is a regularized plug-in prior; it is not claimed to be an exact hierarchical posterior. Missing peer evidence gives exactly smoothed2. Other benchmark labels have no effect.

Select kappa from2,8,32 using outer-training benchmarks only, seed73,max25pairs/BM,cohort4. Select strong same-subject smoothing alpha from1,2,4,8,16 using the exact same shared evidence. All3 selections precede outer scoring. Freeze selections for cohort sizes1/2/6; no scenario-specific retuning.

Report per-budget,fold,benchmark, paired bootstrap and visible-config collisions. Even a conditional statistical gate pass cannot resolve actual shared-evidence availability or independent-generalization gates. Preserve all prior isolated-pair results and fallback ZIP. Promote no new submission solely from an unverified context assumption.
