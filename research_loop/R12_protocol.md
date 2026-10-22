# R12 / V2-03: challenge the hierarchical explanation with matched shrinkage

Registered2026-10-22 before R12 scores. Alternate-explanation self-review, not external expert feedback.

R11 fixed/adaptive hierarchy beat R9 plug-in in cohort4. Adaptive minus fixed is only-0.000198 with CI crossing0. Both hierarchies also changed the effective own-prior strength relative to R9's strength2. Thus R11 does not isolate heterogeneity integration as the cause. A cheaper explanation is stronger shrinkage to the same peer estimate.

Operator: Audit load-bearing attribution -> matched coefficient control -> retain only demonstrated extra computation. Before combining hierarchy with R7, ask whether hierarchy adds anything beyond a plug-in update with the same concentration4.

Frozen conditions: R11 three folds, cohort4, seeds0/1/2, same visible-subject matching and current-budget evidence. Same kappa selected by R11 training (8/8/8). Control `p=(4*q_peer+Sown)/(4+nown)`, with `q_peer=(kappa/2+Speer)/(kappa+npeer)`. If no distinct peer exists, use smoothed2, matching hierarchy fallback. Contrast against fixed-c4 hierarchy, adaptive hierarchy, R9 alpha2 plug-in, and same-evidence smoothed2. Additionally select alpha from1/2/4/8/16 on outer-training seed73cohort4 as a stronger plug-in baseline; this is a control, not evidence of algorithmic novelty.

Interpretation: if matched plug-in explains hierarchical gains or wins, do not credit Bayesian integration. If fixed hierarchy beats matched plug-in across folds and cluster CI, joint uncertainty/evidence processing remains supported, but posterior heterogeneity specifically still requires adaptive-vs-fixed evidence. Neither conditional scenario nor repeated public-fold reuse becomes an independent generalization test.

No new network, feature or cohort tuning. Report six-budget/fold/BM deltas and choose next action based on this control.
