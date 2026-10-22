# V2 iterations 04–08: completed evidence and judgment changes

Authority: root loop_1.md and its four local references. Recorded2026-10-22 from saved JSON; no external review or blind test is claimed. All comparisons preserve pair-equal ALC, nested budgets and current permitted labels. The36 reused public benchmarks remain development evidence. R18 is a subsequent experiment and is not included among these completed results.

## V2-04 / R13: two kinds of current evidence

**High-level claim / rivals.** R10 identifies centered item signal; R12 supports a simple peer prevalence estimator. These may encode complementary information, or the apparent gains may duplicate a mean correction. The information-use object is the prediction's mean versus centered item variation, not network size.

**Operators and downward test.** Decompose → differentiated treatment → factorial. Freeze R7 ridge and R12 peer parameters; compare base, local-only, peer-only and combined on identical cohorts and labels, three trajectory seeds. Change neither representation nor acquisition.

**Observed.** Cohort4 ALC: base0.208503347, local0.207286928, peer0.204959086, combined0.203771020. Interaction0.000028352 is near additive. Combined-minus-peer −0.001188067; three folds improve and benchmark-cluster CI excludes zero. Cohort1 exactly replays R7. Cohort6 is sensitivity, not a new primary setting.

**Upward update.** The two components have complementary value in this conditional evidence setting. This does not establish the hosted cohort distribution or pass the strong-simple improvement threshold. Next uncertainty is peer composition, which changing only item seeds does not address.

## V2-05 / R14: uncertainty in the evidence population

**Claim / rivals.** R13 benefit may depend on which subjects happen to share a cohort, rather than general item adaptation.

**Operators.** Audit statistical unit → composition stress. Three frozen cohort assignments × three trajectory seeds; benchmark-cluster uncertainty after averaging the nine correlated draws within benchmark.

**Observed.** Combined0.203217100, peer0.204392535, base0.208264899. Difference against base−0.005047799, against peer−0.001175435. All folds improve; worst benchmark regression versus base remains about+0.02605 on braveguard.

**Upward update.** Conditional complementarity is reproducible, but clearing0.005 against the weaker comparator alone is insufficient. Peer identity and subject heterogeneity are distinct from trajectory noise. No promotion. Source-only subject ordering becomes a discriminator, rather than trying another pooling strength.

## V2-06 / R15: does an historical ability axis transfer?

**Claim / rivals.** Sparse source observations may still support a benchmark-centered subject ability correction, or subject identity/configuration and ranking may fail to transfer.

**Operators.** Audit accessible identity → remove benchmark intercept nuisance → fit a fixed regularized scalar ability. Full visible subject configurations have zero cross-benchmark overlap; a declared name/provider/reasoning key has some overlap. Fit only each outer fold's training benchmarks; require two source benchmarks for using a coefficient. Current benchmark ID is never a memorized feature.

**Observed.** Actual supported targets188/737. Only6/16 rankable benchmarks have positive covariance with the source ability. Corrected peer0.204982696 loses uncorrected peer0.204959086; two folds worsen. Sparse coverage and heterogeneous rankings both remain relevant.

**Upward update.** Reject expansion of this particular scalar ability lookup. This does not refute multi-dimensional ability models. A primary observational-scaling paper uses a much denser, more standardized matrix than this one; its evidence does not justify assuming a transferable one-dimensional axis here.

## V2-07 / R16: audit what outcome1 measures

**Claim / rivals.** The preceding ranking failure may partly reflect incompatible label semantics, rather than only feature capacity.

**Operators.** Audit measurement → change source formation while freezing model and target evaluation. Metadata:54 higher-direction,4 lower-direction,3 unordered binary benchmarks. Outcome1 is not universally correctness. Keep all raw target labels; train the same ability estimator on higher-direction source benchmarks only.

**Observed.** Filtered ability0.204981901 remains worse than peer0.204959086. Historical MLP B0 on higher-direction outer tasks is still0.274379, worse than0.25. Filtering is not a sufficient repair.

**Upward update.** Separate the numerical outcome prediction task from interpreting scores as one common competence trait. The audit invalidates a universal semantic interpretation, not the already computed raw-outcome Brier arithmetic. Official/public-label reconciliation remains open. Stop expanding this scalar correction and stop treating label direction as the sole root cause.

## V2-08 / R17: can reliability weighting alone close the gap?

**Claim / rivals.** R7 has a real small item signal; perhaps selectively shrinking it could attain the frozen0.005 improvement. Alternatively its available prediction direction is too weak even with perfect coefficient choices.

**Operators.** Oracle bound → remove an entire repair family. For b=smoothed2 and d=R7−b, minimize observed squared error of b+γd over γ∈[0,1], separately for each pair and budget. This explicitly uses evaluation labels and is a diagnostic, never a deployable estimator or hyperparameter selector.

**Observed.** Base0.207408367; actualR7 0.206243741; pair-budget oracle0.205276891. Maximum improvement0.002131476 remains below0.005. Direct quadratic-loss identities and replay checks pass.

**Upward update.** No attenuation-only repair of these frozen predictions can meet the gate on this replay. This is not an information-theoretic lower bound for all predictors, amplification, different kernels or additional information. Move from coefficient choice to information retained by the interface. R18 audits token retention before changing representation.

## Self-audit and gates

This is an alternate-explanation self-review, not an independent reviewer report. Public folds were reused adaptively; clustered intervals do not correct all research selection. Conditional cohort experiments cannot replace the original isolated-pair gate. Research gate remains false; independent-generalization evidence unresolved; no new submission promoted. The existing smoothed2 ZIP remains the verified fallback.

Exact fold/benchmark/budget results and source snapshots: runs/research_loop/R13, R14, R15, R16, R17. Individual pre-experiment protocols specify selection scope and controls. Commands: python -m mlp.run_component_round; python -m mlp.stress_cohort_composition; python -m mlp.run_ability_round; python -m mlp.audit_target_semantics; python -m mlp.oracle_shrinkage_audit.
