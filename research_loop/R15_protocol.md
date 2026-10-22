# R15 / V2-06: audit whether relative subject capability can transfer

Registered2026-10-22 before this audit. R14 retains peer gains across cohort assignments, but gains assume peers are informative about the target's prevalence. The current estimates treat different visible subjects as exchangeable. A potential missing object is **relative subject capability**, distinct from unknown benchmark base rate and item-relative difficulty.

Competing explanations: source data may support stable cross-benchmark subject ordering; alternatively subjects/configurations scarcely overlap, or rankings depend on the task so strongly that transferring them would hurt. Observational Scaling Laws (arXiv2405.10938) is a literature lead for capability structure, not permission to import its datasets or assume the same low-dimensional law applies here.

First action R15a is input-only: inspect normalized subject identities and count how many training benchmark directories contain each. Key variants are full visible configuration, normalized_name alone, and normalized_name/provider/reasoning_effort. Do not identify hidden benchmarks or infer anonymous subject IDs. Evaluate coverage only on the existing public target pairs; do not read their outcome means for this audit. Subject table presence is potential coverage, not proof of usable binary responses.

If cross-benchmark coverage is negligible, do not build an identity lookup ability estimator; return to transferable attributes or within-call relations. If sufficient bridge subjects exist, next preregister a source-only outcome aggregation and benchmark-centered relative capability diagnostic with held-out tests. No model, score or success claim follows from input overlap alone.

Operator: Audit information correspondence -> test formation/identifiability prerequisites -> only then construct a relative-ability solver. This is neither another MLP architecture search nor using validation benchmark prevalence as a prior.
