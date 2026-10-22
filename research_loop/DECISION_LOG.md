# Decisions

<!-- current-v2-r29 -->
## 当前模型决策（R29完成后）

不晋级新方法。R25是独立pair三种子最佳点估计；原代理改善0.00146，真实扩展至5242合格pair后改善0.00356，均不足冻结0.005；R20仅在条件共享场景相对强peer改善0.00212。保留原smoothed2 ZIP。负结果全保留；不能用已复用benchmark的新seed或扩大pair数宣称盲测。下一待论证诊断是evaluation item上限与重复响应噪声。
<!-- end-current-v2-r29 -->

- V2 transition2026-10-22: root loop_1.md now authoritative. Read the four local suffix-free reference files and targeted relation/operator sections. R8 had already completed when the user redirected; collect rather than rerun. Defer unimplemented hierarchy suggestion, register R10 measurement first.
- R10: gain decomposes into centered-item0.00108968 and mean0.00007495. Preserved support counts but broken item-label correspondence loses gains. Update H1: current representations do contain usable local item signal; not enough evidence to call global-zero-label information absent. Keep core fallback and all gates. The appropriate next object is low-budget prevalence, separately from high-budget item discrimination.

- 2026-10-22: Continue under loop.md without confirmation. Preserve all historical artifacts and fallback. AnySearch available; search/extract succeeded. A mistaken unsupported `--output` argument was corrected to shell file output, not an API failure.
- R1: Separate field hashing from linear capacity control. Do not combine with previous pair-balanced sampler, new text embeddings, or adaptation tuning. Official rules rechecked via AnySearch: six budgets and pair-equal ALC unchanged; final deadline October30 23:59 AoE.
- R1 completed: neither candidate beats smoothed2. Block improves old MLP in3/3 folds, but remains worse than smoothed2 in3/3; no promotion. Linear lowers pooled old-model ALC but loses one old-model fold and all baseline folds. R2 isolates structural text extraction, using input audit evidence rather than an arbitrary larger network.
- R2 fold0 fails gate, remaining folds continue. Register R3 local ridge diagnostic. This changes the source of item-specific evidence: fit slopes only from current allowed labels, while retaining the frozen smoothed mean intercept. R3 hyperparameters and a stronger mean baseline are selected strictly from outer-training episodes before outer evaluation.
- R3 completed and verified3seeds: small high-budget benefit persists; third fold not improved, minimum effect gate missed. R4 tests frozen hidden representations before enlarging a model; two inner folds turn them off.
- R5 directly optimizes a regularized local metric; almost no benefit over identical fixed metric. R6 addresses the n=1 degeneracy through input-dependent covariance; it worsens early-budget Brier and does not beat matched smoothing. These negative results stop those particular expansion routes, not the overall research loop.
- R7 tests a genuinely different information representation, frozen pretrained MiniLM, with the exact same retained text window given to a lexical control. Model and temporary dependencies are isolated on D:. AnySearch model-card retrieval succeeded; a separate HF2.2 transport decoder error was recovered using isolated HF0.35.3. Offline forward shape(2,7,384) validated. No competition-specific data outside the allowed corpus used.
