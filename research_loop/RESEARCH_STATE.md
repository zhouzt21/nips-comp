# PAIEC continuous research state

## 2026-10-22 当前：plan_5 R65–R68办结

全部真实执行与条件审计完成：R65诊断、R66十二fit、R67三refit及5242pair×3seed全外层；R67_item_segment=0.194448914，弱于R56探索分段且未过研究Gate。ADeLe等待资源，共享路线等待完整hosted合同；两项未执行效果实验不算负训练。没有后台任务、无提交，R25/fallback保持。恢复入口：[TRANSFERABLE_PRIOR_DECISION.md](TRANSFERABLE_PRIOR_DECISION.md)、[主报告](../reports/transferable_prior_mechanism_findings.md)、[执行/审计入口](../reports/transferable_prior_reproduction.md)。下方原状态不覆盖本轮结论。

## 2026-10-22 当前：plan_4补证完成

R62/R63/R64完整外层及保存概率重计分全部完成。R56_segment有探索性点估计优势0.001318527但BM/一折/CI未通过；R55 query仅7.3e−6改善；R64模板平均虽改善原R46仍略差R25。原Research Gate未通过，R25/fallback不变，无正式提交，无待运行实验。恢复见[本轮状态](HISTORICAL_POSITIVE_FOLLOWUP_STATE.md)；下方旧状态不覆盖本轮执行。

## 2026-10-22 当前：R59–R61来源阶段已执行

按`doc/plan_3.md`完成冻结权重回放、Brier精确分解、预算分段和唯一B1首标签门控，新增fit0。MLP优势来自pair均值，但选择可靠性不足；R61三池ΔALC=+0.000313793/−0.000746241/−0.000396236，未通过联合来源条件，完整外层`condition_not_met`。数值、缓存哈希及单一只读结构复核完成；无训练任务、无新隐藏/全量成绩、无提交或fallback修改。主入口：[结果与机制认识](../reports/low_budget_prior_composition_findings.md)、[评分前合同](LOW_BUDGET_PRIOR_COMPOSITION_SPEC.md)、[算子决定](LOW_BUDGET_PRIOR_OPERATOR_DECISION.md)。

## 2026-10-22 最新：R54–R58执行完成

用户 `doc/plan_2.md` 五候选已按顺序执行及审计条件。36fits保存996个epoch checkpoint；普通MLP迟发恢复停止条件成立，R55/R56/R57未通过全来源准入；R58未触发新训练。没有新outer/隐藏成绩、组合或提交，原Research Gate仍未通过。当前无后台训练；ADeLe继续等待资源。最新入口：[TRAINING_MECHANISM_V5_STATE.md](TRAINING_MECHANISM_V5_STATE.md)、[结果与机制分析](../reports/training_mechanism_v5_findings.md)。

## 2026-10-22 最新：R51/R52 来源内复现完成

依据用户 `doc/复现与组合.md` 执行的CNP/attention与amortized IRT均未通过预声明进入条件，停止outer/扩容/组合；原研究Gate未通过、fallback不变。18次真实拟合完成、28项实现检查通过。ADeLe依用户决定等待资源，仅已完成来源与dry-run输入准备。无后台训练。恢复入口为 [REPRODUCTION_V4_STATE.md](REPRODUCTION_V4_STATE.md) 和 [结果分析](../reports/reproduction_v4_findings.md)；下方状态为历史检查点。

<!-- current-v2-r29 -->
## 当前状态：R10–R31 均已完成

2026-10-22。下方原有章节是历史检查点；其中“R22 running”等文字已被本节替代。当前无研究训练进程。主报告 reports/research_loop_v2_2026-10-22.md；最新推理链 ITERATION_V2_13_20.md 与 ITERATION_V2_21_22.md。

- 独立pair三种子：R25=0.205952959，R7=0.206243741，smoothed2=0.207408367；相对均值改善0.001455407，三折改善，但远低于冻结0.005。R25相对R7有一折略退化。
- 条件cohort4三种子：R20=0.202834515，强peer=0.204959086；增量0.002124571，仍低于强对照门槛。hosted共享范围未验证，不能替代独立pair证据。
- R22真实训练扩大池未修复OOD prior；R23软subject效应无稳定增益；R24/R25分离了新增字段与正则改变；R26在线获取CI跨零；R27预算强度退化；R28对称先验仍输固定均值；R29二阶源任务协方差三折内层全部关闭。
- 原fallback保持：dist/smoothed_mean_strength2.zip，SHA256 E55EE0B267F3B345685EF072CFA511C3C017D9082256573793961EA20FEADCB9。没有晋级新模型、提交Codabench或公开发布。
- HF当前revision仍为54074453da7b9911be0f48e4c560be547bc84d21，实时检查见runs/research_loop/release_check.json。文件列表达100000项可能截断，不是仓库总文件数。没有发现新的独立benchmark发布。
- R30发现原100pairs/BM代理只覆盖737/5242合格pair。R31冻结模型，实际扩大四个被截断benchmark、复用其他完整benchmark；三种子全部合格pair R25=0.195123926，smoothed2=0.198687363，改善0.003563437。原子集逐预算回放一致。该覆盖敏感性仍保留64evaluation items上限，不是全response评分；alignment_faking占81.6%，也不是新增独立泛化证据。未替换原晋级门槛。
- 最终63项tools回归通过；R30/R31另有真实数据回放断言。没有新增正式提交包。

研究目标尚未全部完成。局部机制证据成立不等于0.005研究门槛、独立泛化与提交门槛通过。不能把已有公开集合重分折称为新盲测。
<!-- end-current-v2-r29 -->

Updated: 2026-10-22. Authority: `../../loop_1.md`, then compatible constraints in `../../loop.md`, `../../plan_1.md`, `../../plan.md`. Current scientific state is in HIGH_LEVEL_STATE.md and RECOGNITION_GRAPH.md.

## Gate and evidence boundary

- Frozen primary comparator: smoothed mean strength 2; training-only selected strong comparator uses strengths 2, 2, 4 on outer folds 0, 1, 2 (historical binomial-risk selection).
- Gate: pooled pair-equal delta ALC <= -0.005 against **both**; improve all 3 folds; benchmark-equal improves; benchmark-cluster paired bootstrap 95% upper bound < 0.
- Risk threshold fixed before new scoring: any per-benchmark ALC regression > 0.03 requires explanation and prevents promotion until resolved by a prespecified robustness check.
- Mechanism must have a controlled comparison; a public grouped-CV gate pass does not establish new-benchmark blind generalization. All 36 scoreable public outer benchmarks have already been reused.
- No new dataset release verified in prior phase. Hidden Codabench submission is not authorized. Generalization gate remains unresolved without independent evidence.
- Submission gate: portable offline inference, valid ZIP, no label leakage, training/export equivalence. Existing fallback is `dist/smoothed_mean_strength2.zip`.

## Verified starting point

Old MLP residual2 ALC 0.215339 vs smoothed2 0.208331. Prior Brier 0.275033.
Pair mean-bias squared 0.099386; centered item gain -0.001079. Negative transfer primarily occurs in budgets 0 and 1.
Previous global prior, pair-balanced sampler, and empirical Bayes rounds did not pass promotion; retain their results under `runs/mlp_failure_2026_10_22`.

## Completed round R1

Mechanism: shared hash coordinates allow cross-field interference; alternatively MLP capacity rather than hash interference causes nontransferable fitted correlations.
Controlled candidates: block-hash MLP; original-hash linear sigmoid. Original response sampling cap400, dimensions2048, inner benchmark early stopping, seed2026, loss and adaptation unchanged.
Block residual2 ALC0.212291315; linear residual2 ALC0.211938858; both lose all3 folds versus smoothed2. Gate false. Results and bootstrap in `runs/research_loop/R1/summary.json`; budget/benchmark CSV saved. Original predictions reproduced aggregate0.2153385007 exactly. No new ZIP promoted.

## Completed subsequent rounds

- R2 structured extraction: ALC0.213186511; no promotion.
- R3 local item ridge: seed0 ALC0.207545029 vs0.208330904. Three-seed verification0.206545560 vs0.207408367; delta-0.000862807, folds[-0.00128315,-0.00134151,+0.00003992], benchmark-cluster CI[-0.00178803,-0.00015037]. Small high-budget benefit, gate false.
- R4 frozen MLP hidden features:0.208315491; two folds select no local slope, no useful improvement.
- R5 episodic diagonal metric:0.207678705 vs matched fixed metric0.207694244. Learned metric gain only0.00001554; CI crosses0. Do not expand this hash metric without new information.
- R6 constant+RBF covariance:0.208546007, worsens B1/B3 and loses matched smoothing0.208185879. No promotion.

## Completed R7-R10 and v2 transition

R7 frozen MiniLM local ridge: seed0 ALC0.207180035. Frozen three-seed verification0.206243741 vs smoothed2 0.207408367; all three folds improve, delta-0.001164626, CI[-0.00219199,-0.000246395]. Less than the frozen0.005 threshold. Pretrained90.9MB and isolated dependencies onD; offline forward passed. Global HF2.2 decoder incompatibility recovered with isolated HF0.35.3; unrelated to AnySearch. No global Python package modified.

R8 semantic MLP residual2 0.212225902; same-window lexical MLP0.210770688. Both lose all folds to smoothed2. Six real weight sets and command logs retained.

R9 conditional shared evidence cohort4:0.206835842 vs same-evidence smoothed2 0.209226828; fold1 regresses and cluster CI crosses0. Organizer cohort distribution unverified, not primary-gate evidence. Cohorts1/2/4/6 all executed.

R10/v2-01 exact local signal decomposition: semantic gain0.001164626 = pair-mean0.000074945 + centered-item0.001089680. Support correspondence permutation ALC0.208339189. Item gain positive in all folds; controlled evidence supports local information use. Does not establish causal semantics or universal generalization.

Best isolated-pair research candidate by pooled ALC is R7, with uncertainty relative to lexical R3 and matched-window hash. Submission fallback unchanged. Current report `reports/research_loop_v2_2026-10-22.md` supersedes the earlier checkpoint. No process from R1-R10 remains active. New actions follow the v2 diagnosis, not a fixed old algorithm list.
