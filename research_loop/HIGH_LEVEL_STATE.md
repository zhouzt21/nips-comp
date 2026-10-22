# High-level problem state — loop_1.md

<!-- current-v3 -->
## Loop v3 当前状态

Authority: `doc/loop_2.md`，及四份相同目录的研究参考材料；取代旧loop_1的控制流程，保持所有已冻结实验纪律。R1–R31历史结果保留。

R32-R50 completed/audited. Current six-field decisions are in ITERATION_V3.md. No job is implied to be running by these files. Gate remains unmet; fallback unchanged. R48 uses extra labels, R49 failed source admission, pixel runtime contract unverified.

当前六字段研究判断与下一动作以 ITERATION_V3.md 为准。以下R32段落是已完成的测量审计摘要，不是仍待执行的任务。

全5242pair中5224受64-item限制；完整2179077queryitems/seed，原335247。R31收益77%来自预算15/31；B0/B1相同是当前算法的构造事实，不能直接推论输入不含冷启动信息。重复响应的经验within-item方差在两个预测器Brier差异中精确抵消，不能直接用该方差解释全部未达标差距。

R32先按benchmark×评价池大小×重复响应分层，从计数预先选择pair；每个被选pair完整评价pool，三个原种子、同一acquisition labels。分层外推与实际全pair评分分开。code/protocol/selection/log均在runs/research_loop/R32及research_loop/R32*。缺失语义向量只用原模型离线补D盘缓存。

研究Gate尚未通过；fallback不变；未正式提交。恢复时先检查最新轮次的进程、日志与summary，不要重复启动已完成的R32。日志存在不代表仍在后台运行，单benchmark结果不代表全轮完成。R33–35的幅度估计、R36–38的条件共享、R39的似然求解及当前输入分组研究的详细结论均见ITERATION_V3.md；历史v2段落不覆盖v3控制流程。

R32c已完成：主导benchmark全部4278pairs，三个原种子，全部evaluation items/responses。R25 full ALC=0.19066693872042262；smoothed2=0.1947777824570729。改善从capped约0.004122变为full约0.004111。所有旧capped逐pair/seed/budget回放通过。仅此benchmark不能替代36benchmark汇总；重复响应经验方差不是不可约噪声下界。

R32分层测量结论（不是全5242pair完整评分）：

{
  "hybrid_cap_effect": 1.8690490164230804e-05,
  "approx_design_ci95": [
    -0.00016079132519827036,
    0.000198172305526732
  ],
  "cap_corrected_gain_estimate": 0.0035447463018261067,
  "remaining_gain_to_gate_estimate": 0.0014552536981738934
}

<!-- end-current-v3 -->

<!-- current-v2-r29 -->
## 下一层认识与尚未解决的问题

已测试的证据支持“当前少量标签下的item对应关系”比“跨任务固定正确率映射”更可靠，但不是普遍的架构定理。扩大样本覆盖、文本窗口、先验分布形状、固定非线性核、源任务二阶协方差均未带来足够增益。它们排除的是具体实现/假设，不能推论所有特征学习无效。

R30/R31现已完成pair覆盖审计与真实扩展。采样上限显著改变公开集合权重，但完整pair改善0.00356仍不达到门槛。下一开放问题是64-item评价样本及同一item重复response噪声与可被item输入解释的差异；尚未执行，不得当作已发现的限制。只有诊断证明缺口足以影响目标，才进入新特征或学习机制。不可继续原先已失败的rank/rho、固定mix、窗口或数据量网格。

独立泛化验收需要未参与设计的数据；目前没有已核实可用的新发布。该外部证据缺口不意味着所有本地研究都无法继续，也不能据此声称目标已经完成。
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

## Latest transition through R21; R22 running

R20three-seed sharedcurrentitemslopes0.202834515 versusown-selected0.203917943, originalown0.203771020, peer0.204959086, base0.208503347. Peercorrespondencepermutation0.204833741 removes added information gain. Allfolds improve; strongpeerincrement0.002125stillmissesgate. R21degree4has smallseed0increment−0.000384versuslinear, onefoldworse andCIcrosses0. Do not scan moredegrees. R22 returns to the unresolved source-data coverage question for the original supervisedMLP: pool400to2000while fixing validation400 and pervalidationcycle trainingexposure/steps toR8. No assumption that moredata mustsolve shift; check actualexposure and heldoutprior error before choosing further action.

R18 controlled256-token semantic0.207230 versus128semantic0.207180; lexical2560.207818 versus1280.207755. Both fail to improve. Input-only omission does not establish useful omitted signal; stop enlarging this window. New128implementation exactly matches original forward features on the declared check. R19three-seed exact-peer-item transfer0.204403 versuspeer0.204959; shuffledcorrespondence0.205501. Small supported conditional signal, but belowR13combined0.203771 and far below0.005increment. R20 now tests whether sharing CURRENT centered item evidence across subjects improves semantic slopes, holding representation and peer means fixed. No new model download or ability prior.

R13/R14 verify complementary peer-mean and local-item components across three cohort assignments and three trajectories. Conditionalcohort4 combined0.203217 vsbase0.208265 and strongsimplepeer0.204393. Gain overpeer0.001175 misses the frozen meaningful-effect gate; isolated behavior is still R7. No candidate promoted.

R15 learns source-only benchmark-centered scalar abilities. Actual multiple-benchmark coverage188/737; only6of16rankablebenchmarks show positive covariance. Correction0.204983 loses simplepeer0.204959. R16directionfilter0.204982 does not fix this. Stop scalar ability lookup expansion. Sparse identity/config overlap and task-specific rankings remain distinct problems.

Metadata54higher/4lower/3unordered shows binary1 is not uniformly correctness or a common capability trait. Preserve all original labels and raw Brier results; see reports/label_semantics_audit_2026-10-22.md. Website/public-ingestion semantic reconciliation remains unresolved. Higher-only tasks still show oldMLP B0=0.274379, so label semantics is not the entire explanation.

R17 oracle: even evaluation-label-optimal pairwise convex shrinkage of frozen R7 towardsbase yields0.205277 versusbase0.207408, maximumgain0.002131<0.005. This rules out that repair family on these observations, not all local prediction. Current action: audit information discarded by encoding (text window and modality/interface), then choose a representation experiment only if this identifies a concrete missing input component. Do not restart hierarchy/ability/mix sweeps.

Updated2026-10-22. v2 authority supersedes loop.md; plan_1/plan retain compatible constraints. Four local references have suffix-free names: `Research Down–Up Loop.md`, `High-level 认识维度与相互关系.md`, `Model-Nature_High-Level_Recognition_Paper_List_v1_1.md`, `常见的组合思路.md`. Their paper summaries are leads, not verified primary scientific evidence.

## H0 -> H1: the current reframing

Old question: improve a supervised MLP prior for unseen benchmarks.

New evidence: R1/R2/R8 alterations reduce some old-MLP error but remain worse than smoothed2. R3/R7 local predictors have small positive gains with fixed zero-label predictions. R7 semantic three-seed ALC0.206244 vs0.207408; R8 semantic MLP residual2 ALC0.212226. No outcome establishes that all neural models fail, that inputs lack all information, or that semantic embeddings necessarily help.

Revised object: which components of `P(Y=1 | visible subject, item, currently permitted evidence)` are transferable, and which become estimable only after local evidence? Distinguish pair prevalence from centered item variation. Measure the positive local signal before inventing another solver.

## Ranked uncertainties

1. **Supported:** prior mean error and small-budget negative transfer in the historical MLP. **Open cause:** data coverage, conditional shift, learned shortcuts, or representation.
2. **Locally supported:** R7 local gain is mainly item discrimination; R10 label correspondence controls remove the gain. This is stronger than a score-only observation, still not a universal semantic mechanism.
3. **Protocol boundary:** labels may span sampled pairs; R9 shows conditional pooling benefit, largely on fold0. Actual hosted cohort distribution remains unverified. This warrants a separate information-access branch, not changing the primary isolated-pair gate.
4. **Not established:** information-theoretic inability of zero-label prediction on the actual benchmark population. General arbitrary-shift counterexamples cannot prove this empirical limit.

## Research controller

R10 completed: centered item gain0.001089680, mean gain0.000074945, total0.001164626. All three folds have positive centered gain; support permutation removes it. This supports local information use, and weakens mean-only/clipping explanations for the observed net gain. It does not establish that pretrained semantics is superior to all lexical representations.

Current mode: next upward/downward transition. High-budget item discrimination is a real but small component; low-budget prevalence remains a separate target. R9 suggests peers can inform it but also hurt some benchmarks. Before richer combined models, investigate conditional pooling and heterogeneity with matched evidence. Exact hosted scope still remains open. R7/R8 are useful contrasting observations but not a clean causal experiment of global versus local learning.

R11/R12 update: three-seed hierarchy0.204838, matched simple peer0.204959, same-evidence smoothed2 0.208503. Fixed hierarchy0.205059 fails to beat strength-matched plug-in. Additional hierarchy computation is not supported. This changes the search space: stop hierarchy expansion; use the simple mean estimator for a factorial test with the independently supported local slope (R13). Current conditional evidence need and isolated-pair performance gate remain separate.

Next branch is result-dependent: centered gain supports studying task-conditioned similarity; mean-only gain favors prevalence estimation; concentrated/unstable gains require support and domain diagnostics. Do not jump directly to larger architectures.

Fallback unchanged: smoothed_mean_strength2.zip. Mechanism gate partial, research gate false, independent-generalization gate unresolved, new semantic submission gate untested.
