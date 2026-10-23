# High-level problem state — loop_1.md

## 2026-10-23 当前：plan_10 R80/R81完成，R82/R83未触发

R80去重191pairs/35BM×3item seeds，half/LOIO/grouped5fold在B15/B31完成。LOIO减小fit-size失配与风险MAE，但符号BA .537646/.581774，相对half −.018586/+.007418、CI均跨零；83%–90%的请求条件区间跨零，名义有害组未被query证实。`sharing_sign_not_identifiable`仅限定本轮方法/预算，不是信息论不可能性。按事前条件停止R82/R83，没有新校准器、门控或策略外层。

R81完整5242×3×6/5cohort重放完成。partial−pool平均−.000179972、CI全负；partial−point −.000017538、partial−exact-cost −.000192802，两CI跨零、BM-equal和去alignment均略退化。point保留约90.25%的partial−pool点估计收益；方向翻转存在，但独立效用未证实。25BM真实偏移有弱重复cosine .0747/.1292（23BM可估），留出风险改善CI跨零；合成异质世界partial胜pool，却未稳胜成本控制，B31差于own。支持收缩/逐item幅度整形的重要性，未证明真实Subject方向恢复主导。

主Partial8 ALC .190547856/B31正确率79.1862%保持；dave对R25仍退化+.006632866，收益仍集中alignment。不晋级、不改fallback，固定Partial8保持conditional_simulation，Hosted继续暂停。全部预测与机制实验结束，无后台模型运行；独立重计分/raw clip/风险分解误差<1e−15，合成参数/标签/概率重建误差0。

入口：[Q1–Q5及下一步](PARTIAL8_NEXT_DECISION.md)、[R80](../reports/partial8_identifiability_findings.md)、[R81](../reports/partial8_mechanism_findings.md)、[指标/正确率/成本/资源](../reports/partial8_decision_tables.md)、[复现入口](../reports/partial8_next_reproduction.md)。任务`D:/todo/tasks/graph/nips-comp.partial8-identifiability-2026-10-23.md`。以下“当前”为历史截点。

## 2026-10-23 当前：plan_9 R77/R78与条件R79来源阶段完成

R77补齐7502 partial，并完成五冻结cohort与主cohort压力评价：全部5242pairs×3item seeds×六预算。partial在5/5组成中优于同cohort pooled，平均ΔALC−0.000179972，BM-cluster CI[−0.000201013,−0.000040292]；未新增相对pooled的>=.005严重回归。93.77%增量仍来自alignment，dave相对R25五组均退化；不选最低分cohort、不自动晋级。H3辅助异质性关联不稳定，H4对幅度收缩的CI跨零，未唯一识别Subject特异方向适配。

R78去重191pairs/35BM、B15/B31支持交叉预测诊断完成。partial的BM内平均rho .240675、CI[.065278,.405159]，方向BA .628506；但单预算CI跨零，“不可靠”组平均query Δ仍为−.000846417。有限正面排序信号满足预设入口，不代表收益零点校准可靠。

R79实际执行严格嵌套来源策略验证；只有pool0一条内层lane准入，其余回退partial。gate−partial +.000020318、gate−matched-mean +.000006623，0/3来源池同时优于两对照；`source_policy_no_independent_gain`。错误关闭收益损失.000107159超过避免错误共享的.000086841。完整外层condition_not_met，停止本二值门控，不扫阈值/罚项、不扩B3/B7。固定Partial8仍为conditional_simulation研究候选；Hosted暂停、fallback未替换。

R77/R78/R79保存概率重计分最大误差分别8.88e−16/0/0；R79来源隔离及预测构造复核5.55e−17。无后台实验。总入口：[研究决策Q1–Q5](SHARED_RELIABILITY_DECISION.md)、[R77](../reports/shared_cohort_robustness_findings.md)、[R78](../reports/support_sharing_reliability_findings.md)、[R79来源与停止](../reports/shared_reliability_policy_findings.md)、[指标/正确率/资源全表](../reports/shared_reliability_tables.md)。任务`D:/todo/tasks/graph/nips-comp.shared-reliability-2026-10-23.md`。以下“当前”为各历史轮次截点。

## 2026-10-23 当前：plan_8 R73–R76双路线完成

原始授权 `doc/plan_8.md`：Hosted下载/审计继续暂停。R73 Luna 512主标注与64独立复标完成：436有效评分/76null、108items四维完整，双方有效48条的一致率68.75%，>1级分歧2.08%。原作者rubric/prompt重建与解析通过；未用subject成功率标注。原生模型快照/温度未暴露，属于四维Luna适配，不称原GPT4o精确复现。

R74教师B0相对同输入base退化+0.088423451，CI全正；CPU学生grade MAE0.991676仅略好于fit均值0.999972，学生先验新增ALC改善0.000079265的CI跨零，局部增量约0.000001632。联合来源门槛失败，`stop_at_pilot_keep_R25`；没有新No-Peer完整外层或扩标，不与共享模型组合。用户本次Luna授权已结束此前资源等待，但本轮效果条件未通过。

R75来源诊断/消融触发固定partial8一次完整评价；R76完成5242pairs×3seeds×六预算。主随机cohort fully pooled为0.190728761，partial8为0.190547856；同cohort ΔALC −0.000180905，CI[−0.000202247,−0.000038056]，三fold/BM均改善。相对旧R72 pooled3的CI跨零、BM略差，不能称稳定取代历史R72。相对R25约96.98%增益来自alignment，dave仍退化+0.006632866；四项数值Gate通过但严重回归尚未解决，仍只保留conditional_simulation研究候选，不晋级部署。

R74/R76保存概率重计分最大误差分别0/8.88e−16；16项合成协议smoke和fresh-item CPU wrapper smoke通过，单一只读结构复核无阻断。adapter按可见配置排序选peer，与离线随机cohort不等价；新item本地编码能力不等于Hosted验收。R25/R56/R72科学资产及两份fallback保留，全局状态和ledger按授权更新。总入口：[双路线结果与认识](../reports/dual_track_demand_sharing_findings.md)，[A全表](../reports/demand_pilot_tables.md)，[B全表](../reports/shared_slope_tables.md)，[交付验证](../paiec_baseline/runs/research_loop/dual_track_v8/delivery_validation.json)。任务登记 `D:/todo/tasks/graph/nips-comp.dual-track-demand-sharing-2026-10-23.md`。以下“当前/等待/运行中”为各轮历史截点，不覆盖本段。

## 2026-10-23 当前：plan_6 R69–R72完成

R69官方接口/client与PAVA原文/作者实现核查完成；没有真实Hosted样例，coordinator组成仍未知。R70/R71/R72均完成5242pairs×3seeds×6budgets及保存概率重计分，最大误差9.99e−16。R70 independent .195141844761、monotone/global=R64 .195129046979；R71 grouped .195138170529，均未改善R25，停止当前预算/近邻分组族。R71控制RNG错误部分批次保留且排除，修复后完整重跑。

R72条件模拟出现正面增量：same3 .194752318424、pooled1 .193447868573、pooled3 .190683911490、diff3 .190885867939。pooled3比R25改善.004440014486，三fold/BM改善、clusterCI [−.005055159,−.000319164]；相对两均值基线过条件数值门槛。删除同输入peer后仍保留95.45%点估计收益，peer置换和仅增加标签均值控制支持内容对应的作用。96.35%总增益仍来自alignment，相对R56/匹配R36的CI跨零，BM-equal略差R36；不宣称普遍支配或Hosted成绩。

无实验仍在运行；原183项冻结资产、R25/R56/R64与fallback不变，无提交。下一步等待实际Hosted peer样例/抽样合同，再回放冻结共享算子；ADeLe继续等待资源。入口：[主报告](../reports/lexical_peer_evidence_findings.md)、[分数/正确率与切片](../reports/lexical_peer_evidence_tables.md)、[复现说明](../reports/lexical_peer_evidence_reproduction.md)、[算子决定](LEXICAL_PEER_OPERATOR_DECISION.md)。任务卡：`D:/todo/tasks/graph/nips-comp.lexical-peer-evidence-2026-10-23.md`。下方为历史轮次状态，不覆盖本段。

## 2026-10-22 当前：plan_5 R65–R68完成

R65冻结概率诊断、R66匹配特征消融、R67全外层及R68资源/接口审计完成。新增15fits（R66十二次30epochs、R67三次短refit）。item-only在来源两池胜R25/full/常数，subject-only不稳定；固定item分段全外层ALC0.194448914，比R25改善0.000675012但弱于R56_segment，一折/BM/clusterCI未通过，收益仍由alignment主导。停止当前先验权重/门控/架构修补。

高层认识：合法item字段可保留部分低预算均值信息，显式subject描述不是必要条件；这不证明非加性交互、通用任务难度或可靠跨Benchmark迁移。query整组统计仅离线诊断。R65 metadata描述缓存已修复，最终使用`R65/descriptor_v2`，原分解不变；R66/R67身份和保存概率重计分误差0，旧冻结输入/fallback未变。ADeLe512作者请求重建一致仍等待资源；官方多pair labeled可见性有依据，但hosted peer范围/状态/item匹配合同未完整确认。无后台实验、无晋级或提交。入口：[机制报告](../reports/transferable_prior_mechanism_findings.md)、[决策](TRANSFERABLE_PRIOR_DECISION.md)、[指标及正确率](../reports/transferable_prior_tables.md)。下方为各历史轮次状态。

## 2026-10-22 当前：plan_4历史正面信号补证

R62/R63/R64均完成5242pair×3seed完整评价和保存概率重计分，误差0。R56分段点估计0.193805399但收益集中主导Benchmark，BM/一折/CI未通过；R55 query仅改善约7.3e−6。R64模板平均0.195129047，修复原词法平均的大部分退化但仍略差R25。新增fit10次，复用R43权重3份，无新embedding、无待运行实验、无晋级或提交。详见[本轮状态](HISTORICAL_POSITIVE_FOLLOWUP_STATE.md)、[结果与机制认识](../reports/historical_positive_followup_findings.md)。下面“未做完整外层”等记录只指各历史轮次当时的状态。

## 2026-10-22 当前：R59–R61低预算先验诊断与条件组合

用户`doc/plan_3.md`已完成R59冻结预测诊断、R60单一无训练B1门控及R61来源pilot，数值与单一只读结构复核完成，无阻断问题。824次来源池pair记录对应412个不同pair，池之间重叠。R54/R56低预算总体优势由pair均值项主导；同pair内各题/variant的概率变化为净成本。R56主要改善均值项，而非普遍减少预测方差或提供独立信息源。

无条件B0/B1预算分段在池0退化。首个合法支持标签的prior误差与收益有同向关系，故仅冻结`error<.25`的B1选择器，其他预算回退R25；实测三池ΔALC为+0.000313793/−0.000746241/−0.000396236，1/3池满足数值条件，整体未通过。没有阈值搜索、门控训练或新MLP训练；完整外层`condition_not_met`。主报告：[low_budget_prior_composition_findings.md](../reports/low_budget_prior_composition_findings.md)，合同：[LOW_BUDGET_PRIOR_COMPOSITION_SPEC.md](LOW_BUDGET_PRIOR_COMPOSITION_SPEC.md)。

高层更新：pair均值信息存在于某些来源，不代表一条支持证据能够识别其可靠性。停止当前prior分段/单标签门控修补，下一问题是有哪些合法输入能在未知Benchmark上标定pair均值与误差方向。已复核R20/R36，跨subject仍有输入共享契约与强基线增益限制，不据旧弱peer对照自动开发。R25/fallback和原.005Gate保持；ADeLe等待资源。下方章节为历史。

## 2026-10-22 最新：plan_2五候选已完成

R54–R58按用户顺序和条件完成。36fits、996个epoch checkpoint：普通MLP三来源池双方向均为训练改善而来源外持续恶化；R43重采样、Benchmark风险及R43平均均未通过全部来源条件。MLP平均/初始化诊断未触发，不把未执行项说成训练失败。当前无后台训练；无新outer成绩或晋级，fallback不变。结果和理解见 [training_mechanism_v5_findings.md](../reports/training_mechanism_v5_findings.md)，恢复入口 [TRAINING_MECHANISM_V5_STATE.md](TRAINING_MECHANISM_V5_STATE.md)。下方R51/R52及更早内容保留为历史。

## 2026-10-22 最新复现阶段

用户授权 `doc/复现与组合.md` 后，R51 CNP/AttentiveCNP 与 R52 amortized IRT 已完成真实来源内实验，全部未通过进入条件；没有外层新成绩、组合或晋级。ADeLe原文与输入准备完成，用户确认等待资源。当前无后台训练。最新状态见 [REPRODUCTION_V4_STATE.md](REPRODUCTION_V4_STATE.md)，结果与认识见 [reproduction_v4_findings.md](../reports/reproduction_v4_findings.md)。以下v3及更早章节保留为历史证据；原Research Gate/fallback保持。

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
