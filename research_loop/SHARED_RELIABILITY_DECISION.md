# plan_9：R77–R79 最终研究决定

2026-10-23。原始授权：[plan_9.md](../doc/plan_9.md)。R77完整评价、R78来源诊断、R79条件来源策略验证均已实际执行。**R79完整外层未触发**，按预设条件停止。本轮工作完成，没有仍在运行的实验。

## 结论与阶段边界

|阶段|实际范围|结果|决定|
|---|---|---|---|
|R77|36 BM、5242 pairs、3 item seeds、6预算；五冻结cohort及主cohort压力场景|五套partial−pooled均负，cohort平均ΔALC −0.000179972，95% BM-cluster CI [−0.000201013,−0.000040292]|保留组成稳健性证据；不选最低分cohort、不扫描罚项|
|R78|3重叠来源池；去重35 BM/191 pairs/573轨迹；B15/B31、五场景|partial风险差的BM内平均Spearman 0.240675，CI [0.065278,0.405159]；单预算CI均跨零|正面排序信号满足预设诊断入口，但不是回退规则已有效|
|R79 source|严格Benchmark嵌套选择、匹配平均强度/平方位移对照；完整保存1146来源轨迹暴露|gate−partial +0.000020318，gate−matched-mean +0.000006623；0/3池同时优于两对照|`source_policy_no_independent_gain`，停止当前二值配置；不跑完整外层|

R77主cohort分数仍为pool **0.190728761**、Partial8 **0.190547856**；第二cohort补评为pool **0.190667308**、Partial8 **0.190485837**。三个新cohort的partial为0.190494634/0.190463198/0.190412262，各自只与同cohort pooled比较。不把最小值选成正式新成绩。

## Q1：不同peer组成下是否仍有Partial8增量？

**在五套冻结组成中有。** 配对增量范围−0.000181470至−0.000178265；跨cohort平均后BM-equal Δ为−0.000106978。主cohort的1peer/2peers、missing、uneven也保持负增量；0peer和配置歧义精确回退R25。缺失与不均衡同时改变信息量和构成，不能据这些场景独立识别“少观测”与“peer方向”的因果贡献。

增量仍不普遍：约93.77%的partial−pool平均收益来自alignment；去alignment再归一Δ仅−0.000060984。bertaqa、cmmlu、exploitgym在不同cohort间有小幅方向翻转。没有新增partial相对pool的>=.005严重回归，但dave相对R25五套均退化，其中三套超过+.005；主cohort为+.006632866。

这属于已反复使用公开Benchmark上的组成敏感性验证，不是新的独立Benchmark盲测。item seeds与cohorts没有被当作独立BM样本。证据：[R77报告](../reports/shared_cohort_robustness_findings.md)、[配对原始结果](../paiec_baseline/runs/research_loop/R77/comparisons.json)。

## Q2：当前支持标签能观察哪些差异？

能观察到拟合后局部方向的差异、重复mask下的预测风险以及它们与query收益的部分关联；**尚不能把这些量等同于真实、可迁移的Subject能力异质性**。

辅助H3在每BM/cohort固定四人组的B31诊断中，五套H–query增量rho方向混合，按BM平均后rho=+0.076568；未支持“异质性越大，partial增益越大”的稳定负相关解释。该诊断是第二组成补评后的探索性补充，非主要确认检验。

H4幅度对照未排除纯正则解释：partial−来源投影收缩的平均Δ虽为−0.000385459，其CI [−0.000456114,+0.000081939]跨零，BM-equal为+0.000007171，去alignment为+0.000028072。更好的总体点估计不能证明已准确识别Subject特异方向。位移平方成本约为pooled的93.8%，是修正幅度成本，不是中心化预测方差。

## Q3：支持预测一致性能预测query收益吗？

**在来源聚合层面有有限正面证据；在实际单预算决策层面仍不稳定。** partial/pooled风险差的方向balanced accuracy为0.628506/0.646630，mean-only为0.526594；mean-only支持风险在B15/B31相对own均值分别恶化+0.016040/+0.026325。peer标签置换后斜率关联接近零。这支持item对应的局部关系参与了信息，而不只是平均正确率相近；未对四指标差异作独立显著性检验。

关键反例是：partial的“可靠”组query风险差为−0.005688682，“不可靠”组仍为−0.000846417。**风险排序不等于收益零点已经校准。** 主场景B15/B31单独rho仅0.178472/0.159425，各自CI跨零；总体rho只有24/35 BM可估，不能把不可估BM当零相关或已通过。

遮挡后own拟合样本量约减半，query使用全部当前标签；fit规模变化、有限样本噪声和真实结构差异都是仍可能的解释。当前证据不能唯一裁定其因果来源。证据：[R78报告](../reports/support_sharing_reliability_findings.md)、[所有预算/场景/方向和response准确率](../reports/shared_reliability_tables.md)。

## Q4：保守门控是否减少错误共享并优于简单收缩？

**本轮没有建立这一结论。** 真正嵌套来源选择中，仅pool0的一条内层lane满足准入、选择pooled风险差；另五条保持原partial。去重后20.42%的预算请求有已准入门控，名义关闭率11.43%。大量零差值反映证据不足后的回退，不表示各BM门控都被充分检验。

关闭有益共享59次、有害共享54次、中性18次（单位pair×seed×预算）。挽回损失0.000086841 ALC，错失收益0.000107159，净退化+0.000020318；对matched-mean/+matched-cost的差值分别+0.000006623/+0.000021573，CI均跨零。dave来源切片改善−0.001249885，但algotune等损失抵消；未运行完整dave外层，不能称严重回归已解决。

matched对照参数只由fit半组估计，validation实际平均成本不必完全相同。无query标签进入运行时z；目标outer BM不参与任何规则选择。证据：[R79来源与停止报告](../reports/shared_reliability_policy_findings.md)、[隔离验证](../paiec_baseline/runs/research_loop/R79/source_isolation_validation.json)。

## Q5：下一轮决策

1. 保留固定Partial8为`conditional_simulation`研究候选，认可其在本轮peer组成中的平均结构增量；不部署、不改fallback、不挑cohort。
2. 停止当前`g=1[z<0]`二值回退，不自动扩展B3/B7、阈值网格或新门控网络。R78标记为正面来源诊断，R79标记为策略门槛失败，不能合并成“支持一致性完全没有信息”。
3. 重新开放可靠性研究须先有预算匹配的收益符号校准/来源外辨识证据，以及足够的Benchmark组样本；它应解释为什么半样本支持风险能预测full-support query风险。可辨识性与校准是待证问题，不是已授权追加的新实验。
4. Research Gate及Hosted条件均未因R77通过而自动满足。dave严重回归未解决，真实peer契约仍未验证；按用户指示保持Hosted工作暂停。

## 可复现与证据完整性

- 原文依据与作者代码：[KDD2004/MALSAR、DomainBed版本及适配映射](literature/shared_reliability/source_evidence.md)；AnySearch原始检索留档。复用真实R76算子，不宣称复现文献原始基准数字。
- 完整合同、环境、数据/模型revision、代码commit+dirty源码hash、命令与文件映射：[复现说明](../reports/shared_reliability_reproduction.md)；原始R77/R78/R79各目录保留概率、query身份、response计数和支持mask。
- R77两阶段保存概率重计分最大Brier误差8.88e−16；R78支持z/query Brier/accuracy误差0；R79风险重计分误差0、输入及门控构造复核最大5.55e−17。检查者为主agent独立计算与确定性核对，不声称外部独立review。
- 三张科学图已逐图打开检查；修正了坐标拥挤和色条裁切，不修改实验数值。阶段墙钟耗时与进程峰值见全表；最大约4359.30MiB。CPU最多4线程，无新网络训练。
- 主3peer场景B31平均own31+peer91.79=122.79标签，最大124。Partial与matched pooled同信息，不能宣称相同总31标签预算下优于R25。
- R1–R76科学资产及两份fallback按已有manifest核对；仅追加全局state/ledger。最终完整清单及验证：[shared_reliability_v9](../paiec_baseline/runs/research_loop/shared_reliability_v9/)。
- Dtodo任务：`D:/todo/tasks/graph/nips-comp.shared-reliability-2026-10-23.md`；执行留痕在同任务source_refs。没有将AI运行时间计为用户投入。
