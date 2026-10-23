# R78：支持标签上的共享可靠性是否预示query收益？

2026-10-23。实际来源诊断完成；状态 **`support_consistency_predictive_source_entry`**。本报告与[R77](shared_cohort_robustness_findings.md)分别回答可共享结构和可观测可靠性。

## 1. 主要结果与停止决定

|指标|BM平均Spearman|95% CI|方向balanced acc|可靠−不可靠query风险差|BM内组差|mask符号一致率|
|---|---|---|---|---|---|---|
|partial|0.240675|[0.065277778, 0.405158730]|0.628506|-0.004842264|-0.002029117|0.874346|
|pooled|0.247024|[0.062797619, 0.419149306]|0.646630|-0.005222281|-0.002989271|0.895288|
|mean|0.194013|[0.002772817, 0.383367522]|0.526594|-0.001392247|-0.001260970|0.806283|
|combined|0.246528|[0.078670635, 0.417658730]|0.569308|-0.002161291|-0.001729050|0.806283|

|变量|未通过的预设项|
|---|---|
|partial|全部通过|
|pooled|全部通过|
|mean|全部通过|
|combined|全部通过|

R79诊断入口：**True**。通过后仍须嵌套来源策略比较，不能直接进入完整外层。

这里“可靠”固定指z<0；query风险差目标为full-support partial−own。rho正值表示较坏support相对风险伴随较坏query风险，负值则反向。方向识别排除query差为零的中性pair；balanced accuracy为有利/有害两类召回均值，不能与PAIEC response准确率混淆。

**结果的关键限制：** partial的可靠组query Δ为-0.005688682，不可靠组仍为-0.000846417，两组平均都从共享获益。组间排序较好不等于z=0能标定query收益的正负边界。partial组间差的BM-cluster CI为[-0.004471887, 0.000249538]，跨零；本轮预设入口只要求该组差的点估计方向，不能把通过入口写成所有检验显著。

主场景单独B15/B31的rho分别为0.178472/0.159425，各自CI均跨零。预设跨预算、seed汇总得到的正面信号，强于单次预算下实际可使用的证据。只有24/35个BM可估rho，来源至多8pairs/BM，不等同完整5242pairs分布。

## 2. 实际协议与隔离

沿用R75每BM至多8个唯一可见配置的来源名单；三来源池有重叠。总诊断记录11460，对应1146次来源轨迹暴露；固定每BM首个合法来源池去重后573条轨迹。每轨迹比较B15/B31及主7501、第二7502、missing、uneven、peer-label permutation五场景，不是新增独立Benchmark。

每预算做两个固定随机二分重复、每重复互补两mask，共4次fit；每条own支持标签恰好被留出两次。fit-own均值、中心化和斜率都只用未遮挡部分；peer只用当前预算合法前缀。query outcome只在预测完成后评分。改变heldout-own标签不改变fit预测的对抗断言逐实例执行。

保存527160条support留出预测记录、361770条query variant记录。独立重计分：query Brier误差0、accuracy误差0、support z误差0。mask、身份、peer前缀及原概率均已落盘。

## 3. 均值一致与斜率一致

- partial/pooled：各Subject均值各自保留，检验局部斜率预测heldout own标签的相对Brier。
- mean：仅比较peer等权平滑均值与fit-own平滑均值，两者都没有局部方向。
- combined：partial的已裁剪预测移除own截距、加peer均值再clip；固定构造，没有增加拟合参数。它是均值借用控制，不宣称精确恢复未裁剪的潜在斜率。

四类变量均事前固定，没有按dave或外层误差设计阈值；预设z<0与z>=0分组。先平均同pair预算和item seeds，再在每BM内算Spearman。少于3pairs或常数统计的BM标不可估，不填0。CI以可估BM为cluster；它并不代表那些无法估计BM的表现。

主场景mean-only支持风险相对fit-own均值在B15/B31分别恶化0.016040418/0.026325264；平均成功率借用本身未显示益处。斜率指标方向BA约0.629/0.647，均值约0.527；combined虽rho约0.247，方向BA约0.569。固定控制更支持局部对应关系提供排序信息，但没有对四指标差异作独立显著性检验，不能声称已唯一识别因果机制。

## 4. 反例与机制边界

完整的各预算、missing/uneven/第二组成/置换和每BM统计见[全表](shared_reliability_tables.md)、R78/associations.json与benchmark_associations.csv。置换控制破坏的是当前peer对应关系；若某可靠性变量只能识别大幅破坏，不足以证明它能区分真实共享中较小的有害误差。

|场景|partial BM rho|95% CI|方向BA|可靠−不可靠query差|
|---|---|---|---|---|
|alternative|0.170536|[-0.038293651, 0.368159722]|0.638889|-0.004960987|
|main|0.240675|[0.065277778, 0.405158730]|0.628506|-0.004842264|
|missing|0.249855|[0.061958907, 0.434101104]|0.591185|-0.003466372|
|permuted|-0.014187|[-0.207646329, 0.177579365]|0.467271|0.000037109|
|uneven|0.228472|[0.034322917, 0.409722222]|0.578616|-0.003531623|

主场景的斜率关联在peer-label permutation后接近零、CI跨零，支持真实item对应关系参与了该信号；这并未消除小样本噪声或证明跨来源泛化。

|预算|切片|支持partial−own|query partial−own|z>=0请求比例|
|---|---|---|---|---|
|15|all|-0.002548243|-0.003151155|52.88%|
|15|alignment|-0.003983380|-0.006154972|29.17%|
|15|without_alignment|-0.002485505|-0.003019840|53.92%|
|15|dave|0.007762891|0.005167011|70.83%|
|31|all|-0.004588807|-0.004271270|43.80%|
|31|alignment|-0.008852604|-0.010847197|4.17%|
|31|without_alignment|-0.004402411|-0.003983798|45.54%|
|31|dave|0.009382644|0.004128125|66.67%|

dave的z与query风险均偏坏，当前信息能捕捉部分错误共享；alignment之外仍有组间排序信号。但dave只含8个来源pair，不能据该切片推出完整dave已修复。源数据中的风险排序不提供按BM名称关闭共享的授权。

R78比R43/R61/R71新增了“peer局部方向对暂时隐藏own标签的预测”这一可观测量，直接接触共享机制。但最多31个标签再做holdout会减少own拟合信息，support风险与full-support query风险并非同一个估计问题。相关弱或跨BM不稳可以由有限样本噪声、fit规模变化或真正结构异质性共同造成；失败不能证明所有可靠性信号不存在。

R77结构增益成立也不推出z能可靠决定每次是否共享。本轮不从事后错误切片制造专门gate，不把均值接近自动解释为斜率一致。已进入严格嵌套R79来源策略验证，结果见[门控决策](shared_reliability_policy_findings.md)；未通过其进入条件，停止完整外层。

## 5. 来源、资源与复现

原文与作者代码为KDD2004/MALSAR的共享结构以及DomainBed的模型选择纪律；R78变量和mask是本项目预设机制实验，不是假称文献已验证的PAIEC方法。[来源定位](../research_loop/literature/shared_reliability/source_evidence.md)，[执行命令](shared_reliability_reproduction.md)，[严格OOF补充](../research_loop/SHARED_RELIABILITY_OOF_ADDENDUM.md)。

CPU运行墙钟耗时/真实进程峰值见R78/execution.json与analysis_execution.json；无新网络、LLM标注、外部付费API或Hosted工作。公开数据反复使用及来源池重叠使本轮仍是探索性诊断。图：[支持风险与query关联](shared_reliability_figures/support_query_association.png)。
