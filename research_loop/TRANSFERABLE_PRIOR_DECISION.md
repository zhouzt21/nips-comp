# R65–R68 decisions

2026-10-22。原始用户计划`doc/plan_5.md`保留。所有关键数字和理解以[主报告](../reports/transferable_prior_mechanism_findings.md)及原始run文件为证。

## R65 — decompose → compare rivals

实际15726轨迹重计分，分解精确。R56的全来源常数对照B0更差，不能把R56简化为全局prevalence。mean改善占97.7%，B1优势主要继承B0，首标签相对更新缩小优势。输入统计与误差关联跨fold反转，不能直接形成可靠性门控。H1/H2/H3仍需特征消融，故R66进入。

数据修正：最初descriptor缓存只以content为key，遗漏相同内容不同metadata的差异。最终使用`R65/descriptor_v2`；旧root保留给R66评分前引用，Brier分解经重算完全不变。初始非fit零行来源错误也保留冻结快照，不作为结果。

## R66 — controlled information deletion

12新fits×30epochs，复用原R56六条完整轨迹。subject-only数值条件不通过；item-only在两个来源池改善且胜fit-only常数，合并pair/BM及移除alignment诊断方向均改善，满足事前探索条件。因此进入一个R67候选。所选epoch subject=[2,1,1]，item=[1,2,2]。

机制判断：item字段足以保留来源信号，不需要先假设显式subject能力或复杂交互。item-only包含metadata和interactors；不等于纯文本、不证明所有subject信息无效，也不识别非加性交互的因果效应。来源验证参与选点、池间重叠，不作独立确认。

## R67 — minimal prior → full exploratory evaluation

实际3个全来源短refits及5242pairs×3seeds完整评分。item-only固定B0/B1，B3–31固定R25。ALC0.194448914，ΔR25−0.000675012；相对R56_segment退化+0.000643516。三fold一折退化、BM等权退化、CI跨零、整体收益仍由alignment贡献。

决定：**complete_not_promoted**；Research Success=false；停止当前item-only/R56的架构、阈值、mix与学习率修补。没有在外层选择新预算/字段/权重。保留R25主要研究参照、R56分段探索记录和原fallback，不正式提交。

## R68 — acquire genuinely new information only when available

- ADeLe输入忠实性审核完成：512请求与作者builder逐条完全相同。仍无可用judge snapshot或需求标签，`waiting_for_annotation_resources`；无API费用、无新模型下载或效果实验。遵循用户原先等待决定。
- 官方文档明确允许使用当次labeled传入的多pair观察；但是hosted peer组成、跨调用状态/顺序、可靠item匹配仍缺合同。`blocked_by_runtime_contract`；未重复R20/R36模拟或假设隐藏共享标签。

## Final evidence boundary / next action

R65/R66/R67概率和身份审计完成，单一只读结构复核无剩余阻断；不是双审或人类最终科研claim验收。公开36BM反复开发，单模型seed与三item-split seeds不同。没有新未使用Benchmark，不能声称无偏泛化。

下一步只在新增资源/官方runner样例到位后恢复相应信息路线。当前执行任务已完成，研究成功目标尚未达到；未执行ADeLe效果与hosted共享效果不算负实验。
