# plan_4 R62–R64 当前状态

2026-10-22：R62/R63/R64全部完成5242pair×3seed完整外层及保存概率重计分，误差0。本轮无待运行实验。R64/summary.json的`outer_status=required`是冻结的评分前进入决定，保留不改；最终结果在R64/full/summary.json，执行状态在R64/status.json。

- 控制原文：`doc/plan_4.md`，保持不变。
- 冻结合同：`HISTORICAL_POSITIVE_FOLLOWUP_SPEC.md`。
- R62：R56_segment ALC0.193805399，收益集中alignment_faking，BM/折间/CI未通过；未晋级。
- R63：R55 query ALC0.195116628，仅比R25好0.000007298，区间跨零；未晋级。R43和episode输R25。
- R64：来源通过后完成外层，template_average ALC0.195129047，比原平均好0.000904020、比R25差0.000005121；未晋级，无系数搜索。
- 实际命令：`reports/historical_positive_reproduction.md`。来源和权重hash已冻结，正常resume复用完成Benchmark。未完成Benchmark需先检查其原始chunks，禁止覆盖。
- 暂停过一次R62慢速编码，保存全部中间产物后以精确缓存继续；7个重叠chunks逐字节一致。数值审计也保留一次解压性能中断记录。
- 本轮task：`D:/todo/tasks/graph/nips-comp.historical-positive-followup-2026-10-22.md`。
- R25/fallback保持；没有正式提交。ADeLe独立等待资源。
- 最终报告、正确率与机制认识：`reports/historical_positive_followup_findings.md`及`historical_positive_tables.md`。任务办结不代表.005研究目标达成；后续优先核实低预算先验的独立迁移可靠性，不按本轮外层重调阈值。
