# 复现阶段当前状态

2026-10-22。控制方案为用户原文 `doc/复现与组合.md`，实验冻结合同为 `REPRODUCTION_V4_SPEC.md`、`R51_protocol.md`、`R52_protocol.md`。冻结文件中的“当前/尚未评分”描述的是评分前快照；最新状态以本文件和结果JSON为准。

## 已完成

- AnySearch检索、原论文/作者代码逐项核对、固定版本和源文件hash。
- R51 CNP/AttentiveCNP：12次完整60epoch拟合，三来源池全部未通过gate；停止扩容与outer。不是完整latent ANP复现。
- R52 author-calibration IRT+PAIEC冷启动适配：6次完整拟合并收敛，三来源池全部未通过B0/B1门槛；zero-subject-head消融同样失败；停止outer及组合。
- 28项合并实现检查、R25来源回放、单一独立结构复核；原始JSON/权重/曲线/误差分解/固定阈值正确率均保存。当前无后台训练。

## 用户确认等待

R53 ADeLe：源文件核对和128items×4rubrics=512请求的dry-run已完成；用户明确“先完成CNP/IRT，ADeLe等待资源”。没有API调用、标注或效果分数。待已有本地生成模型，或确定的服务/模型/预算可用，再冻结小样本标注资源实验。

## 研究结论与下一步

没有候选晋级、没有新外层成绩、没有提交或改动fallback。R25保留研究参照地位，原Research Gate仍未通过。来源内失败不能泛化为整个CNP/ANP/IRT方法家族无效，也不能据此保证ADeLe需求特征有效。当前不继续同配置扩容/调参/组合。

阅读入口：[结果与理解汇总](../reports/reproduction_v4_findings.md)、[六预算指标CSV](../reports/reproduction_v4_metrics.csv)、[来源映射](literature/reproduction_v4/)、[R53准备记录](R53_annotation_feasibility.md)。原始结果：`../paiec_baseline/runs/research_loop/R51/summary.json`、`R52/summary.json`。
