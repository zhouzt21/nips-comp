# R77 H3辅助诊断范围

记录于第二cohort补评完成、三新cohort扩展运行尚未完成时；未查看这些新cohort分数。属于次要探索诊断，不用于选择cohort、模型或R79规则，也不更改R77已冻结预测。

每个BM×五cohort只选择可见subject_key字典序中首个完整四人组。使用item seed0、own/peer当前B31 acquisition，复用R75 `heterogeneity`：16/15互补两半、32次共同斜率fitted Bernoulli null，seed为`7709|BM|cohort`。记录H、repeatability、null95与四人配置。

把H同该四人组、同seed0、同B31的partial−pooled query Brier对照；不把B31诊断解释为B3时合法可得的feature。各cohort的跨BM Spearman，以及先在同BM平均cohort后跨BM的相关单独报告。组数/BM覆盖受限，fitted null非真实异质性证明；关联不建立因果。

失败/零方差项明确不可估。不得选择query表现最好的组；不把该诊断输入R78/R79 gate。主要R77决策仍按原五cohort配对ALC合同。
