# R78/R79 来源选择隔离补充（运行R78之前）

2026-10-23。不改变R77 frozen合同或其任何参数。R78尚未开始评分；本文件在R78合同中冻结。它明确落实原plan_9的“所有规则严格来源外拟合”。

三个source pools互相重叠。综合去重rho/分组和“2/3池”只用于机制报告/是否有必要继续研究，不能将union上选择的metric直接应用于每个outer fold。

如R78存在正面信号：

1. 对每个outer fold，只用其合法source pool，按固定候选顺序partial/pooled/mean/combined选择首个通过的metric。该pool独立要求可估BM平均rho的95% cluster CI下界>0、预定可靠/不可靠组风险差pair与BM均负、去alignment同方向且两组各>=10pairs、>=5BM。没有通过则此fold固定回退原partial，不借另一pool的metric。
2. R79 source策略验证再按Benchmark hash两半；每次只在fit半组执行上述metric选择（小fit半组的最低样本仍不下调）、确定matched-mean共享比例，再应用于validation半组。不能先在整个source pool选metric再把同pool拆分称OOF。无fit指标通过时，该内层策略为原partial，matched对照同样不变。
3. R79外层前的候选、每折metric、关闭的预算、对照常数仅由该fold全部source数据 refit。新target query outcome从不进入参数选择。全局诊断的来源池重叠和自适应选择仍需披露。
4. 若R78原预设总体诊断失败，则不运行R79；若总体通过而嵌套策略门槛失败，同样停止，不降低门槛或增加变量。

此补充为防止交叉来源选择泄漏，不是根据R78/R79结果调整门槛。
