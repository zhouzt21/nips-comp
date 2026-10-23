# R59 → R60 算子决定（R61评分前）

2026-10-22。依据已完成R59原始诊断，冻结以下唯一新候选；不是依据候选实际分数选型。R59本身基于已查看来源，仅探索性。

## Current Bottleneck

R54/R56的低预算优势不足以支持无条件预算分段：池0退化，池1/2改善；没有B0可靠冷启动门控证据。全量先验替换也牺牲高预算R25收益。

## Upward Diagnosis

R59六个method×pool的B0/B1 `Var(P)-2Cov(P,Y)`均为正；总体优势来自pair均值项，不能解释为稳定item区分收益。R56并非普遍压低方差：三个池B0 Var(P)均高于R54，但均值偏差更低。相同数据/架构不同风险，不据此声称独立信息源。

预算分段R56相对R25 ALC变化为+0.000127949、−0.002550145、−0.002129535。MLP均值置信度与方差在B0的收益相关方向不一致；R54/R56分歧与收益方向一致但较弱，并且没有阈值或误差保证。不能由这种分歧随意扫描门控。

B1有一条新增可见证据：第一个已揭示标签对R56原始prior的平方误差，与相对R25收益的Benchmark级Spearman分别−0.319658、−0.328063、−0.325099。它与真实任务有直接联系，但一个标签方差大、并不证明可靠性已经可辨识。此关系也部分受B1残差校正公式影响，必须以无标签泄漏的候选实评分判定。

## Operator Choice

`Audit Reliability → Budget-specific Conditional Selection`。将预算不足的问题拆开：B0无可靠证据，采用R25；B1只用唯一允许标签进行proper-loss比较；B≥3保留R25。

冻结候选名 `R56_support1_gate`：

```
if budget == 1 and (R56_prior(support_1) - observed_label_1)**2 < 0.25:
    use frozen R56 budget-1 probability (original residual-strength2 adjustment)
else:
    use R25 probability
```

0.25来自恒定0.5预测对任意二值标签的平方误差，不进行阈值或mix扫描。严格小于，平局回退R25。仅一个候选、不训练门控、不增加MLP训练。选R56是因为其B0/B1对匹配R54三个池均改善；R54仅作既有对照。没有跨池最佳模型拼接。

与R43区别：R43在B≥3学习局部语义方向的幅度g；本候选在B1利用新揭示的一条Bernoulli证据，在冻结历史prior与平滑均值之间作一次离散选择。它不学习query-specific修正，也不声称克服R43所发现的少样本估计困难。

## Experiment

R61回放R59冻结预测。门控仅取R59按支持输入计算的 `first_support_sqerr`，不使用query标签、pair成功率、Benchmark/subject ID、未来support标签或oracle结果。保存每pair门控决定和预算预测，评分另读counts/successes。来源划分、预算、重复响应、checkpoint选择均与R59一致；不得重新选epoch。

对照：R25、完整R54/R56、R54/R56预算分段、smoothed2、training-selected smoothed。报告六预算Brier/预测正确率、pair/BM ALC、逐BM变化、paired bootstrap和严重回归。已有checkpoint的来源选择偏差明确保留，绝不称独立盲测；候选没有学习步骤，因此不存在新门控cross-fitting拟合。

## Evidence/Gate

来源门槛原样保持：三个池pair ALC严格改善且BM等权不退化，机制方向一致，没有未解释的单BM ALC≥.005回归。若失败，停止此候选；不得再改阈值、改prior或用相同来源训练更复杂门控。只有满足才进入原5242pair×3种子外层。现有相关性和oracle均不算通过证据。

## Next Move

实际执行这个最小B1门控后更新理解。如果失败，应区分“某些pair确有均值先验收益”与“单条支持证据足以辨认它”，降低现有prior修补优先级；不否定所有合法可靠性信息或所有先验模型。
