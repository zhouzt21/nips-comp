# PAIEC Research Prompt：R69–R72——词法残差收益提取、预算自适应修正与跨 Subject 当前标签信息验证

## 一、研究角色与核心目标

你是一名 **Probabilistic Prediction Researcher、Statistical Learning Expert、OOD Generalization Diagnostician、Scientific Experiment Designer 与 ML Systems Engineer**。

PAIEC 项目目前已完成 R1–R68。大量实验表明，继续调整现有 MLP 的架构、训练轮数和先验门控，难以稳定改善跨 Benchmark Brier ALC。

但还有两类尚未充分利用的正面证据：

**第一类：R64 的词法残差修正。**

R64 已经发现，模板处理后的词法预测包含与 R25 误差有利对齐的信息，但固定融合的平方位移成本几乎完全抵消了收益。

**第二类：R20/R36 的跨 Subject 当前标签证据。**

同一 Benchmark 内其他 Subject 的已揭示标签，可能包含额外的 item 难度或任务结构信息。但之前实验主要在条件共享场景下完成，尚未建立与正式 Hosted 输入完全一致的共享证据合同。

本轮目标是：

1. **通过预算自适应收缩，减少词法修正的无效位移成本。**
2. **识别并保留可靠的词法残差修正，而非不加区分地平均两个预测器。**
3. **首先通过官方文档、代码与运行时样例明确跨 Subject 标签的真实可用性，然后在合法信息范围内开展共享预测实验。**
4. 判断哪些机制能够进一步改善 R25 的 Brier ALC，以及它们是否具有跨 Benchmark 稳定性。

本轮需要**实际执行实验、保存原始结果并更新研究认识**，而不是只生成下一份研究计划。

继续遵循：

**Observed Signal → Mechanism Diagnosis → Operator Selection → Minimal Intervention → Discriminating Experiment → OOD Evaluation → Research Decision。**

不以算法 novelty 为主要目标，不进行大规模无依据超参数搜索。

---

## 二、恢复当前研究状态

项目目录：

`D:\nips-comp\nips-comp\paiec_baseline`

首先阅读：

- `reports/historical_positive_followup_findings.md`
- `reports/historical_positive_tables.md`
- `reports/transferable_prior_mechanism_findings.md`
- `reports/transferable_prior_tables.md`
- `reports/research_loop_v3_findings.md`
- `reports/low_budget_prior_composition_findings.md`
- `research_loop/HIGH_LEVEL_STATE.md`
- R20/R36 的原始实验、共享证据合同和配对结果
- R42/R45/R46/R64 的词法编码、模板提取、语义预测、组合公式及逐 item 概率
- R25 当前完整预测实现及冻结评价协议

重点恢复：

`runs/research_loop/R64/`

以及相关 R25、R42、R46、R20、R36 原始产物。

优先使用冻结权重、已保存概率、item 身份、support/query 对应、重复 response 计数与原始实验合同。

若已有产物足以完成实验，不重复训练模型。

### 当前成绩

原公开数据代理评价：5,242 pairs × 3 item-split seeds。

| 方法 | Brier ALC ↓ |
|---|---:|
| Smoothed mean strength 2 | 0.198687363 |
| R25 | 0.195123926 |
| R56_segment | 0.193805399 |
| R64 模板词法等权组合 | 0.195129047 |

R25 仍为主要研究参照；R56_segment 是公开数据上的正面探索候选，但未通过研究晋级标准。

### R64 已有机制证据

令：

- \(p\)：R25 预测；
- \(d\)：R64 模板词法组合相对 R25 的预测变化；
- \(y\)：真实 response 标签。

则：

\[
\Delta Brier
=
2\mathbb E[(p-y)d]+\mathbb E[d^2]
\]

当前外层结果：

\[
2\mathbb E[(p-y)d]=-0.000177352
\]

\[
\mathbb E[d^2]=+0.000182473
\]

\[
\Delta ALC=+0.000005121
\]

说明存在有利的残差对齐，但固定等权融合的成本略高于收益。

同时，不同预算出现不同方向：

| 预算 | R64 相对 R25 的 Brier 变化 |
|---|---:|
| B3 | −0.000085113 |
| B7 | −0.000023805 |
| B15 | +0.000059317 |
| B31 | +0.000150413 |

B0/B1 不变。

这些已查看的外层结果只用于**提出和诊断研究假设**，不能直接用于拟合最终修正系数或事后确定可靠性门控。

---

## 三、R69：优先核实跨 Subject 标签的真实运行时契约

**这是跨 Subject 实验必须先完成的阶段。**

同时，R70/R71 的词法实验可以在不依赖此契约的情况下独立进行。

### 3.1 检索和核查顺序

优先访问：

1. PAIEC 官方竞赛规范  
   https://aims.stanford.edu/competition/

2. 官方 baseline 仓库及实际源码  
   https://github.com/aims-foundations/paiec_baseline

3. Codabench 官方起始包、评分器说明和能够合法访问的运行时样例。

4. 本地已有的官方源码快照、starter 文件及 R68 runtime audit。

如已配置 AnySearch，使用其 academic/code/web 检索及原文提取能力交叉核查。

对于公开网页、作者代码和 Hosted 运行时之间的差异，采用以下证据层级：

**实际 Hosted runner / 官方可执行样例 ＞ 官方明确接口文档 ＞ 官方 baseline 源码 ＞ 本地模拟假设。**

不能因为本地模拟可以构造某种 `labeled`，就声称隐藏测试必然会提供它。

### 3.2 需要逐项确认的问题

官方文档目前说明：

```python
predict(input, labeled) -> float
```

其中：

```python
input = [subject, item]
labeled = [
    [[observed_subject, observed_item], observed_label],
    ...
]
```

`labeled` 的条目可能涉及多个 sampled subject–benchmark pairs，item 中存在稳定的匿名 `benchmark_id`。

但必须进一步确认：

**A. Peer 可见范围**

- 一次 `predict()` 调用中的 `labeled` 是否实际包含其他 Subject 的标签？
- Peer 数量和组成由什么决定？
- 不同标签预算下，peer 证据如何变化？
- 是否可能没有任何 peer？
- B0 是否始终完全没有标签？
- 是否能同时看到当前 Subject 的标签与其他 Subject 的标签？

**B. Item 可匹配性**

- 是否存在稳定 item ID？
- 如果没有，能否使用 `item_content`、`item_features`、`interactors` 在不同 Subject 间识别同一题目？
- 相同正文但不同 metadata 或 interactor，是否应视为不同 item？
- 匹配冲突、多对一和无法判定的情况如何处理？
- 当前 target 与 peer acquisition item 之间是否存在实际重叠？

**C. 调用顺序与状态**

- `labeled` 是否是本次调用允许使用的全部证据？
- 是否需要依赖历史 `predict()` 调用顺序或跨调用缓存？
- Hosted 是否保证进程与内存状态连续？
- 是否存在不同预算之间的独立隔离？

不得依赖未经确认的跨调用持久状态。

**D. 合规边界**

只使用本次接口明确提供的属性及已揭示标签。

不得使用隐藏评价标签、未授权私有数据、未揭示的 peer 标签，或借助匿名反馈反向识别隐藏 Benchmark。

### 3.3 运行时合同结果

输出机器可读合同，逐条标注：

- `verified_official`
- `verified_runtime`
- `inferred_not_guaranteed`
- `unknown`
- `unsupported`

优先取得真实 starter runner 或可重复的官方运行时样例。

若公开材料不足以确认全部细节，不应长期停留在网页搜索；应明确列出未决问题，准备简洁的组织者询问文本，并基于**已确认的最低信息合同**设计保守实验。

如果仍无法确认真正的 peer 可见性，跨 Subject 实验只能作为条件模拟，不能宣称 Hosted 端可获得同等收益。

---

## 四、R70：词法修正的预算自适应收缩

### 4.1 核心问题

R64 的词法修正方向有一定预测价值，但高预算下引入更多无效位移。

需要判断：

**预算越大，是否应越保守地使用词法残差修正？这种规律能否在未知 Benchmark 上成立？**

不要简单根据已经看到的外层结果将 B15/B31 关闭。

### 4.2 数学机制

设：

\[
p_{\alpha,n}=p_n+\alpha_n d_n
\]

其中：

\[
0\leq\alpha_n\leq1
\]

\(\alpha_n=0\) 对应 R25，\(\alpha_n=1\) 对应原 R64。

预算 \(n\) 的风险变化：

\[
\Delta B_n(\alpha_n)
=
2\alpha_n A_n+\alpha_n^2 C_n
\]

其中：

\[
A_n=\mathbb E[(p_n-y)d_n]
\]

\[
C_n=\mathbb E[d_n^2]
\]

理论上的最优缩放系数：

\[
\alpha_n^*
=
\operatorname{clip}_{[0,1]}
\left(-\frac{A_n}{C_n}\right)
\]

分母为零时返回零修正。

**这只是数学关系，不允许用外层真实评价标签直接估计最终 \(\alpha_n\)。**

### 4.3 实验设计

首先从现有 R64 预测中计算各预算、各 Benchmark 的：

- 残差对齐项；
- 平方位移成本；
- 两项比值；
- 各预算净收益；
- 不同来源之间的稳定性；
- 主要退化 Benchmark 的贡献。

然后在**训练来源内部**构建 benchmark-grouped out-of-fold 预测，使用其残差估计修正系数。

至少比较以下机制明确的方案：

**方案 A：预算独立收缩**

仅使用来源内 OOF 数据估计每个预算的修正幅度。

**方案 B：单调预算收缩**

检验随支持标签增加，词法修正幅度非增的约束是否能改善泛化。

使用简单的约束优化或低维统计估计，不进行大规模预算权重网格搜索。

**方案 C：R25 与原 R64**

作为必要的固定对照。

允许使用当前完整外层已知最优缩放作为**事后 oracle 诊断**，但不得作为训练过程中的参数选择依据。

### 4.4 主要判别问题

1. 预算增大时，残差对齐下降得比平方位移成本更快吗？
2. 这种变化是否在多个 Benchmark 中一致？
3. 来源 OOF 估计的系数在留出 Benchmark 上是否稳定？
4. 单调收缩约束是否比自由预算系数更可靠？
5. 改善是否仅来自缩小全部预测变化，而非真正利用词法信息？

需要同时检查：

\[
\mathbb E[d^2]
\]

与：

\[
-\mathbb E[(p-y)d]
\]

的相对变化，而不仅看最后的 Brier。

**必须实际执行至少一个事前冻结的预算收缩候选和匹配对照。**

若来源内结果较弱，但候选训练和完整评价成本很低，可以保留一次预声明的探索性外层评价；不能因为一池小幅退化就自动停止。

---

## 五、R71：识别并保留可靠的词法残差修正

这是与 R70 不同的研究问题。

R70 调整的是：

**不同预算整体使用多少词法信息。**

R71 调整的是：

**对于某个可观察条件，当前词法信息是否值得使用。**

### 5.1 核心假设

R64 的词法修正可能包含两种成分：

\[
d=d_{\mathrm{useful}}+\epsilon
\]

其中：

- \(d_{\mathrm{useful}}\)：能够纠正 R25 残差的部分；
- \(\epsilon\)：模板、无关 token、偶然匹配或错误局部关联带来的扰动。

这是待检验的结构性假设，不预设已经找到了这两个成分。

需要研究：

**哪些合法可观测条件，能够可靠区分词法修正的有用性？**

### 5.2 首先做冻结预测诊断

对 R64 的逐 item 修正，计算：

\[
g=(y-p)d
\]

和：

\[
c=d^2
\]

其中 \(g\) 越大，表示修正与真实残差越一致；\(c\) 表示位移代价。

注意：\(g\) 使用了评价标签，因此**只能用于离线训练/诊断和合法 OOF 监督，不能作为测试时的输入特征**。

从现有合法信息中寻找少量具有明确语义的候选可靠性变量，例如：

- 当前 query 与支持 item 的词法相似度；
- 词法最相似支持样本与语义最相似支持样本是否一致；
- 词法匹配的 top-1/top-2 margin；
- 有效支持匹配数量；
- 模板去除后实际保留的差异 token 数量；
- 支持标签的一致性；
- 语义与词法修正的方向冲突；
- 当前标签预算。

这些只是候选。必须首先核对现有实验是否已经测试过等价信号，不重复已有 R43、R45/R46 的失败设计。

不能基于已查看的外层收益反复筛选出恰好有利的特征。

### 5.3 研究算子

选择：

**Residual Alignment Analysis → Reliability Stratification → Conservative Correction。**

在训练来源 OOF 数据上，考察不同可观察条件 \(z\) 下：

\[
\mathbb E[(y-p)d\mid z]
\]

与：

\[
\mathbb E[d^2\mid z]
\]

的关系。

若证据支持，可以建立：

\[
p'=p+\alpha(z,n)d
\]

并限制：

\[
0\leq\alpha(z,n)\leq1
\]

最小实现优先采用：

- 少量预定义条件分组；
- 基于 OOF 残差的保守修正估计；
- 小样本或未知条件下回退 R25；
- 避免高维学习的可靠性门控。

如果训练来源没有稳定的条件收益，不应直接训练新的可靠性 MLP。

### 5.4 必要对照与判别实验

至少比较：

1. R25；
2. R64 原固定词法组合；
3. R70 预算收缩；
4. R71 条件可靠性修正；
5. 在机制证据支持时，R70 + R71 的简单组合。

**R70 + R71 不得默认相加。** 必须先证明条件机制能带来超出单纯预算收缩的增量。

实施以下关键诊断：

- 保留词法 item 对应关系 vs 打乱合法支持标签对应关系；
- 相同修正幅度下，可靠性选择是否优于随机选择；
- 条件门控是否只是整体降低 \(d^2\)；
- 真正与残差对齐的修正是否被保留下来；
- 不同 Benchmark 中该条件是否具有稳定方向。

置换必须重新计算所有依赖已观察支持标签的 R25 与词法修正，严格保持预算、身份与响应计数的对照。

在裁剪、归一化与重复 response 权重可能影响诊断时，明确记录相关限制，不能把置换结果直接当作无条件因果证明。

### 5.5 停止条件

如果：

- 可靠性信号不能在多个留出 Benchmark 中保持方向；
- 条件修正不优于简单预算收缩；
- 改善只在事后 oracle 分组中出现；
- 需要目标评价标签才能识别修正何时有利；

则停止当前条件选择路线。

保留有价值的误差分解认识，不继续扩大门控网络或扫描大量阈值。

---

## 六、R72：跨 Subject 当前标签信息的实际实验

本阶段必须根据 R69 已确认的合同执行。

### 6.1 重新定位研究问题

R20/R36 已经发现：

- 在某些共享场景下，其他 Subject 的标签能够提供 item 相关信息；
- 但相对于强 own-only 对照，R36 的增量仅约 0.0000717，一折退化，聚类置信区间跨零。

所以本轮不能简单重跑原 R20 pooled item 模型。

需要检验：

**当合法的 `labeled` 中真实提供其他 Subject 的标签时，这些标签能否为 R25 提供其当前支持样本之外的独立预测信息？**

这里至少区分：

- 同一 item 的 peer outcome；
- 同一 Benchmark 不同 item 的 peer evidence；
- Peer 反映的共同 item 难度；
- Peer 仅反映不同 Subject 的整体能力或成功率。

不能把它们混成一个共享均值。

### 6.2 建立合同一致的数据回放

从原始公开数据重建合法 acquisition/evaluation 分割。

必须满足：

- 只暴露 acquisition pool 的标签；
- 不暴露任何 evaluation response；
- 保持各标签预算及嵌套标签集合；
- 保留同 item 重复 response 的实际记录及计数语义；
- 保证 target 自身支持标签与 peer 标签严格区分；
- Peer 组成不得超出正式接口已确认的可见范围；
- 不依赖隐藏调用顺序或未经确认的跨调用持久状态。

对于同 item 匹配，优先使用官方明确提供的稳定标识；若不存在，才研究通过允许字段构造严格、保守、可审计的 item identity。

相同文本但 metadata/interactors 不同的情况，必须单独检查。

不可确认的匹配不应强行合并。

### 6.3 首先测量真正的共享信息覆盖率

在实际模型实验前，报告：

- 每个 Benchmark 有多少 Subject；
- 每个 target pair 可以看到多少 peer labels；
- 其中多少 peer 标签与 target query 属于同一 item；
- 同 item 的真实匹配成功率；
- 不同预算的 peer 覆盖分布；
- peer 缺失或无法匹配的比例；
- 某些 Benchmark 是否完全没有共享机会；
- 提供的 peer 信息与目标成功率之间是否具有可迁移的预测关系。

必须区分：

**信息在公开完整数据库中存在**

与：

**信息在正式某次 `predict()` 调用中可被观察。**

如果 hosted 样例仍无法确定 peer 组成，只能建立明确标记为 `conditional_simulation` 的保守场景，并单独给出覆盖率敏感性分析；不得把它的分数当作真实 Hosted 可达到的收益。

### 6.4 最小预测方法

优先选择简单、可解释、与 R25 互补的机制。

**候选 A：同 item peer correction**

在确实有同 item peer 标签时，利用它们估计 item-level residual 或共同难度信息。

需要控制 Subject 能力差异，避免直接把其他 Subject 的成功率当作当前 Subject 的成功率。

**候选 B：共享 Benchmark 局部结构**

只有当 peer item 不完全重合、但具有可靠内容对应关系时，考虑将其作为额外的局部预测信息。

使用冻结 MiniLM 与现有 R25 表示，不重新训练大型编码器。

**候选 C：保守共享校准**

当有效 peer 数很少、信息不一致或匹配不可靠时，退回 R25。

具体修正形式应由 R20/R36 及本轮覆盖率、残差诊断决定，不预先指定复杂网络。

### 6.5 关键消融和控制

至少需要：

- R25 own-only；
- R20/R36 原强对照；
- 同 Benchmark、同 item peer；
- 同 Benchmark、不同 item peer；
- 正确 item 对应与同预算置换对应；
- 有 peer / 无 peer / 低覆盖 peer 场景。

对比真实共享修正与仅增加标签数量所带来的普通平滑收益。

特别检查：

\[
\text{Additional Peer Information}
\]

是否真的包含 R25 当前支持标签中没有的信息，而不是重复统计或混入未来标签。

若实验显示收益只存在于高覆盖、但 Hosted 不保证的场景，应将其保留为条件证据，不晋级部署。

---

## 七、统一实验协议、探索与晋级规则

### 7.1 先来源内，再完整外层

R70/R71 的参数和条件规则必须通过训练来源 OOF 数据拟合。

对外层每个目标 Benchmark，训练、标准化、模型选择、规则学习和校准均不能接触该目标的评价标签。

R72 必须额外满足已核实的共享输入合同。

所有候选在完整外层运行前，冻结：

- 数学规则；
- 超参数和选择方式；
- 训练来源；
- 预测特征；
- 标签可见范围；
- 随机种子；
- 实现源码与模型版本。

### 7.2 完整公开探索评价

符合合法性与实现条件的候选，执行：

**5,242 pairs × 3 item-split seeds × 六预算。**

主要指标为：

\[
ALC=
0.1B_0+0.2B_1+0.2B_3+
0.2B_7+0.2B_{15}+0.1B_{31}
\]

每 pair 等权、pair 内保留重复 response 权重。

报告：

- Pair-equal ALC；
- Benchmark-equal ALC；
- 三fold差异；
- 六预算 Brier；
- Benchmark-cluster paired bootstrap；
- `alignment_faking` 与其他 Benchmark 的贡献；
- 最大 Benchmark 回归；
- 预测正确率（仅作辅助）；
- 与 R25、R56_segment 及 frozen smoothed mean 的比较；
- 实际 CPU 开销与 Hosted 可部署性。

**探索宽进，晋级严出。**

允许有明确机制依据、执行成本低的候选进入一次探索性外层，即使来源内结果不完全一致。

但不得使用已查看的外层结果反复修改预算系数、可靠性条件、peer 匹配或共享强度，再把同一外层当作新的确认集。

### 7.3 原研究 Gate 不变

仍要求：

- 相对冻结 smoothed mean 和强训练来源基线达到至少 0.005 ALC 的绝对改善；
- 三fold改善；
- Benchmark-equal改善；
- Benchmark-cluster CI 上界低于零；
- 没有未解释的严重 Benchmark 回归。

对于未达到原 Research Gate、但出现较小且合理收益的方法，单独记录为竞赛探索候选，不自动否定其可能的工程价值。

原 R25、R56_segment 及 `dist/smoothed_mean_strength2.zip` 保持不变，不自动正式提交。

---

## 八、执行顺序与资源控制

本轮采用以下实际顺序：

**R69：** 优先检索并核实官方共享标签运行时合同；形成可验证事实、未知项和保守实验输入边界。

**R70：** 执行词法预算收缩的冻结预测分析、来源 OOF 系数估计和实际验证。

**R71：** 执行词法可靠性分组、残差对齐与位移成本分析，并验证至少一个证据支持的保守选择候选；评估其相对 R70 的独立收益。

**R72：** 在 R69 允许的信息条件下执行跨 Subject 覆盖率审计、强 own-only 对照和最小共享预测实验。

各阶段不要因为其他阶段暂时受阻而停止整个任务。

例如 Hosted 契约未完全确认时，R70/R71 仍应继续完成；R72 则完成已确认条件下的分析与模拟，并如实报告其边界。

所有新训练使用纯 CPU。优先复用冻结 MiniLM、R25、R64 及已有特征缓存。

禁止：

- 无依据地扩大架构；
- 大规模扫描融合权重或门控阈值；
- 使用目标评价标签进行模型选择；
- 把事后 oracle 当作合法提交结果；
- 把运行时不可获得的 peer 标签当作真实输入；
- 重复已经充分验证失败的原 R43/R55/R61 配置。

---

## 九、交付要求

建立新轮次 R69–R72 的独立目录，不覆盖历史记录。

建议：

- `reports/lexical_peer_evidence_findings.md`：主报告，机制、结果、风险与下一步。
- `reports/lexical_peer_evidence_tables.md`：六预算、三fold、逐 Benchmark、词法成本与收益分解、peer 覆盖率。
- `research_loop/LEXICAL_PEER_EVIDENCE_SPEC.md`：事前冻结的实验合同。
- `research_loop/PAIEC_RUNTIME_EVIDENCE_CONTRACT.md`：官方证据、代码定位、真实运行时已确认和未知字段。
- `research_loop/LEXICAL_PEER_OPERATOR_DECISION.md`：研究算子选择及进入/停止理由。
- `runs/research_loop/R69–R72/`：真实预测、权重、计数、指标、置换、来源隔离、代码 hash 与审计。
- 更新 `research_loop/HIGH_LEVEL_STATE.md` 与历史信号清单。

文件名和执行命令以实际仓库结构为准，不得凭空声称脚本已经存在。

报告按照以下顺序组织：

1. **Existing Positive Evidence**：R64、R20/R36 的正面信息。
2. **Remaining Unknown**：平方位移成本、可靠性和 peer 可用性究竟缺少什么证据。
3. **Runtime Contract Findings**：官方确认、实际观察、未确认条件。
4. **Mechanism Experiments**：预算收缩、词法残差筛选、跨 Subject 信息。
5. **Numerical Results**：完整对照与不确定性。
6. **High-Level Understanding**：新增信息、表示、条件可靠性与迁移的关系。
7. **Research / Submission Decision**：保留、停止、晋级或等待外部条件。
8. **Next Move**：下一轮最有信息价值的具体研究问题。

特别回答三个最终问题：

**Q1：词法修正是否存在可以由预算变化规律稳定解释的收益？**

**Q2：能否从合法可见条件中识别词法修正真正有利的部分，使残差对齐收益超过新增平方位移成本？**

**Q3：正式 PAIEC 输入是否提供可用的跨 Subject 当前标签证据？若提供，它能否在强 R25 own-only 对照下形成独立、可迁移的概率预测增量？**

如果答案是否定的，也应明确说明是哪一层失败：

- 信息本身不足；
- 信息存在但不能可靠估计；
- 来源内有效却无法跨 Benchmark 迁移；
- Hosted 输入条件不支持；
- 方法有局部收益但不足以改善主指标。

不能将这些失败合并成“整个研究方向无效”。

---

## 十、现在开始执行

首先执行 R69 官方运行时契约核查，并与本地 R68 audit 对照。

之后开始 R70/R71 两项词法实验，优先从 R64 冻结预测、支持标签和逐 response 计数恢复真实的残差对齐与平方位移成本。

同时为 R72 建立实际可见的 peer 数据合同；只有确认标签可用范围，才运行对应的真实共享预测评价。

**本轮研究成功的关键，不是简单找到一个更小的融合系数或更多 peer 标签，而是验证能否通过合法可观测信息，将已存在的局部正面信号可靠地转化为跨 Benchmark Brier ALC 改善。**

实际完成后，给出真实分数、机制认识、证据边界和下一步决策；不只返回代码或执行状态。