# PAIEC Research Prompt：R80–R83——Partial8 共享机制归因、有限支持标签风险辨识与保守共享决策

## 一、研究角色与目标

你是一名 **Multi-Task Learning Researcher、Statistical Decision Theory Expert、OOD Generalization Diagnostician、Scientific Experiment Designer 与 ML Systems Engineer**。

当前 PAIEC 项目已完成 R1–R79。

本轮只围绕 Partial8 的两个尚未解决的问题开展研究：

**问题 A：Partial8 为什么能够稳定优于 Fully Pooled？**

判断其增量究竟来自 Subject-specific 斜率方向，还是主要来自正则化、收缩幅度和有限样本方差变化。

**问题 B：能否仅利用当前合法支持标签，更准确地识别共享对目标 Subject 的收益或损失？**

尤其检查 R78 半支持集诊断与真正 Full-Support Query 风险之间的差异，以及该差异是否导致 R79 的错误关闭决策。

本轮目标不是立刻训练新的门控，而是：

**识别 Partial8 当前优势的真实来源，检验共享收益符号是否具有统计可辨识性，并且只在证据充分时构造一个能够减少负迁移的最小共享决策方案。**

采用：

**Observed Phenomenon → Competing Explanations → Discriminating Experiment → Mechanism Judgment → Minimal Intervention → OOD Evaluation → Decision。**

必须实际执行实验，不只返回计划。

---

## 二、恢复研究状态

项目目录：

`D:\nips-comp\nips-comp\paiec_baseline`

首先阅读：

- `research_loop/SHARED_RELIABILITY_DECISION.md`
- `reports/shared_reliability_tables.md`
- `reports/shared_cohort_robustness_findings.md`
- `reports/support_sharing_reliability_findings.md`
- `reports/shared_reliability_policy_findings.md`
- `reports/shared_reliability_reproduction.md`
- `reports/dual_track_demand_sharing_findings.md`
- `reports/shared_slope_tables.md`
- `research_loop/HIGH_LEVEL_STATE.md`

检查 R72、R75–R79 的真实代码、支持 mask、cohort、逐 pair 概率、response counts、模型预测及来源选择合同。

重点恢复：

`runs/research_loop/R77/`

`runs/research_loop/R78/`

`runs/research_loop/R79/`

### 当前冻结基准

所有共享成绩仍是 `conditional_simulation`。

| 方法 | ALC ↓ |
|---|---:|
| R25 own-only | 0.195123926 |
| R76 三 peer Fully Pooled | 0.190728761 |
| R76 Partial8 | **0.190547856** |

R77 五套冻结 cohort：

\[
\Delta ALC_{\mathrm{partial-pooled}}
=-0.000179972
\]

95% Benchmark-cluster CI：

\[
[-0.000201013,-0.000040292]
\]

五套 cohort 全部改善，但约 93.77% 的增量来自 `alignment_faking`，`dave` 仍存在明显负迁移。

R78 发现：

- Partial 支持风险与 Query 收益的平均 Spearman ≈0.241；
- 聚合层 CI 全正，但单独 B15/B31 的 CI 跨零；
- 支持风险认为“可靠”的组，Query Brier 改善约0.005689；
- 支持风险认为“不可靠”的组，Query Brier 仍改善约0.000846。

R79 二值门控：

\[
\Delta ALC_{\mathrm{gate-partial}}
=+0.000020318
\]

错误关闭有效共享的损失大于避免有害共享的收益，且未优于简单收缩。

**R79 当前配置已经失败，不能在下一轮换一个阈值重新扫描。**

---

# R80：Full-Support 风险可辨识性审计

## 三、要回答的问题

R78 使用大约一半 own-support 标签拟合，再在另一半标签上计算相对风险。

然而正式 Query 预测使用的是全部预算内支持标签。

因此：

\[
z_{\mathrm{half-support}}
\]

与：

\[
\Delta_{\mathrm{full-support-query}}
\]

可能对应不同的统计估计问题。

本轮首先判断：

**R78 的弱风险排序以及 R79 的错误回退，有多少可能来自 Half-Support → Full-Support 的分布与估计条件不一致？**

不得直接假设它就是主要原因。

## 四、竞争假设

**H1：Sample-size mismatch**

半支持集拟合的 own/shared 风险差不能准确反映完整支持集预测的相对风险。

**H2：High-variance risk estimation**

支持标签最多31个，二值标签带来的估计噪声过大，难以可靠判断共享收益符号。

**H3：Support–Query distribution mismatch**

即使风险估计准确，已揭示的支持 item 与待预测 query item 之间仍可能存在表示分布差异。

**H4：Genuine subject-level heterogeneity**

部分 Subject 的共享方向确实不适用，其风险差能够通过合法支持标签识别。

这些假设可以同时成立，必须区分。

## 五、实际实验

### R80-A：比较支持风险估计方式

在相同来源 Benchmark、cohort 和预算下比较：

1. R78 原始 Half-Support Masked CV；
2. Leave-One-Item-Out Cross-Validation（LOIO）；
3. 复用大部分支持标签的 Grouped K-Fold Cross-Fitting；
4. Own-only 与 Partial8 的相同验证划分对照。

优先预算：

\[
n\in\{15,31\}
\]

B7/B3 只有在现有证据显示估计可行时才补充。

所有预测必须遵守：

- 只能使用预算内已经揭示的标签；
- 留出的 own item 不参与其拟合、中心化和参数选择；
- 同一 item 的重复 response 必须整体分组，避免标签泄漏；
- Peer 只使用当前合法预算前缀；
- 保留原 R25、Fully Pooled、Partial8 的中心化及求解器；
- 同一验证样本对各方法公平使用；
- 不使用 Query outcome 构造运行时特征。

LOIO 与 K-Fold 的目标是检验风险估计的偏差、稳定性和样本利用效率，不预设它们优于 Half-Support。

### R80-B：Full-Support 风险校准

针对各支持风险估计器 \(z\)，检查：

\[
z
\longrightarrow
\mathbb E[\Delta_{\mathrm{query}}\mid z]
\]

其中：

\[
\Delta_{\mathrm{query}}
=
L_{\mathrm{Partial8}}-L_{\mathrm{R25}}
\]

Query 标签仅用于合法来源验证中的目标风险评价。

必须分别分析：

- 对 Query 收益大小的排序；
- 对收益正负的判别；
- 对风险差数值的校准；
- 不同 Benchmark 的稳定性；
- Mask 重复带来的方差；
- 有限标签下估计区间的宽度；
- B15 与 B31 是否有系统性差异。

重点区分：

**能判断哪个 Subject 更可能受益**

与：

**能判断某个 Subject 使用 Partial8 是否会造成净损失。**

前者不能替代后者。

### R80-C：可辨识性边界

根据重复 mask、合法支持标签重采样和来源 Benchmark 分组，估计：

- 对共享风险符号的错误判断概率；
- 估计符号在不同 mask 间的稳定性；
- 需要多少标签才能减少符号不确定性；
- 有多大比例的请求无法形成可靠判断。

明确区分：

- 数据确实支持共享有害；
- 数据只支持共享收益较小；
- 证据不足，无法判断。

第三类不应强行归为共享有害。

必要时使用已知机制的合成数据作校准诊断，但合成数据不能代替真实 PAIEC 证据。

### R80 交付结论

回答：

**更充分复用支持标签，是否能够显著改善共享收益符号的可辨识性？**

如果只是提高相关系数，而收益正负仍无法可靠判断，应停止基于此类风险估计的决策开发。

---

# R81：Partial8 的结构增量究竟来自哪里？

## 六、核心问题

R77 已经证明 Partial8 相对 Fully Pooled 在五套 cohort 中稳定改善。

但它同时使预测修正的平方位移成本降到 Fully Pooled 的约93.8%。

因此存在两个竞争解释：

**解释 A：Subject-specific slope correction**

\[
\beta_s=\beta_{\mathrm{shared}}+\delta_s
\]

\(\delta_s\) 提供了与真实 Subject 响应差异相关的额外方向信息。

**解释 B：Regularization / Shrinkage**

Partial8 主要减少不可靠的预测位移，并没有有效恢复 Subject-specific 方向。

R77 的来源拟合幅度收缩对照没有完全排除解释 B。

## 七、实际机制实验

### R81-A：精确的方向与幅度分解

对相同 cohort、同一 Query item，定义：

\[
d_{\mathrm{pool}}=p_{\mathrm{pool}}-p_{\mathrm{own}}
\]

\[
d_{\mathrm{partial}}=p_{\mathrm{partial}}-p_{\mathrm{own}}
\]

分析：

- 两者预测方向一致的比例；
- Partial8 改变修正方向的比例；
- 仅缩小修正幅度的比例；
- Partial8 新增的方向差异；
- 不同预算、Subject、Benchmark 上的变化；
- Clip 前后方向和幅度变化的区别。

只使用已有冻结预测及必要的合法特征，不训练新模型。

### R81-B：更严格的收缩对照

建立两个互补对照：

**对照1：来源拟合的全局幅度收缩**

沿用 R77 已有对照，检查其与 Partial8 在预测位移平方成本和实际风险上的差异。

**对照2：匹配预测位移大小的方向控制**

在不使用 Query outcome 的前提下，使控制方法与 Partial8 尽可能具有相同的预测修正幅度或平方位移成本，但保留 Fully Pooled 的修正方向。

记录由于概率边界裁剪造成的无法精确匹配部分。

这一对照首先用于**离线机制诊断**。若使用了正式单次调用不保证可见的整组 Query 信息，则不得作为可部署预测器。

重点回答：

**当平方位移成本得到控制后，Partial8 的方向改变是否仍具有额外 Brier 收益？**

### R81-C：共享斜率异质性的判别

结合 R75 的异质性诊断，检查：

- Subject-specific 偏移是否在不同支持划分中具有可重复方向；
- 偏移是否能预测已揭示 own 标签的残差；
- Peer 标签置换后这种关系是否消失；
- 固定共同斜率的合成 Null 下，Partial8 是否仍获得类似的收益；
- 显式加入 Subject-specific 斜率差异时，Partial8 的机制表现是否符合预期。

合成实验应复用真实特征几何与标签预算，明确标记为机制检验。

不能根据合成结果宣称在真实 PAIEC 中已经恢复了真实 Subject 能力结构。

### R81 交付结论

分别判断：

1. Partial8 是否稳定减少平方位移成本；
2. 是否稳定改变预测修正方向；
3. 方向改变是否带来独立收益；
4. 当前证据是否足以证明 Subject-specific slope 是主要收益来源。

如果仅能证明它是一种稳健的结构化正则化器，也应如实保留这一正面结论。

不必为了证明更强机制而继续扩大模型。

---

# R82：共享收益符号的保守校准

## 八、进入条件

只有 R80 显示：

- 合法支持风险估计能够提供稳定的收益符号信息；
- 相对于 R78 的半支持估计有实质改善；
- 该信息在 Benchmark 分组验证中具有可迁移证据；

才进入 R82。

否则记录：

`sharing_sign_not_identifiable`

R82 及其后续门控实验停止。

## 九、研究目标

R79 的失败并不是没有风险排序信息，而是错误关闭有效共享代价过高。

因此不再采用：

\[
g=\mathbf 1[z<0]
\]

而是研究：

**能否通过来源内的风险差校准及不确定性估计，仅在有足够证据证明共享可能造成损失时，才修改默认 Partial8？**

默认行为仍然是 Partial8。

### R82-A：风险差估计

根据 R80 中已通过验证的合法统计 \(z\)，在 Benchmark-grouped OOF 来源数据上估计：

\[
m(z,n)
=
\mathbb E[
L_{\mathrm{Partial8}}-L_{\mathrm{R25}}
\mid z,n
]
\]

其中 \(n\) 为当前标签预算。

同时估计：

- 风险差不确定性；
- 不同 Benchmark 的预测误差；
- 对真正共享有害案例的识别率；
- 错误关闭有效共享的预期损失。

禁止使用目标外层 Query outcome 校准。

### R82-B：决策可行性审计

比较：

- 预测共享有害的概率；
- 风险差期望值；
- 风险差置信区间；
- 决策造成的错误关闭成本；
- 不修改 Partial8 的基准效用。

特别检查：

\[
\mathbb E[\Delta_{\mathrm{query}}\mid z,n]
\]

的零点是否具有可校准性。

如果估计区间普遍跨越零，或者所需支持标签数超过预算，则明确宣布无法作出可靠判断。

**不允许因为存在弱相关性，就强行构造新的门控。**

---

# R83：条件执行的最小保守共享策略

## 十、触发规则

只有 R82 证明来源外收益符号能够在部分请求中得到可信辨识，才进入 R83。

此时冻结一个最简单的策略。

建议机制为：

**Partial8 默认开启；只有共享风险为正具有预声明的统计证据时，才回退 Own-only 或减弱共享。**

具体不确定性水平、规则和模型选择必须在评分前，通过合法来源确定。

不能扫描大量阈值。

### 必要对照

1. R25 Own-only；
2. R72 Fully Pooled；
3. R76 Partial8；
4. R79 已失败的二值门控；
5. 来源拟合的简单全局收缩；
6. 新 R83 保守共享策略。

重点检验：

**新策略是否比 Partial8 和匹配收缩对照更好，而不是只比 R25 好。**

### 来源验证

采用严格 Benchmark-grouped nested validation：

- 来源 fit 组估计风险校准；
- validation 组评价决策效果；
- 目标 Benchmark 不参与校准或阈值选择；
- 同一候选固定后跨来源比较；
- 不根据已知 `dave` 回归制定 Benchmark 例外规则。

要求至少检查：

- Pair-equal ALC；
- Benchmark-equal；
- 风险差校准；
- 关闭共享比例；
- 正确避免的损失；
- 错误关闭造成的收益损失；
- 与 matched-mean、matched-cost 对照的增量。

### 外层探索进入

不要求所有来源都必须严格改善，但应至少具有：

- 多个来源方向一致的证据；
- 总体配对 Brier 改善；
- 相对 Partial8 和简单收缩的独立增量；
- 没有新的严重、无法解释的来源回归；
- 合法可部署的特征与计算预算。

不满足则停止 R83，不运行新的完整外层。

满足后，冻结唯一候选，执行原协议：

**5,242 pairs × 3 item-split seeds × 六预算。**

不得根据外层结果继续修改风险阈值或选择新的 Gate。

---

## 十一、统一证据和评价要求

### 1. 信息隔离

所有预测与风险特征仅使用当前条件模拟允许的：

- Own acquisition labels；
- Peer acquisition labels；
- 合法 item/subject 字段；
- 当前预算；
- 当前调用的可见信息。

Query success/outcome 只能用于离线评价。

严禁：

- 目标 Query 标签进入规则拟合；
- 当前支持标签被留出后又间接进入中心化；
- 使用内部 hidden subject/item ID 作为预测特征；
- 用 Benchmark 名称识别应当关闭共享的情况；
- 依赖未经确认的跨调用缓存。

### 2. 统计解释

明确区分：

- 来源诊断；
- 来源内 OOF 验证；
- 完整公开外层探索；
- 合成机制实验；
- 真实 Hosted 评价。

五套 cohort 共享 Benchmark，不是五个独立测试数据集。

三次 item-split seed 不是三次训练初始化。

Bootstrap 以 Benchmark 为主要 cluster，保留同一 Benchmark 内的 pair、seed、cohort 重复测量依赖。

公开 Benchmark 已被反复使用，新的外层评价仍不能称为独立盲测。

### 3. 主要指标

继续使用 Pair-equal Brier ALC：

\[
ALC=
0.1B_0+0.2B_1+0.2B_3+
0.2B_7+0.2B_{15}+0.1B_{31}
\]

必须同时报告：

- 三个 outer fold；
- Benchmark-equal；
- Benchmark-cluster CI；
- 各标签预算；
- `alignment_faking` 与其余 Benchmark；
- 最大严重回归；
- Peer 数量与实际额外标签成本；
- 预测正确率，仅作为辅助指标；
- CPU 运行时间、峰值内存。

### 4. 研究 Gate

原 Research Success Gate 保持不变：

- 相对两种强均值 baseline 绝对改善至少0.005；
- 三 fold 改善；
- Benchmark-equal 改善；
- Benchmark-cluster CI 上界低于零；
- 没有未解释的严重回归。

Shared-Peer 还需要真实 Hosted 输入合同验证。

即使 R83 通过公开数值标准，也只标记为 `conditional_simulation`，不自动部署。

---

## 十二、执行顺序

采用以下执行顺序：

**R80：必须执行。**

补齐 Half-Support、LOIO 与 Cross-Fitting 对照，判断风险估计偏差、方差及收益符号可辨识性。

**R81：必须执行。**

在匹配信息条件下分析 Partial8 的方向与幅度，完成更严格的收缩对照和必要的异质性机制检验。

**R82：条件执行。**

只有 R80 出现足够的来源外风险符号信息，才研究保守的风险差校准与不确定性。

**R83：条件执行。**

只有 R82 证明可以作出有统计依据的共享决策，才构造并验证一个新策略。

若 R80/R81 已经提供明确停止证据，不得为了完成轮次强行训练或评分。

---

## 十三、交付要求

在项目中建立新的 R80–R83 研究资产，不覆盖 R1–R79。

建议：

- `reports/partial8_identifiability_findings.md`
- `reports/partial8_mechanism_findings.md`
- `reports/partial8_decision_tables.md`
- `research_loop/PARTIAL8_NEXT_SPEC.md`
- `research_loop/PARTIAL8_NEXT_DECISION.md`
- `runs/research_loop/R80/`
- `runs/research_loop/R81/`
- `runs/research_loop/R82/`（触发时）
- `runs/research_loop/R83/`（触发时）

更新：

- `research_loop/HIGH_LEVEL_STATE.md`
- 现有正负证据信号 ledger

每阶段保存完整：

- 实验合同；
- 模型/数据/代码 hash；
- OOF 分割和 support mask；
- Peer cohort 与合法标签前缀；
- 原始预测概率；
- response counts/successes；
- 配对风险及不确定性；
- 机制对照；
- 实际资源开销；
- 独立概率重计分；
- 未触发原因及失败留痕。

执行完成后按下面的认知层次回答：

**Q1：Partial8 是否真正利用 Subject-specific 方向，还是主要发挥幅度正则化作用？**

**Q2：相比 R78 的半支持预测，LOIO / Cross-Fitting 能否更可靠估计完整预算下共享收益的符号？**

**Q3：当前支持标签数量是否足以作出统计可靠的共享/回退决策？是否存在可辨识性边界？**

**Q4：如果可以估计风险，是否能在减少错误共享的同时避免 R79 式错误关闭收益损失？**

**Q5：下一轮应该继续研究共享风险校准、改进 Partial8 的估计结构，还是保留固定 Partial8 并停止当前分支？**

必须区分原文/文献支持、实际实验、模型推论及尚未验证的猜想。

---

## 十四、现在开始

首先执行 R80 与 R81 的冻结机制实验，不直接训练新门控。

本轮最重要的研究问题是：

\[
\boxed{
\text{Useful Reliability Ranking}
\neq
\text{Identifiable Benefit Sign}
\neq
\text{Improved Decision Utility}
}
\]

R78 已发现第一层的有限正面证据，R79 说明第三层尚未实现。

**现在应当研究中间缺失的“收益符号可辨识性”，以及 Partial8 究竟在利用什么结构，避免在没有新增证据的情况下继续进行阈值、模型规模或正则强度搜索。**

任务完成后提供真实实验结果、机制认识、决策及下一步建议，而不只是执行日志。