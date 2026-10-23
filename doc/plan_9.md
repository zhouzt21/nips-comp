# PAIEC Research Prompt：R77–R79——Partial Sharing 跨 Peer 组成稳健性验证与错误共享的可观测诊断

## 一、研究角色与核心目标

你是一名 **Probabilistic ML Researcher、Multi-Task Learning Expert、Statistical Generalization Researcher、Scientific Failure Diagnostician 与 ML Experimental Engineer**。

当前 PAIEC 已完成 R1–R76。

本轮只针对 Shared-Peer 路线，补齐两个关键实验：

**实验 A：Peer-Cohort Robustness**

检验 R76 Partial8 相对完全共享斜率的增量，能否在不同 peer 组成、缺失情况与标签覆盖条件下保持。

**实验 B：Support-Based Sharing Reliability**

检验能否仅利用当前合法 acquisition/support 标签，提前识别哪些 Subject 的共享斜率可靠、哪些可能发生负迁移。

本轮最终目标不只是降低平均 Brier ALC，而是回答：

> **共享斜率的收益能否稳定迁移到不同 peer 组成？如果部分 Subject 不适合共享，能否通过预测时合法可见的信息提前发现，并保守回退？**

必须实际执行条件允许的实验，而不是只制定计划。

继续采用：

**Observed Failure → Competing Explanations → Discriminating Experiments → High-Level Understanding → Minimal Intervention → OOD Validation → Research Decision。**

不开展新的广泛文献调研、不扩大神经网络、不扫描 Partial Sharing 正则系数。

---

## 二、恢复现有研究状态

工作目录：

`D:\nips-comp\nips-comp\paiec_baseline`

首先阅读：

- `reports/dual_track_demand_sharing_findings.md`
- `reports/shared_slope_tables.md`
- `reports/shared_slope_findings.md`
- `reports/shared_slope_reproduction.md`
- `reports/lexical_peer_evidence_findings.md`
- `reports/lexical_peer_evidence_tables.md`
- `research_loop/SHARED_SLOPE_CONTRACT.md`
- `research_loop/HIGH_LEVEL_STATE.md`
- R72、R75、R76 的实际模型、预测代码、来源数据、cohort 构造、权重及原始结果

重点检查：

- `runs/research_loop/R75/`
- `runs/research_loop/R76/`
- `research_loop/shared_slope_adapter_cpu.py`

根据实际文件组织定位实现，不凭空假设脚本入口。

### 当前冻结基准

完整公开评价为 5,242 pairs × 3 item-split seeds、六档预算：

| 方法 | ALC ↓ |
|---|---:|
| R25 own-only | 0.195123926 |
| R72 pooled3（旧 cohort） | 0.190683911 |
| R76 同 cohort 完全共享 | 0.190728761 |
| **R76 Partial8** | **0.190547856** |
| R76 另一冻结 cohort 完全共享 | 0.190667308 |

R76 Partial8 相对同一 cohort 完全共享：

\[
\Delta ALC=-0.000180905
\]

Benchmark-cluster 95% CI：

\[
[-0.000202247,-0.000038056]
\]

三个外层 fold 均改善，Benchmark-equal 也有小幅改善。

但仍存在以下问题：

1. Partial8 仅在主 cohort 上完成了完整评价，另一 cohort 目前只有完全共享结果。
2. 大约 96.98% 的 Partial8 相对 R25 的总体收益来自 `alignment_faking`。
3. `dave` 上相对 R25 退化约 0.006633。
4. Partial8 允许 Subject-specific slope，却没有可靠判断是否应该共享的机制。
5. Hosted peer 实际可见性仍未确认，全部 Shared-Peer 分数仅属于 `conditional_simulation`。

### 冻结结构

当前 Partial8 采用：

\[
\hat p_{s,i}
=
\operatorname{clip}
\left(
\bar y_s+
\phi(x_i)^\top(\beta+\delta_s),
0,1
\right)
\]

对应的平方损失结构为：

\[
L=
SSE+\|\beta\|_2^2+
8\sum_s\|\delta_s\|_2^2
\]

其中：

- \(\beta\)：跨 Subject 共同斜率；
- \(\delta_s\)：当前 Subject 的斜率偏移；
- \(\bar y_s\)：由当前 Subject 自己的支持标签估计的成功率基准。

必须恢复真实代码中的中心化、核求解、裁剪和计数处理，不能仅根据上述示意公式重新实现一个不等价模型。

---

# 实验 A：Partial8 的 Peer-Cohort 稳健性验证

## 三、R77：研究问题与假设

核心问题：

**R76 Partial8 的增量究竟来自具有迁移性的部分共享机制，还是特定 peer 组成带来的偶然优势？**

检验以下竞争假设。

**H1：稳定结构增益**

Partial8 能在多种合法 peer 组成下持续优于对应的完全共享方法。

**H2：Peer-composition dependence**

增益依赖某些 peer 的响应方向、数量或标签覆盖情况；改变 peer 组成后可能消失。

**H3：Shared-slope heterogeneity**

只有存在明显跨 Subject 响应异质性的 cohort，Partial8 才能体现额外价值。

**H4：Regularization benefit rather than heterogeneity adaptation**

Partial8 的作用可能主要是降低联合估计的有效修正幅度，而不一定准确识别了 Subject 特异方向。

这些假设并非互斥，需要通过匹配实验区分。

## 四、R77：实际执行

### A1. 首先补评已经冻结的第二套 Cohort

R76 已经对另一套冻结 cohort 执行过完全共享评价：

\[
ALC=0.190667308
\]

但尚未提供相同 cohort 下的 Partial8 完整评价。

**第一步直接补齐这个最重要的缺失对照。**

使用完全相同的：

- peer 组成；
- acquisition 标签；
- 标签预算；
- item 表示；
- ridge 正则与 Partial8 强度；
- query 身份与重复 response 计数；
- 外层 fold 和 item-split seeds。

不得根据第二套 cohort 已知的完全共享分数修改 Partial8。

重点计算：

\[
\Delta_{\mathrm{partial}}
=
ALC_{\mathrm{partial8}}
-
ALC_{\mathrm{fully\ pooled}}
\]

并报告：

- 总体差值；
- 三 fold；
- Benchmark-equal；
- Benchmark-cluster CI；
- 不同预算；
- 逐 Benchmark；
- 与第一套 cohort 的增量方向是否一致。

### A2. 新增少量预冻结 Peer 组成

在首次运行前固定少量新的 cohort seeds。

建议再增加 3 套不同 peer 组成，使主 cohort、已有第二套 cohort 与新增组成形成一个可比较的敏感性实验集合。

选择 cohort 时：

- 只能依靠合法可见的 Subject 属性和预设随机规则；
- 不得依据 response outcome 或历史候选分数选择 peer；
- 不得专门选择能改善 `alignment_faking` 的 peer；
- 相同可见 Subject 配置冲突时执行原保守排除；
- 所有 cohort seeds、排序和选择规则在评分前冻结。

每套 cohort 必须同时运行：

1. R25 own-only；
2. 完全共享 pooled；
3. Partial8。

**不能拿一个 cohort 的 Partial8 与另一个 cohort 的完全共享直接比较机制增量。**

这些组成变化仍发生在已反复使用的公开 Benchmark 上，属于 peer 组成敏感性证据，不是新的独立 Benchmark 盲测。

### A3. Peer 缺失、数量不足与不均衡

针对冻结的主 cohort，复用或重放 R76 的预定义场景：

- 0 peer；
- 1 peer；
- 2 peers；
- 3 peers；
- 部分 peer 缺失；
- 不同 peer 的标签预算不均衡；
- 可见 Subject 配置歧义。

每个场景比较完全共享与 Partial8。

重点分析：

**Partial8 是否只在完整 3-peer 场景下有优势？**

如果缺少 peer 后改善消失或退化，需要确定原因：

- 总观测数减少；
- 有效 peer 数减少；
- 不同 peer 的标签分布变化；
- Subject-specific 参数难以估计；
- 部分共享增加了不必要的估计自由度。

不要仅使用整体 ALC 描述这些现象。

### A4. 判断增量是否稳定

至少报告：

- 每个 cohort 的 Partial8 − pooled 差值；
- 多 cohort 平均增量与分布；
- 有多少 cohort 的增量为负；
- Benchmark × cohort 的收益/退化矩阵；
- `alignment_faking` 与其他 Benchmark 的独立贡献；
- Partial8 对最大严重退化的影响；
- 同一 Benchmark 在不同 cohort 中是否发生收益方向翻转。

Bootstrap 以 Benchmark 为主要 cluster，保留该 Benchmark 内各 pair、item seeds 和 cohort 重复评价的依赖关系。

不得把多套 cohort 或三个 item-split seeds 当成新增的独立 Benchmark 样本。

### A5. R77 决策

如果 Partial8 在多套冻结 cohort 中持续优于完全共享，且没有因部分共享新增明显严重退化，则保留其为具有更强组成稳健性证据的研究候选。

如果增益仅在少数 cohort 出现、方向明显翻转，则更新判断为：

`partial_sharing_composition_sensitive`

并停止继续扫描 Subject 偏移收缩强度。

无论结果如何，不能因为观察到新的最优 cohort，就将其选为正式预测时的固定有利 peer 组合。

---

# 实验 B：什么时候不应该共享？

## 五、R78：Support-Based Reliability Diagnosis

### B1. 核心问题

R76 的 `dave` 等负迁移说明：

**共享斜率可能在部分 Subject 上不适用。**

但当前 Partial8 只是允许偏移：

\[
\beta_s=\beta+\delta_s
\]

并没有可靠判断共享方向是否值得应用。

因此本阶段要回答：

> **能否只使用当前合法支持标签，判断其他 Subject 学到的局部响应关系是否对目标 Subject 有预测价值？**

首先做机制诊断，不直接训练门控网络。

### B2. 核心竞争假设

**H1：Support Predictive Consistency**

如果 peer 学到的共享方向能够预测目标 Subject 已知支持标签，它也更可能改善未见 query。

**H2：Apparent Agreement from Small Samples**

由于已揭示标签太少，表面上的一致性主要来自抽样噪声，无法可靠预测 query 收益。

**H3：Shared Mean vs Shared Slope Confusion**

某些一致性仅反映成功率均值接近，而不是局部响应方向相似。

**H4：Insufficient Identifiability**

部分 Subject 即使具有真实不同的局部响应结构，也无法从最多 31 个自身标签中稳定判断是否适合共享。

要求通过有判别力的实验判断这些解释，而不是预设条件门控必然有效。

## 六、R78：交叉支持集预测诊断

### B3. 构造合法的 Support Validation

对每个 target Subject，仅使用其当前预算允许揭示的支持标签。

优先研究：

\[
n=15,\quad n=31
\]

随后根据证据判断是否扩展到 B7、B3。

使用预先冻结的 support masking 或 leave-out 协议，将当前 Subject 的部分支持标签暂时隐藏。

以剩余支持标签和当前合法 peer 标签拟合：

1. Own-only R25；
2. 完全共享 pooled；
3. Partial8。

然后预测被隐藏的 own support 标签，计算相应 Brier。

**被隐藏的支持标签只能用于该诊断的评分，不得参与本次拟合、均值估计、标准化或正则选择。**

不同模型必须使用完全相同的支持划分。

Peer 自己的合法 acquisition 标签仍可按已冻结情景使用，但不得加入任何 query outcome 或未来预算标签。

建议先在现有来源数据上开展少量固定 support masks，控制计算成本；不要一开始对所有 pair 执行高成本的完整 LOO。

### B4. 构建共享可靠性指标

首先考虑直接的支持集预测风险差：

\[
z=
\operatorname{Brier}_{support}
(\mathrm{shared})
-
\operatorname{Brier}_{support}
(\mathrm{own})
\]

其中 shared 可分别使用完全共享与 Partial8。

如果：

\[
z<0
\]

意味着共享方法对这部分已知支持标签预测更好。

但这**不等于共享也能改善未知 query**。

因此必须检查：

\[
z
\]

与真实 query-level：

\[
\Delta_q=
\operatorname{Brier}_{query}
(\mathrm{shared})
-
\operatorname{Brier}_{query}
(\mathrm{own})
\]

之间是否存在跨 Benchmark 稳定的关系。

Query outcome 仅能用于离线验证和来源内监督，不得进入运行时可靠性特征。

除风险差外，可以考虑少量与机制直接相关的候选统计：

- 目标 Subject 的支持残差与共享方向是否一致；
- Peer 之间的共享方向是否一致；
- 当前支持样本对斜率差异是否具有足够辨识信息。

最多保留少量事前冻结的候选变量，不进行高维特征搜索。

### B5. 区分均值一致与斜率一致

这是本轮必须执行的机制对照。

共享收益可能来自：

- 当前 Subject 与 peer 的整体成功率相似；
- 不同 Subject 对 item 特征变化的响应方向相似；
- 两者都有作用。

因此至少比较：

**Mean-only consistency：** 只考虑不同 Subject 的平均成功率或支持残差均值。

**Slope-specific consistency：** 控制各 Subject own mean 后，检验共享方向对目标 Subject 支持标签变化的预测能力。

**Combined consistency：** 同时考虑均值和方向，但不让更高复杂度自动成为优势。

需要判断：

> 真正能预测 query-level 共享收益的是 Subject 均值接近，还是中心化后的局部响应方向一致？

这直接关系到我们是否真正识别了 R72/R76 的有效共享条件。

### B6. 诊断指标与判别要求

对每个预设预算与 cohort，报告：

- Support-level 相对 Brier；
- Query-level 相对 Brier；
- 共享有利/有害的预测方向识别能力；
- Benchmark-grouped Spearman 或其他适当的关联指标；
- 按预定义可靠性分组的实际收益；
- 有效支持标签数量；
- 指标稳定性与不确定性；
- 大小不同的 Benchmark 是否结论一致。

重点检查：

- `alignment_faking` 与其余 Benchmark；
- 低 own 支持标签预算；
- peer 缺失及覆盖不均；
- peer 标签置换；
- `dave` 等历史错误共享场景。

`dave` 只能用于诊断和反例解释，**不得用 Benchmark 名称、已见 query 成功率或外层误差拟合专门的回退规则。**

### B7. 证据进入条件

只有在 Benchmark 分组的来源验证中，至少一种支持集可靠性指标能够稳定预测 query-level 的相对风险，才进入下一阶段。

如果支持集一致性与真正的 query 共享收益没有稳定关系，则记录：

`support_consistency_not_predictive`

停止门控开发。

不能因为某个事后切片有漂亮的收益，就直接把它固化成规则。

---

# 条件实验：共享可靠性控制

## 七、R79：仅在 R78 成立时执行

R79 是条件阶段，不允许预设必须训练新门控。

### 7.1 最小方法设计

如果 R78 验证了合法可观测的共享可靠性指标，则根据其结构设计最小修正，例如：

\[
p_{\mathrm{final}}
=
p_{\mathrm{own}}
+
g(z,n)
\left(
p_{\mathrm{partial}}-p_{\mathrm{own}}
\right)
\]

其中：

\[
0\leq g(z,n)\leq1
\]

\(z\) 为 R78 验证有效的支持集可靠性指标。

优先考虑：

- 保守的二值共享/回退；
- 少量可靠性分组；
- 统计证据不足时回退 own-only；
- 在支持证据较弱时减少共享强度。

不优先采用新的 MLP gate。

特别注意：R43、R61 和 R71 已经出现过可靠性估计失效，因此必须说明当前支持集共享可靠性指标新增了什么此前不存在的信息。

### 7.2 Gate 必须严格来源外拟合

所有可靠性规则、阈值和校准参数只能通过训练来源 Benchmark-grouped OOF 数据估计。

目标 Benchmark 的 query labels 不能参与：

- Gate 拟合；
- 阈值选择；
- 参数选择；
- 是否开启共享的决定。

不得根据 R76 已知的 `dave` 回归手动设计特殊门控。

### 7.3 必要对照

至少比较：

1. R25；
2. 完全共享 pooled；
3. Partial8；
4. 新可靠性控制方法；
5. 相同平均共享强度或平方位移成本的简单收缩对照。

最后一个对照十分重要。

如果新门控只是总体缩小修正幅度，它不一定真的识别了错误共享。

需要证明它相对于简单收缩有额外的、与预测残差对齐有关的价值。

### 7.4 完整探索评价

来源实验具有足够正面证据时，冻结一个最简单的新候选。

执行：

5,242 pairs × 3 item-split seeds × 六预算。

重点比较：

\[
ALC_{\mathrm{new}}-ALC_{\mathrm{partial8}}
\]

以及：

\[
ALC_{\mathrm{new}}-ALC_{\mathrm{pooled}}
\]

同时关注：

- Pair-equal 主指标；
- Benchmark-equal；
- 三个外层 fold；
- Bootstrap；
- `alignment_faking` 之外的收益；
- 最大负迁移；
- 共享被关闭的比例；
- 错误关闭导致的收益损失；
- 有效 peer 数量和总标签预算。

**如果总体 ALC 略有改善，但严重负迁移没有减少，不能声称已经解决共享可靠性问题。**

如果严重负迁移减少而总体分数略微退化，应独立记录其鲁棒性价值，不能以辅助风险指标替代竞赛主指标。

---

## 八、统一执行协议

### 8.1 信息合法性

所有实验必须严格区分：

- Own acquisition labels；
- Peer acquisition labels；
- Target evaluation labels；
- 内部 identity 和审计数据。

预测只能使用前两者。

Benchmark ID 只允许作为当前任务的相等分组 token，不能作为历史预测特征。

不能使用完整公开数据库中实际上不在本次条件输入里的 peer 标签。

不得依赖跨调用私有状态。

### 8.2 公平性与统计边界

实验 A 的核心公平条件是：

**相同 peer 组成下比较完全共享与 Partial8。**

实验 B 的核心公平条件是：

**相同支持集划分下比较不同共享估计器的预测可靠性。**

不要把额外 peer 标签带来的收益误认为在相同总标签预算下改进了模型。

三次 item-split seeds 不等于三次独立训练初始化。

多个 cohort 也不等于新增了同等数量的独立 Benchmark。

Bootstrap 必须保留上述依赖结构。

### 8.3 探索 Gate 与研究 Gate

继续采用：

**探索宽进，晋级严出。**

对于低成本且有明确机制依据的冻结实验，允许进入一次完整探索性评价。

但 Research Success Gate 保持：

- 相对两种强均值基线改善至少 0.005 ALC；
- 三fold改善；
- Benchmark-equal改善；
- Benchmark-cluster CI 上界低于零；
- 没有无法解释的严重回归。

此外，Shared-Peer 必须单独满足实际 Hosted 信息可用性要求。

当前 Codabench 服务问题由外部事项跟踪，本轮不处理 Hosted 下载、正式提交或私有 coordinator 逆向。

所有 Shared-Peer 结果继续标记为：

`conditional_simulation`

不得宣称实际线上成绩。

---

## 九、执行顺序与停止规则

本轮采取以下顺序：

**R77：先执行实验 A。**

补齐第二套已冻结 cohort 的 Partial8，然后在少量新预冻结 cohort 上完成 Partial8 与完全共享的配对验证，并检查缺失、不均衡和 peer 数量变化。

**R78：执行实验 B。**

先用合法支持标签做交叉预测诊断，判断 support-level 共享一致性是否能够预测 query-level 收益。

优先 B15/B31；只有具备可信证据时再考虑低预算。

**R79：条件执行。**

只有 R78 发现可迁移可靠性信号，才构造一个简单、保守的共享控制候选并进行冻结验证。

不要因为想完成所有轮次就强行运行 R79。

### 停止规则

- R77 若显示 Partial8 的优势高度依赖某一 cohort，则停止当前结构的进一步参数优化，保留条件结果。
- R78 若无法找到稳定的支持集可靠性指标，则停止门控设计，不训练更复杂模型。
- R79 若没有优于简单收缩及 Partial8 的独立收益，则停止该门控配置，不继续网格搜索。

如出现新的机制性异常，应先上钻诊断，不直接添加自由度。

---

## 十、交付要求

沿用现有项目的研究目录结构建立 R77–R79 独立产物。

建议交付：

- `reports/shared_cohort_robustness_findings.md`
- `reports/support_sharing_reliability_findings.md`
- `reports/shared_reliability_tables.md`
- `research_loop/SHARED_RELIABILITY_SPEC.md`
- `research_loop/SHARED_RELIABILITY_DECISION.md`
- `runs/research_loop/R77/`
- `runs/research_loop/R78/`
- `runs/research_loop/R79/`（仅实际触发时）
- 更新 `research_loop/HIGH_LEVEL_STATE.md`
- 更新历史正负证据 ledger

每轮保存：

- 冻结合同；
- 数据与源码 hash；
- cohort identities 与可见属性；
- acquisition/query 隔离；
- 随机 seeds；
- 原始概率；
- response counts；
- 逐 pair / Benchmark / budget 指标；
- paired bootstrap；
- 失败实验与异常留痕；
- CPU 时间与峰值内存；
- 独立数值重计分。

不覆盖 R72/R76 原始产物及已有 fallback。

### 最终报告必须回答

**Q1：Partial8 在不同 peer 组成下是否仍优于完全共享？**

明确区分同一 cohort 内结构增量与跨 cohort 敏感性。

**Q2：Subject 间共享斜率的哪些差异可以通过当前支持标签观察到？**

区分真实响应异质性与有限样本噪声。

**Q3：支持集上的共享预测一致性，能否预测未见 query 的共享收益？**

这是 R78 最关键的判别目标。

**Q4：如果能够识别可靠性，保守门控是否真的减少了错误共享，而不只是整体收缩预测？**

必须使用匹配的简单收缩对照。

**Q5：下一轮应该继续部分共享、研究可观测可靠性，还是停止当前配置？**

给出证据、反例、适用边界和具体决策，不仅返回排行榜。

---

## 十一、现在开始执行

首先读取 R75/R76 冻结协议和实际代码，确认 Partial8、主 cohort 与第二套 cohort 的实现及数值身份。

然后：

1. 运行 R77，补齐不同 cohort 下的匹配结构评价。
2. 运行 R78，建立严格支持标签隔离的共享可靠性诊断。
3. 只有可靠性机制证据达到预设条件，才进入 R79 的最小方法验证。

**本轮最重要的 High-Level Research Question：**

\[
\boxed{
\text{Shared Structure Exists}
\;\not\Rightarrow\;
\text{Shared Structure Is Appropriate for Every Subject}
}
\]

我们已经有证据支持第一部分。

**现在要判断第二部分能否由当前合法支持信息识别，并转化为真正可靠的 Brier 改善。**

不要为了一点平均 ALC 收益重复搜索收缩强度。应优先建立可解释、可证伪、具有跨 Benchmark 和跨 peer 组成证据的共享机制。

完成后给出实际实验结果、机制理解、风险边界和下一步科研决策，而不是仅提供代码实现或执行计划。