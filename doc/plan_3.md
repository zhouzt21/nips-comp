# Research Prompt：PAIEC R59+——低预算 MLP 先验与 R25 的互补性诊断、条件组合及下一轮优化

## 一、任务目标

你是一名 **ML Researcher、Scientific Failure Diagnostician、Probabilistic Prediction Expert 与 Experimental Systems Engineer**。

本轮目标是：

**基于 R1–R58 已有实验，判断 R54/R56 的低预算 MLP 优势是否包含 R25 尚未利用的可迁移预测信息，并在证据支持时构造和验证能够稳定改善 R25 Brier ALC 的组合方法。**

继续采用：

**实验观察 → 上钻理解机制 → 选择研究操作 → 下钻设计方法 → 判别实验 → 更新认识。**

不要直接把 MLP 与 R25 相加，也不要开展无依据的超参数搜索。

这是竞赛优化任务，优先考虑可复现的实际得分改善，不要求算法具有新颖性。

## 二、恢复已有研究状态

项目：

`D:\nips-comp\nips-comp\paiec_baseline`

优先读取：

- `reports/training_mechanism_v5_findings.md`，特别是第 2、3、5、7、9 节；
- `reports/training_mechanism_v5_metrics.csv`；
- `reports/training_mechanism_v5_benchmark_metrics.csv`；
- `research_loop/TRAINING_MECHANISM_V5_SPEC.md`；
- `reports/reproduction_v4_findings.md`；
- `reports/research_loop_v3_findings.md`；
- `research_loop/HIGH_LEVEL_STATE.md`；
- `runs/research_loop/training_mechanism_v5/R54–R58/` 中实际存在的预测、权重、选择结果及日志。

同时检查 R25、R54、R56 的实际预测实现与数据调用关系。若逐 pair 预测记录不齐全，应优先从冻结权重和已有配置回放，而不是重新训练。

### 当前关键事实

5242-pair、三种子历史评价：

| 方法 | ALC |
|---|---:|
| Smoothed mean strength 2 | 0.198687363 |
| R25 | **0.195123926** |

冻结成功目标：ALC 至少降低 0.005，相应目标不高于 0.193687363。

R54/R56 来源内实验显示：

1. R54 MLP 在来源池 1、2 的 B0/B1 优于 R25，在来源池 0 则没有优势。
2. R56 Benchmark 混合风险相对匹配的 R54 MLP，在三个来源池的 B0/B1 均有改善，但相对 R25 仍有来源异质性。
3. R25 在 B7/B15/B31 等较高预算下整体更强。
4. R54 训练后期明显发生来源外退化，没有发现符合预声明条件的延迟泛化恢复。
5. R55 的支持集重采样仅有微小局部收益，尚未证明新增了可迁移的 item–label 对应信息。
6. R57 的模型平均未带来稳定收益，R58 初始化干预未触发。
7. R51/R52 的 CNP 和 Amortized IRT 当前实现均未通过来源门槛，不应直接添加到 R25。

此前关于 R20 跨 Subject 信息共享、R45/R46 词法–语义互补和 R34 幅度校准的认识继续保留，但它们不是本轮首要实验对象。

## 三、R59：上钻——低预算优势到底来自什么？

本阶段**先分析现有预测，不训练新模型**。

### 核心竞争假设

**H1：可迁移的 item-specific information**

MLP 在部分来源上的低预算优势来自对 item 成功概率的有效区分。

**H2：Pair-level prevalence / subject prior**

优势主要来自更准确的 pair 平均概率，与 item-specific prediction 关系较弱。

**H3：来源分布偶然匹配**

MLP 在某些 Benchmark 上恰好具有较好的先验，但不存在可稳定迁移到未知 Benchmark 的规律。

**H4：训练风险改变了偏差–方差结构**

R56 的收益可能来自减小错误先验的方差或系统偏差，而非学到新的信息。

### 必须执行的分析

针对 R54、R56 和 R25，在每个来源池、Benchmark、pair 和预算上开展配对分析。

首先精确分解 B0/B1 的 Brier 差异：

\[
B=\operatorname{Var}(Y)
+(\bar p-\bar y)^2
+\operatorname{Var}(p)
-2\operatorname{Cov}(p,Y)
\]

区分：

- pair 均值预测是否更准确；
- item-level variation 是否贡献正向收益；
- 预测概率的方差是否合理；
- R56 相比 R54 改善的是哪一项；
- 哪些 Benchmark 出现负迁移；
- 优势是否主要由某个大型 Benchmark 贡献。

随后进行**组合可行性审计**：

冻结已有预测，只在预算 0/1 使用 R54 或 R56、预算 3–31 保留 R25，计算逐预算加权结果。

该操作仅用于确定组合潜力，属于对已查看数据的诊断性重组，不能视为新的独立验证成绩。

同时检查该组合是否稳定胜过 R25 与强简单 baseline，而不是只报告改善的来源池。

### R59 必须形成的判断

明确回答：

1. MLP 的低预算优势主要来自 pair mean 还是 item discrimination？
2. R56 是否提供了超出 R54 的独立、可解释的机制改善？
3. 优势是否具备跨 Benchmark 的一致性？
4. 是否存在能够在预测时提前识别 MLP 可靠性的可观测信号？
5. 如果只能事后根据评价标签识别优势，是否应停止组合路线？

形成 **High-Level Mechanism Update**，不能仅输出分数表。

## 四、R60：选择研究操作并构造最小组合

只有 R59 发现足够的组合潜力时，才进入本阶段。

根据诊断，自适应选择研究操作，而不是预先固定方法。

例如：

- 如果低预算与高预算的信息机制不同：`Diagnostic → Decompose → Differentiated Treatment`。
- 如果 MLP 只在可识别的条件下可靠：`Audit Reliability → Conditional Selection`。
- 如果两个预测器的错误具有真正的互补性：`Error Decomposition → Controlled Combination`。
- 如果主要优势无法由合法输入识别：停止组合，重新上钻到 information / identifiability 问题。

### 候选设计原则

优先尝试**不需要重新训练大模型**的最小方法。

可以考虑：

- 冻结的预算分段预测；
- 只由训练来源学习的可靠性门控；
- 在明确条件下使用 MLP prior，否则回退 R25；
- 基于合法可见输入的低维先验可信度估计。

但候选设计必须由 R59 的机制证据决定。

不能依据目标 Benchmark 的真实成功率、评价标签或事后表现选择预测器。

尤其注意：R43 的 query-specific 可靠性 MLP 已经失败。若提出新的可靠性模型，必须明确它与 R43 的区别，以及为什么当前低预算先验问题具有新的可辨识信息。

禁止重新扫描固定 mix、平滑系数、MLP 网络宽度或任意门控阈值。

## 五、R61：下钻——实际验证组合是否有效

在运行前冻结新候选、超参数及停止条件。

### 第一关：来源内机制 Pilot

使用已有三个来源池，保持：

- Benchmark 互斥训练/验证；
- 相同 evaluation responses；
- 相同 item splits 和标签预算；
- 相同 pair 等权 Brier ALC；
- 完整的 R25、R54/R56 与强简单 baseline 对照。

如需学习组合门控，必须在训练 Benchmark 内通过独立分组拟合或 cross-fitting 选择，不能用正在评分的 Benchmark 结果训练门控。

重点比较：

1. R25；
2. R54/R56 低预算预测器；
3. 仅按预算分段的简单组合；
4. 有机制依据的条件组合（如 R59 支持）；
5. 同协议的强简单 baseline。

只允许少量、事前冻结的候选。

报告 B0/B1/B3/B7/B15/B31、ALC、benchmark-equal ALC、各 Benchmark 增减，以及配对误差统计。

**Pilot 进入外层的条件：** 候选相对 R25 在三个来源池的 pair-equal ALC 均改善，benchmark-equal 不退化，关键改善与 R59 所宣称的机制一致，并且没有未解释的严重 Benchmark 回归。

如果条件不满足，不得在同一来源池反复调整门控直到通过。

### 第二关：完整公开数据评价

只有 Pilot 通过，才冻结候选并进入原有 5242-pair、三个种子的外层评价。

同时检查：

- 是否满足相对 frozen smoothed mean 的绝对 0.005 改善门槛；
- 三折方向是否一致；
- benchmark-equal 是否改善；
- benchmark-cluster paired bootstrap 95% 差值区间上界是否低于零；
- 是否存在严重负迁移。

由于公开 Benchmark 已被反复用于模型研究，该结果仍属于公开数据上的开发验证，不能宣称独立盲测。

不得因点估计较好就自动替换已有提交 fallback。

## 六、如果组合失败，如何进入下一轮？

不能只总结“MLP + R25 无效”。

必须根据失败机制更新研究路线。

- **若 MLP 优势主要来自偶然 pair 均值匹配：** 降低继续研究该 prior 的优先级，转向合法冷启动信息的可迁移性。
- **若存在 item-level signal，但无法判断何时可靠：** 研究可观测的可靠性信息与统计可辨识条件。
- **若预算分段有效、但条件融合无增量：** 保留简单方法，不增加不必要的门控模型。
- **若当前局部预测仍是最佳：** 保留 R25，重新评估 R20 的跨 Subject 新证据或其他尚未解决的信息来源。
- **若已有预测的任何组合均缺乏足够潜力：** 不重复修补固定预测，改为寻找新的可用信息或具有不同归纳偏置的模型。

必要时可使用 AnySearch 验证新的高层机制，但本轮不需要重新开展宽泛的文献综述。

## 七、执行与交付要求

本轮是**实际执行任务**，不是建议清单。

有本地访问权限时，应直接读取代码和原始实验数据，完成 R59 诊断；只有条件满足才实施 R60/R61。

使用现有纯 CPU 环境，优先复用冻结预测、缓存和权重，不重复下载数据或重训模型。

输出并保存：

- `reports/low_budget_prior_composition_findings.md`：结果与机制认识；
- `research_loop/`：新实验协议、假设及 operator 决策；
- `runs/research_loop/R59–R61/`：真实数据、预测、日志、配对结果和代码版本；
- `research_loop/HIGH_LEVEL_STATE.md`：更新后的高层问题认识与下一步行动。

若项目已有等价的记录结构，应复用，而不是重复创建。

每轮报告只需六项：

**Current Bottleneck → Upward Diagnosis → Operator Choice → Experiment → Evidence/Gate → Next Move。**

不得虚构不存在的实验或结果。未触发的步骤明确记为 `condition_not_met`。

保留 R25 研究参照和 `smoothed_mean_strength2.zip` 提交 fallback，不自动正式提交。

---

## 八、现在开始

**从 R59 开始，优先复用 R54/R56 与 R25 的真实逐 pair 预测，分析低预算互补性。**

首先回答：

> **R54/R56 在部分来源 Benchmark 的 B0/B1 优势，是不是一种能够在测试时被可靠识别并利用的预测信息？**

如果答案有足够证据支持，就设计最简单的条件组合并执行验证。

如果没有，就明确排除当前组合路线，保留有效的 high-level 认识，转入下一项更有信息价值的研究问题。

**目标不是证明 MLP 一定有用，而是通过严格的机制诊断与配对实验，找到可以稳定提高 R25 Brier ALC 的真实增量；不要让局部点估计优势取代可迁移性证据。**