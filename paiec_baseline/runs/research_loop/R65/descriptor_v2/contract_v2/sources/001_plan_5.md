# PAIEC Research Prompt：R65–R68——R56 低预算优势的可迁移机制诊断、新先验实验与信息增量验证

## 一、研究角色与核心目标

你是一名 **Probabilistic Prediction Researcher、OOD Generalization Diagnostician、Scientific Experiment Designer 与 ML Systems Engineer**。

当前 PAIEC 已完成 R1–R64。下一轮的目标不是继续搜索 MLP 超参数，也不是简单修改 R56 的融合比例，而是：

**明确 R56 在低预算下学到了什么，为什么其优势集中于特定 Benchmark，以及能否从中提取可迁移到未知 Benchmark 的预测机制，最终进一步降低 R25 的 Brier ALC。**

采用以下研究循环：

**Observed Signal → Competing Mechanisms → Discriminating Experiment → Mechanism Judgment → Minimal Intervention → OOD Validation → Research Decision。**

本轮需要真实执行实验，不只是生成计划。

如果现有证据无法支持继续开发 R56，应及时停止，并转向真正新增的信息来源。

---

## 二、读取材料并恢复研究状态

工作目录：

`D:\nips-comp\nips-comp\paiec_baseline`

首先读取：

1. `reports/historical_positive_followup_findings.md`
2. `reports/historical_positive_tables.md`
3. `reports/historical_positive_reproduction.md`
4. `reports/low_budget_prior_composition_findings.md`
5. `reports/training_mechanism_v5_findings.md`
6. `reports/reproduction_v4_findings.md`
7. `reports/research_loop_v3_findings.md`
8. `research_loop/HIGH_LEVEL_STATE.md`
9. R25、R54、R56、R59–R64 的实际代码、模型权重、逐 pair 预测、评价和实验合同
10. 相关历史信号登记与原始迭代记录

重点恢复 R62 的原始逐 pair 概率、支持/查询身份、response 计数、训练来源及特征编码，而不只是汇总指标。

### 冻结研究基准

原 5,242 pairs × 3 item-split seeds：

| 方法 | Brier ALC ↓ |
|---|---:|
| Smoothed mean strength 2 | 0.198687363 |
| R25 | 0.195123926 |
| R56_segment | **0.193805399** |

R56_segment 相对 R25 改善 0.001318527，主要来自：

- B0：Brier 从 0.250000 降至 0.243287；
- B1：Brier 从 0.222119 降至 0.218883；
- B3–B31：保持 R25。

但其证据存在四个重要限制：

1. `alignment_faking` 对整体 ΔALC 的贡献为 −0.001326136，其他 Benchmark 合计 +0.000007609。
2. Benchmark-equal ALC 从 R25 的 0.212920 升至 0.215241。
3. 三折变化为 −0.003783、+0.002048、−0.001388。
4. Benchmark-cluster 95% CI 跨零。

同时，R56 约 97.7% 的净增益来自 **pair 平均预测概率的改善**，而非额外的 item-level discrimination。

R63 已基本排除继续投入当前 R43/R55 条件修正配置的价值；R64 保留了词法模板修复的机制认识，但固定组合没有超过 R25。

**本轮不得将 R56 的公开总体改善误认为已经证明其具备一般跨 Benchmark 泛化能力。**

---

## 三、R65：R56 Pair-Mean 优势的机制解构

本阶段优先复用冻结模型和预测，不训练新模型。

### 3.1 核心科学问题

R56 的低预算优势究竟来自：

**H1：Subject-dependent prior**

模型依据 subject 的合法描述性特征，形成了比 0.5 更准确的能力或成功概率先验。

**H2：Item-distribution-dependent prior**

题目内容和 metadata 虽然未提供稳定的逐题 discrimination，但其整体分布可能间接携带任务成功率信息。

**H3：Subject–Item interaction**

单独的 subject 或 item 信息都不足以解释收益，只有两者的交互才能预测 pair-level prevalence。

**H4：Benchmark-specific prevalence matching**

模型学到的主要是训练来源的成功率模式，在与训练来源相似的 Benchmark 上获益，在其他 Benchmark 上负迁移。

**H5：Calibration rather than new information**

R56 相对 R54 的改善主要来自训练风险调整对概率均值和校准的影响，而不是新增了更丰富的可迁移表示。

这些假设可能同时成立，不要求机械地选择唯一原因。

### 3.2 必须执行的诊断

对 R25、R54、R56 分别分析 B0 和 B1。

**A. 预测误差结构**

使用与正式评价一致的 response 权重和 pair 等权方式，精确分解：

\[
\operatorname{Brier}
=
\operatorname{Var}(Y)
+
(\bar p-\bar y)^2
+
\operatorname{Var}(p)
-
2\operatorname{Cov}(p,Y)
\]

重点判断：

- R56 的 pair 均值优势在哪些 Benchmark 出现；
- 改善来自降低平均概率偏差，还是改变概率离散程度；
- B1 中多少收益来自原始先验，多少来自首个支持标签后的修正；
- 在 `alignment_faking` 和其他 Benchmark 上，这些机制是否相同；
- 哪些输入特征与预测误差具有稳定联系。

评价标签只能用于离线诊断，不能进入预测特征或门控。

**B. 训练来源与目标来源的关系**

审计：

- R56 的训练 Benchmark 与目标 Benchmark 在合法输入特征上的分布差异；
- 预测成功率与真实成功率的关系；
- 是否存在明显的先验高估或低估；
- R56 在不同 Benchmark 上的输出分布是否异常接近训练来源的标签 prevalence；
- `alignment_faking` 的收益是否可以用其他 Benchmark 上可见的输入结构解释。

禁止直接把 Benchmark ID 或真实 Benchmark 成功率当作可部署的可靠性信号。

**C. 预测器的可观测可靠性**

检查是否存在在推理时合法可得、并且可能跨 Benchmark 稳定的变量，可以解释 R56 与 R25 的相对风险，例如：

- subject 描述的结构性特征；
- item 的可计算内容或类型特征；
- 冻结语义表示的低维统计；
- 原模型预测概率及其不确定性特征；
- 合法支持标签提供的证据。

不得假设预测时能够一次访问完整 query 集合。任何依赖整组 item 汇总的信号，必须先确认正式接口支持；否则只作为离线诊断。

### 3.3 R65 交付

形成一份层级化的机制认识：

**从高到低：**

1. R56 是否提供可迁移信息；
2. 若有，信息来自什么变量关系；
3. 哪些 Benchmark 满足或违反该关系；
4. 这些关系能否用预测时的合法信息识别；
5. 现有证据还不能排除哪些竞争解释。

输出支持证据、反例、适用边界及下一步最有判别力的实验。

不能仅列出相关系数或特征重要性，就宣称完成机制识别。

---

## 四、R66：以受控消融验证先验信息来源

根据 R65 结果，选择最有判别力的少量实验。

### 4.1 优先考虑的最小对照

在当前 R56 模型及训练过程基础上，考虑：

| 候选 | 保留信息 | 要回答的问题 |
|---|---|---|
| Subject-only prior | 仅合法 subject 特征 | Subject 先验是否足以解释收益？ |
| Item-only prior | 仅合法 item 特征 | 是否主要学到了内容相关的平均难度？ |
| Subject + Item | R56 原始特征 | 交互信息是否真正必要？ |
| 简单训练期先验对照 | 仅训练 Benchmark 的合法统计信息 | 神经网络是否确实超过了低复杂度先验？ |

这些是**候选判别实验**，不是必须无条件训练四种模型。

先检查 R1–R64 是否已有严格等价的实验；已有结果能够回答的问题直接引用，不重复训练。

如果某个输入消融不具备可辨识性或会产生严重分布外特征，应采用从训练开始去除对应特征的匹配模型，而不是简单将训练好的模型输入置零。

保持相同：

- 训练数据与合法来源；
- 训练目标；
- 模型容量或可比较容量；
- 初始化/训练预算；
- checkpoint 选择协议；
- 外层 Benchmark 隔离；
- pair-equal Brier 评价。

资源条件为纯 CPU。只实施能够判别机制的小规模对照，不扩大网络或扫描学习率。

### 4.2 额外要求：识别是否真的具有 OOD 收益

必须区分：

\[
\text{In-source predictive information}
\]

与：

\[
\text{Transferable predictive information}
\]

要求在 Benchmark 隔离的来源训练/验证中判断：

- 哪些特征组稳定改善 B0/B1；
- 哪些特征只提高训练来源拟合；
- 新先验是否仍高度依赖 `alignment_faking`；
- 是否存在被简单常数或低维统计先验击败的情况。

如果所有消融都表明主要收益来自无法迁移的来源 prevalence，停止 R56 结构改造。

---

## 五、R67：只有发现可迁移机制，才设计新预测器

R67 是条件执行阶段。

不要预设必须产生新模型。

### 5.1 进入条件

R65/R66 至少应提供：

1. 具有明确含义的合法可观测信号；
2. 在来源内 Benchmark 隔离下的正面预测证据；
3. 对主要负迁移 Benchmark 有合理解释；
4. 能与 R25 形成信息互补或可靠性改善；
5. 不需要事后使用目标 Benchmark 标签选择方法。

如果没有达到上述条件，应记录 `condition_not_met`，转向 R68。

### 5.2 设计原则

根据已经确定的机制，选择最小干预。

可能包括：

- 更简洁的 subject-conditioned prior；
- 具有明确可辨识条件的 item-demand prior；
- 将 subject 与 item 的有效结构交互分离建模；
- 具有来源内可靠性证据的保守先验；
- 其他由 R65/R66 证据直接支持的方法。

不要因为 R56 分数接近 0.005 门槛，就继续扫描固定 shrinkage、mix、门控阈值或预算切换点。

特别注意：

**R61 已经证明，仅用第一个支持标签判断是否选择 R56，并不足够可靠。** 新方法如果重新提出可靠性估计，必须明确增加了什么此前缺失的可观测信息。

### 5.3 验证程序

先在训练来源内部固定候选、做 Benchmark-grouped 验证。

如需训练可靠性模块，使用严格的 out-of-fold 预测作为训练输入，不能使用同一模型在拟合样本上的乐观误差标签。

只允许事前冻结的少量候选进入外层探索，不要求探索 Gate 等同研究成功 Gate。

对于通过探索条件的候选，执行原协议：

**5,242 pairs × 3 item-split seeds × 六预算。**

报告：

- pair-equal ALC；
- benchmark-equal ALC；
- 每折和每预算变化；
- 配对 Benchmark-cluster Bootstrap；
- `alignment_faking` 与其余 Benchmark 的单独贡献；
- 最大回归；
- 相对 R25、R56_segment 和两种强简单 baseline 的收益；
- CPU 训练与实际推理成本。

R62 外层结果已经被查看并参与本轮研究决策，因此不能把 R67 在相同外层上的提高称为全新独立验证。

如果可以获得未参与任何候选设计的新 Benchmark，应优先保留为真正独立的最终检验；如果没有，则诚实标记为反复使用公开数据的探索结果。

---

## 六、R68：若原有信息不足，转向真正新增的合法信息

R62–R64 表明，继续组合冻结的旧预测器已经很难稳定获得额外收益。

因此，若 R65–R67 无法提取可靠的可迁移先验，应优先排查以下两个信息方向。

### A. ADeLe 风格的任务需求表示

现有 R53 已完成部分准备，但因资源问题尚未正式标注、训练或验证。

首先确认：

- 原 128 个 distinct items × 4 个需求维度的准备状态；
- 使用的 rubric 与原始论文是否一致；
- 标注资源、成本和权限是否具备；
- 隐藏测试推理时能否合法、稳定地获得相同特征；
- 是否存在不依赖额外付费 API 的可重复 CPU 替代。

如果标注资源尚未授权，不执行收费 API 调用，也不伪造需求标签。

如果资源已经具备，先开展最小规模的特征一致性和独立信息验证：

**Task-demand representation 是否提供 MiniLM、metadata 和当前 R25 无法捕获的可迁移预测信息？**

不直接完整复现 ADeLe，也不立即训练大型预测器。

### B. 跨 Subject 当前标签共享

复核 R20/R36 对共享 item 信息的正面证据与强基线失败原因。

优先从官方接口文档、可访问的 starter package 和实际 `predict(input,labeled)` 字段确认：

- 当前调用能看到哪些 subject 的标签；
- 标签是否可匹配到相同 item；
- 是否能跨当前 pair 保留共享状态；
- 是否允许跨 subject 使用当前 Benchmark 的观察；
- 是否存在评测顺序或沙盒隔离限制。

没有确切接口依据时，不构建依赖共享隐藏标签或跨调用状态的模型。

如果能够确认共享数据合法可见，再考虑新的、具有强 own-only 对照的最小实验；否则将该路线记录为 `blocked_by_runtime_contract`。

---

## 七、实验 Gate 与研究纪律

本轮分开使用三种判断。

**Exploration Gate：** 判断一个低成本、机制明确的候选是否值得扩大评价。允许部分来源退化，但必须有合理的信息价值和严格隔离。

**Research Success Gate：** 沿用原标准。相对冻结 smoothed mean 和强训练来源均值改善至少 0.005 ALC，三折改善、Benchmark 等权改善、聚类置信区间上界小于零，且无未解释的严重退化。

**Submission Decision：** 独立考虑实际竞赛分数、泛化风险、推理资源、部署稳定性及提交截止时间。内部 0.005 不等于官方最低改善要求。

额外要求：

- 不能根据当前外层结果反向学习专门针对 `alignment_faking` 的选择器，再宣称其是无偏的 OOD 收益。
- 不得使用评价标签、oracle 或未来支持标签参与预测。
- 不重复此前已经充分验证失败的 R43/R55 幅度、R46 等权融合、R61 单标签门控等原配置。
- 不将三次 item-split seed 当成三次模型初始化。
- 不把个别 Benchmark 的事后删除作为提高主指标的手段。
- 未执行实验必须记录原因；没有触发条件不能算作负结果。
- 任何新训练都必须保留训练来源、模型选择、代码版本、权重、输入数据及随机种子。
- 优先复用冻结资产与缓存，不进行无依据的大规模 CPU 训练。

---

## 八、交付文件

按现有研究工程规范建立 R65–R68 的独立记录。

建议输出：

- `reports/transferable_prior_mechanism_findings.md`：主要结果与认识，重点呈现 High-Level 机制层级。
- `reports/transferable_prior_metrics.csv`：真实执行的指标及配对差异。
- `research_loop/TRANSFERABLE_PRIOR_SPEC.md`：评分前合同、假设、候选、进入/停止条件。
- `research_loop/TRANSFERABLE_PRIOR_DECISION.md`：机制判断、选用的研究算子、实验取舍。
- `runs/research_loop/R65–R68/`：真实执行产物、训练或回放日志、概率、消融及审计。
- 更新 `research_loop/HIGH_LEVEL_STATE.md` 与研究信号登记。

实际路径应与项目已有目录和脚本结构保持一致；不凭空假设新命令已存在。

主报告按照以下层次组织：

1. **Executive Judgment**：目前最值得继续研究的机制是什么，证据强度如何。
2. **Observed Phenomenon**：R56 的优势及失败在什么条件下出现。
3. **Competing Explanations**：不同解释之间有哪些分歧。
4. **Discriminating Evidence**：哪些实验支持或排除了什么。
5. **High-Level Understanding**：从局部实验上升到对先验、表示、迁移与条件化的认识。
6. **Minimal Method Design**：只有证据支持时才提出新方法。
7. **Evaluation and Decision**：实际分数、风险、停止或晋级。
8. **Next Research Move**：下一轮最有信息价值的行动。

---

## 九、现在开始执行

**首先执行 R65，直接读取 R62 的冻结预测与真实评价数据。**

不要先训练新模型。

本轮第一个必须回答的问题是：

> R56 的 B0/B1 优势究竟是依赖特定 Benchmark 的成功率匹配，还是来自能够由合法 subject/item 特征表达、并迁移到未知 Benchmark 的稳定预测关系？

如果 R65 发现能够区分竞争解释的证据，执行最小 R66 消融。

如果 R66 证明存在可迁移信息，进入 R67，构造并验证最简单的新 prior 或 R25 组合方法。

如果没有，则停止现有 R56 的阈值、权重及架构修补，转入 R68 的新增信息可行性审计。

**最终目标不是将 R56 从 0.193805399 机械地优化到 0.193687363，而是建立能够在未知 Benchmark 上可靠改善成功概率预测的机制，并把这种机制转化为可复现、CPU 可执行、符合 PAIEC 规则的实际得分增益。**

保留当前 R25、R56_segment 的探索记录和原提交 fallback，不自动覆盖或正式提交。