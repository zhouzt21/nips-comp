# PAIEC 双路线研究计划：No-Peer 可迁移任务表示与 Shared-Peer 部分共享斜率

## 一、任务定位

你是一名 **Scientific Research Architect、Probabilistic ML Researcher、OOD Generalization Expert 与 Codex Multi-Agent Orchestrator**。

目前 PAIEC 已完成 R1–R72。

由于 Codabench 服务中断，本轮暂时不处理 Hosted 页面、Starter Kit 下载或真实 coordinator 抽样问题。

**不要让该外部阻塞影响研究执行。**

并行推进两种运行时假设：

- **Track A — No-Peer：** 每次预测只允许利用当前 Subject 自己的支持标签。
- **Track B — Shared-Peer：** 每次预测允许利用同一 Benchmark 内其他 Subject 当前已揭示的标签。

两条路线必须独立评价、独立记录。Track B 的条件模拟结果不能被解释为已经获得的真实 Hosted 成绩。

本轮使用纯 CPU 训练和推理。若需要任务需求标注，由 Codex 显式启动 **GPT-6 Luna（`gpt-6-luna`）子代理**完成。

不要直接让大型模型代替 PAIEC 最终预测器。

最终产物应包括研究结果、可运行代码、原始实验数据、明确的继续/停止决定，以及可回退的预测实现。

## 二、恢复项目状态

工作目录：

`D:\nips-comp\nips-comp\paiec_baseline`

优先读取以下现有报告：

- `reports/lexical_peer_evidence_findings.md`
- `reports/lexical_peer_evidence_tables.md`
- `reports/lexical_peer_evidence_reproduction.md`
- `reports/transferable_prior_mechanism_findings.md`
- `reports/historical_positive_followup_findings.md`
- `reports/reproduction_v4_findings.md`
- `research_loop/HIGH_LEVEL_STATE.md`

读取并核对 R25、R53、R56、R64、R69–R72 的原始实验合同、模型和源代码。

当前必须保留的基准：

| 方法 | ALC ↓ |
|---|---:|
| Frozen smoothed mean 2 | 0.198687363 |
| R25 | 0.195123926 |
| R56_segment | 0.193805399 |
| R72 pooled3 | 0.190683911（条件共享模拟） |
| R72 diff3 | 0.190885868（条件共享模拟） |

R25 是当前 own-only 的主要研究参照；R72 是共享情景的主要研究参照。

不得覆盖原有权重、提交 fallback 或 R1–R72 历史记录。

---

# TRACK A：不共享情况下的研究路线

## A1. 研究问题

此前 R54/R56/R67 表明：

**合法 item 字段包含部分低预算概率信息，但原有文本/metadata 表示没有形成稳定的跨 Benchmark 概率关系。**

本轮研究：

> 能否利用任务需求（Task Demand）这一更结构化的表示，得到比现有 MiniLM/metadata 更可迁移的信息，并改善 R25 的预测？

不要预设 ADeLe 一定有效。

特别区分：

1. 大模型任务需求标注本身是否可靠；
2. 任务需求特征是否有新增预测信息；
3. 新信息能否跨 Benchmark 迁移；
4. 新 item 能否由轻量 CPU 模型自动提取同类特征。

## A2. Codex 调用 GPT-6 Luna 子代理进行标注

首先恢复 R53 已有工作：

- 已准备的 128 distinct items；
- 四个原始 ADeLe rubric 维度；
- 512 个对应标注任务；
- 作者原始 rubric、prompt builder 和版本记录。

不要凭印象重新创造 rubric 或更换四个维度。

### 子代理组织

Codex 主代理负责调度和核验。

显式启动若干 `gpt-6-luna` 子代理，优先按**不同 item 分片**并行标注。

要求：

- 每个 item–rubric 判断使用相同冻结的作者 rubric；
- 评分遵守原始 0–5 离散等级；
- 只能读取合法 item 内容及该 rubric 需要的上下文；
- 不读取 Subject 的真实 response label、评价成功率或 R25 预测误差；
- 不根据模型正确与否调整任务难度分数；
- 每条记录包含 item identity、rubric ID、等级、简要判定依据、模型版本和 prompt hash；
- 无法判断时记录歧义，不编造确定标签。

使用原作者的标注协议，而不是让子代理自由评价题目“难不难”。

### 标注质量控制

随机抽取一部分任务，由**独立 GPT-6 Luna 子代理重新标注**。

审计：

- 等级一致率；
- 分歧超过一级的比例；
- 各 rubric 的等级分布；
- 不同 Benchmark 的标注偏差；
- 输入长度或文本截断的影响；
- 无效输出及解析错误。

必要时由 Codex 主代理根据 rubric 原文复核争议样本，但不得参考目标成功标签裁决任务需求等级。

保存原始响应和最终标注表。

**这批标注是教师产生的任务需求特征，不是 response 成功率的真实标签。**

## A3. 先判断任务需求表示本身是否有价值

使用合法训练来源，构建 Benchmark-grouped 对照。

至少区分两种不同的使用方式：

**方向 A：低预算先验**

检验任务需求特征能否为 B0/B1 提供比现有输入更可靠的成功率估计。

必要对照：

- 常数和平滑均值；
- 历史 R56/R67 先验；
- 原 MiniLM/metadata 先验；
- 加入任务需求特征后的先验。

**方向 B：支持标签条件下的局部对应**

检验任务需求相似性是否补充了 R25 的语义局部对应关系，尤其在 B3/B7/B15/B31。

不要默认任务需求特征只能用于 B0/B1。

但是新模型必须证明：与 R25 相比，增量确实来自需求特征，而非更大模型容量或更多输入字段。

使用任务需求标签置换、特征消融和配对 Brier 对照检验独立价值。

## A4. 蒸馏为 CPU 可部署的需求特征提取器

这是 Track A 的硬条件。

即使 GPT-6 Luna 生成的高质量标签对公开数据有预测帮助，隐藏测试中的新 item 也未必能调用同一标注子代理。

因此必须建立：

**Item → 现有 MiniLM/合法元数据 → 需求维度估计**

先尝试小型 Ridge、浅层 MLP 或其他低复杂度 CPU 预测器。

分别评价：

- 教师标签的预测价值：仅作为信息价值诊断；
- 学生模型的需求等级预测质量；
- 学生特征的实际 PAIEC Brier 增益。

教师标签的效果不能冒充学生模型的部署效果。

如果 128 个 item 的监督量不足，可以在 Pilot 证明有潜力后，扩大训练来源中的标注规模；仍使用 Luna 子代理分批标注，并通过 Benchmark 分组和需求覆盖选择样本。

不要直接无条件扩展到所有 18 个 rubrics 或大规模全量标注。

## A5. No-Peer 完整评价

只有来源内发现任务需求特征的独立价值，才构造一个最小新方法。

外层仍采用：

5,242 pairs × 3 item-split seeds × 六预算。

保持 R25 和 R56 的冻结对照，报告：

- Pair-equal Brier ALC；
- Benchmark-equal ALC；
- 三fold差异；
- 各预算增益；
- Benchmark-cluster bootstrap；
- `alignment_faking` 与其余 Benchmark 的贡献；
- 新任务需求特征是否导致严重回归；
- CPU 训练、推理及新 item 特征提取耗时。

若最终没有可迁移增益，不继续增加维度或扩大需求模型，保留 R25 作为 No-Peer 方案。

---

# TRACK B：共享情况下的研究路线

## B1. 研究问题

R72 pooled3 已在条件共享场景中取得明显收益：

\[
ALC=0.190683911
\]

其主要机制是：

**不同 Subject 共同估计当前 Benchmark 中的局部 item–response 方向，但每个 Subject 保留自己的平均成功率。**

删除同输入 peer 后仍保留约 95.45% 的增益。

然而，目前尚未验证：

- 共享斜率在任意 peer 组成下是否成立；
- 不同 Subject 是否总具有相同的局部响应方向；
- 哪些情况下部分共享优于完全共享；
- 具体收益有多少来自表示、中心化、正则化及新增标签数量。

本轮不要直接扩大 pooled3 的模型复杂度。

先建立机制对照，再决定是否尝试部分共享。

## B2. 共享斜率的稳定性与异质性诊断

使用 R72 冻结预测和合法 acquisition 数据。

重点检查：

**A. Subject 间斜率一致性**

对于同一 Benchmark，不同 Subject 的中心化 response 与 item 特征的关系是否具有共同方向？

在有限标签下无法准确估计单独斜率时，应报告不确定性和样本量，不能将噪声估计当作实际斜率差异。

**B. Peer 数量和组成**

使用预先冻结的随机 cohort 规则，研究：

- 0 peer；
- 1 peer；
- 2 peers；
- 3 peers；
- 缺失部分 peer；
- peer 标签覆盖不均；
- 完整可见 Subject 配置重复或冲突。

不得依据测试表现挑选最有利的 peer。

重点看是否存在稳定的收益随有效 peer 信息增加而改善，以及是否出现负迁移。

**C. 相同输入与不同输入**

保留 R72 same3、pooled3 和 diff3 强对照。

不要重复把同输入覆盖率当作共享效果的全部解释。

## B3. 识别 R72 真正有效的组成部分

原 R36 与 R72 同时存在表示维度和 ridge 配置差异，不能直接将改进归因于某个特征组。

在相同合法 acquisition、Subject 组成、标签预算及评分条件下，采用少量受控消融：

- 384 维语义表示与 R25 语义+metadata 表示；
- 同一表示下的 own-only 与 pooled；
- 保留与去除 Subject 内中心化；
- 在其他条件不变时检查原 R36/R72 正则化设定的影响。

不要展开大规模二维或三维超参数网格。

重点回答：

**R72 的优势究竟来自跨 Subject 的共享证据本身，还是特定表示与估计器配置的交互？**

只使用合法来源验证选择参数，不能利用已查看的外层 Benchmark 反向调节。

## B4. 条件执行：Partial Pooling / Hierarchical Shared Slope

只有 B2/B3 发现不同 Subject 的局部响应方向存在有意义的异质性，才执行本阶段。

考虑如下机制：

\[
\widehat p_{s,i}
=
\operatorname{clip}
\left(
\bar y_s+
\phi(x_i)^\top
(\beta_{\mathrm{shared}}+\delta_s),
0,1
\right)
\]

其中：

- \(\bar y_s\) 来自当前 Subject 自己的支持标签；
- \(\beta_{\mathrm{shared}}\) 使用当前可见 peer 和 own acquisition 估计；
- \(\delta_s\) 是 Subject 特异的局部修正；
- 对 \(\delta_s\) 施加较强收缩，避免有限支持标签下过拟合。

这是一种待验证的层级建模方案，不是 R72 已经实现并验证的模型。

它希望在下面两种极端之间取得更好的折中：

**完全 own-only：** 无法充分利用 peer。

**完全共享斜率：** 可能错误地强迫不同 Subject 具有相同的局部响应关系。

优先采用小规模正则化线性求解，而不是训练新的大 MLP。

最重要的对照：

1. R25 own-only；
2. R72 pooled3；
3. R72 diff3；
4. 新 partial pooling；
5. Peer 标签对应关系置换。

评价 partial pooling 是否在减少严重 Benchmark 退化的同时保留 R72 大部分收益。

如果无法改善，停止该结构，不继续扫描个体偏移强度。

## B5. Shared-Peer 完整评价与工程准备

保留 R72 的条件模拟属性。

在同样的 5,242-pair × 3-seed 评价下，分别报告：

- 0/1/2/3 peer；
- 各预算；
- Pair-equal 与 Benchmark-equal；
- 三fold差异；
- 不同 peer 组成；
- `alignment_faking` 与其他 Benchmark；
- 最大严重回归；
- Bootstrap；
- 每个预测请求的标签数、计算复杂度与内存成本。

构建统一的输入适配器：

- peer 缺失时自动回退 No-Peer 预测器；
- Subject 配置身份不明确时不共享；
- 只使用当前明确可见的标签；
- 不依赖跨调用内存；
- 不使用隐藏 item ID、Benchmark 名称或额外标签；
- 不向线上引入训练时才存在的内部 identity 字段。

使用合成输入完成本地 smoke test，不能宣称等价于真实 Hosted 验收。

---

# 三、两条路线如何组织与评估

## 1. 并行工作原则

Codex 主代理统筹实验。

Track A 的 ADeLe 标注工作委派给 `gpt-6-luna` 子代理。

Track B 的统计实验可以由独立执行代理开展，但两条路线禁止写入彼此的实验目录或同时修改同一核心评分文件。

共享只读基线、数据与缓存。

新增预测器和实验合同各自隔离。

## 2. 使用共同评价协议，但不混淆两种假设

No-Peer 和 Shared-Peer 不能直接作为同一信息预算下的公平算法排名。

应分别报告：

**No-Peer Track：** 当前 Subject 自己的支持标签预算。

**Shared-Peer Track：** 当前 Subject 支持标签预算，加上明确记录的额外 peer 标签数量及组成。

只在给定相同可见信息的情况下比较不同模型机制。

绝不能将 R72 的条件共享收益说成相同总标签预算下的无条件优势。

## 3. 探索与晋级门槛

沿用“探索宽进、晋级严出”。

小型机制 Pilot 只要有明确收益信号和较高信息价值，可进入一次冻结的完整探索评价。

Research Success Gate 继续保持：

- 相对两种强均值基线改善至少 0.005 ALC；
- 三fold改善；
- Benchmark-equal改善；
- Benchmark-cluster CI 上界低于零；
- 没有无法解释的严重回归。

公开 Benchmark 被反复用于研究，不能将本轮外层分数视为独立盲测。

对于 Shared-Peer，即使通过数值 Gate，在真实 Hosted 可见条件得到确认前也只能标记 `conditional_simulation`，不能自动部署。

## 4. 执行优先级

首先并行启动：

**Track A：ADeLe 128 items × 4 rubrics 的 Luna 标注 Pilot。**

**Track B：R72 共享斜率稳定性、peer 组成与核心因素消融。**

这两项完成后：

- 若 ADeLe 显示独立信息，进入 CPU student 和 PAIEC 验证；
- 若共享斜率存在明显异质性，进入 partial pooling；
- 若两者缺少正面证据，分别停止，不继续盲目增加复杂度。

只有在单独增益得到验证后，才考虑将需求表示应用于共享模型；不要把尚未验证的 ADeLe 和 R72 直接组合。

---

# 四、交付要求

采用实际仓库当前可用的轮次编号，执行前先检查实验登记，避免覆盖已有 R73+。

建议分开交付：

**No-Peer**

- ADeLe Luna 标注任务、原始标签与复核记录；
- 需求特征 CPU student；
- 来源内及外层 Brier 评价；
- 新信息、迁移和部署边界报告。

**Shared-Peer**

- 共享斜率异质性与 peer 覆盖分析；
- 表示、中心化及正则化的受控消融；
- 触发时的 partial pooling 实现与评价；
- 合成 runtime smoke tests；
- 条件共享数值与风险报告。

**全局**

- 更新 `HIGH_LEVEL_STATE.md`；
- 更新历史正负证据信号登记；
- 保存实际源码 hash、数据版本、训练与评价身份、概率和重计分结果；
- 保留 R25、R56、R72 原始资产和 fallback；
- 不正式提交，不产生未经允许的收费 API 请求。

完成后分别回答：

**No-Peer：任务需求标注是否提供了超出现有语义与 metadata 的可迁移信息？能否被 CPU student 有效利用？**

**Shared-Peer：R72 的收益来自什么可共享关系？何时应完全共享、部分共享或回退 own-only？**

最后给出两条路线各自的最强实际候选、真实证据强度、资源成本、失败边界和下一步最有价值的行动。

**现在开始：先检查工程状态，启动 Track A 的 GPT-6 Luna 标注子代理与 Track B 的冻结证据诊断。不要仅输出执行计划，要完成条件允许的真实实验。**