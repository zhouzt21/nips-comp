# PAIEC Research Down–Up Loop v3：基于 High-Level Recognition 的持续优化

## 一、任务与目标

你是一名自主科研 Agent，负责继续推进 PAIEC 竞赛算法优化。

采用 **High-Level Recognition 驱动的上钻—下钻循环**，而不是反复尝试不同模型：

\[
\boxed{
\text{现象}
\xrightarrow{\text{上钻}}
\text{机制与问题重构}
\xrightarrow{\text{选择研究算子}}
\text{新研究路线}
\xrightarrow{\text{下钻}}
\text{判别实验}
\xrightarrow{\text{证据更新}}
\text{下一轮}
}
\]

核心原则：

- **上钻**：从实验结果中识别更高层的科学问题，更新对瓶颈的理解。
- **研究算子**：根据瓶颈选择合适的组合思路，而不是机械套用 ideation patterns。
- **下钻**：把新的高层假设转化为可验证的机制、算法和实验。
- **持续迭代**：未达到冻结的 Success Gate，就根据新证据继续研究；不因完成固定轮数而停止。

使用 **AnySearch** 进行必要的外部论文、理论、方法谱系和实现调研。

## 二、从 R31 恢复研究状态

首先阅读现有项目，而不是重新开始：

项目：

`E:\VScode_data\project\nips-comp\paiec_baseline`

重点文件：

- `reports/research_loop_v2_2026-10-22.md`
- `reports/research_loop_v2_reproduction.md`
- `research_loop/HIGH_LEVEL_STATE.md`
- `research_loop/OPERATOR_LEDGER.md`
- `research_loop/RESUME.md`
- `runs/research_loop/R31/`

以及四份研究参考材料：

- `Research Down–Up Loop.md`
- `High-level 认识维度与相互关系.md`
- `Model-Nature_High-Level_Recognition_Paper_List_v1_1.md`
- `常见的组合思路.md`

需要恢复 R1–R31 的关键发现、失败机制、已排除路线、算法与实验代码，而不是只读取最新分数。

### 当前关键事实

R31 已覆盖 5,242 个合格 pair、三个随机种子。

| 方法 | Brier ALC |
|---|---:|
| Smoothed mean strength 2 | 0.198687 |
| 局部语义 R7 | 0.195178 |
| 语义 + metadata R25 | **0.195124** |

当前 R25 相对 frozen baseline 改善约 **0.003563**，距离要求的 0.005 仍差约 **0.001437**。

必须继承以下认识：

1. **已有局部 item 信息利用的正面证据。** R10 的对应关系破坏实验表明，局部语义方法的收益主要来自当前已揭示标签与 item 的关系，而不是简单的 pair 均值修正。
2. **跨 benchmark 的 zero-label prior 仍不可靠。** 原 MLP 已多次失败，扩大数据、部分训练与先验改造没有根本解决问题。
3. **R25 metadata 的独立增益很小。** 完整 pair 覆盖下相对 R7 仅约 0.000055，不应继续无依据地堆叠特征。
4. **评价覆盖会显著影响结论。** 原 737-pair 结果与 R31 的 5,242-pair 结果存在差异；alignment_faking 占后者约 81.6%。
5. **64 evaluation-items/pair 上限仍存在。** 其与完整官方评分的差异尚未充分审计；重复 response 及不可由 item 输入解释的结果噪声仍值得区分。
6. **共享证据是另一条条件研究分支。** R20 在 cohort4 下有正面收益，但 hosted 场景的共享范围尚未确认，不能与独立 pair 主结果直接比较。

以上证据支持研究进展，但不足以宣称新方法已经达到成功门槛。

---

## 三、核心循环：Upward → Operator → Downward

每轮只选择 **一个最值得解决的核心未知**，并完成以下步骤。

### Step 1 — Upward：重新定位瓶颈

从已有或最新的实验异常出发，逐层判断：

\[
\begin{aligned}
&\text{Measurement / Protocol}\\
&\uparrow\\
&\text{Information / Identifiability}\\
&\uparrow\\
&\text{Representation / Inductive Bias}\\
&\uparrow\\
&\text{Data / Objective / Training Dynamics}\\
&\uparrow\\
&\text{Runtime Adaptation / Evidence Use}\\
&\uparrow\\
&\text{Task Formulation}
\end{aligned}
\]

这些不是固定因果顺序，而是可选择的认识维度。

利用 High-Level Recognition Framework 判断：

- 当前失败究竟发生在哪一层？
- 是信息不存在、未被表达、没有学到，还是没有正确利用？
- 现有方法隐含了什么 load-bearing assumption？
- 当前科学问题是否需要重新表述？
- 哪些观察支持这个判断？有哪些竞争解释？

上钻到**证据能够支持的最高层级**即可，不强行制造宏大解释。

必要时通过 AnySearch 开展两类调研：

- **Problem-local**：已有领域方法、失败案例、理论限制。
- **High-level / Model-nature**：可迁移的 representation、architecture、learning、adaptation、information 机制。

对关键论文核实原始方法与适用假设，区分原文证据和跨领域推论。

**每次上钻必须得到一个具体产物：修订后的核心机制假设或 Problem Formulation。**

### Step 2 — Operator：根据诊断选择组合思路

从《常见的组合思路》选择最合适的 Research Operator，允许形成有依赖关系的 Operator Sequence。

例如：

| 已识别瓶颈 | 候选研究操作 |
|---|---|
| 评价含混、混杂因素未分离 | Audit → Confound-Isolating Diagnostic |
| 信息不足或问题对象定义不当 | Audit → Limit → Reframe |
| 有明确结构，但当前表示没有保留 | Audit → Encode by Construction |
| 多种异质机制被统一处理 | Diagnostic → Decompose → Differentiated Treatment |
| 局部证据与历史先验需要不同处理 | Reframe → Adapt by Conditioning |
| 学习目标没有诱导需要的能力 | Audit Objective → Property-Targeting Supervision |
| 存在更合适的数学求解形式 | Reframe → Delegate to Appropriate Solver |

这些仅为候选，不是必须执行的路线。

对于选中的算子，简要说明：

1. 当前要改变什么对象或假设；
2. 为什么该算子适用；
3. 前后算子有什么逻辑依赖；
4. 预期产生什么新机制；
5. 什么实验会否定它。

**禁止为了组合而组合；若一个简单算子已经足够，就不增加复杂度。**

### Step 3 — Downward：设计并执行判别实验

将高层机制具体化：

\[
H_{\text{high}}
\rightarrow H_{\text{mechanism}}
\rightarrow P_{\text{observable}}
\rightarrow E_{\text{controlled}}
\]

每轮优先一个高信息价值实验。

执行前冻结：

- 待区分的竞争假设；
- 核心控制变量与强 baseline；
- 预期结果及不同结果对应的解释；
- 数据分组与评价协议；
- 放弃或继续该路线的条件。

直接实施代码、运行实验并保存真实结果。

不要只生成实验规划。

### Step 4 — Evidence Update：结果反推高层认识

实验完成后回答：

- 结果支持或反对哪个机制？
- 是算法失败、实验设计失败，还是原有 high-level assumption 不成立？
- 哪些认识可以保留，哪些应降低置信度？
- 哪条研究路线应该继续、转向或终止？
- 下一个最值得获取的信息是什么？

将更新写入 `HIGH_LEVEL_STATE.md` 和实验日志。

若 Gate 未达到，就以新的高层判断开始下一轮。

---

## 四、R32 的初始研究方向

不要直接继续扩展 encoder、metadata、正则强度或训练数据。

**首先审计：当前 0.003563 的改善为什么仍没有达到 0.005，以及剩余差距究竟受什么限制。**

建议先进行一次上钻：

### 高层问题 A：Measurement 是否限制了我们对收益的认识？

R31 已覆盖全部合格 pair，但仍限制为每 pair 最多 64 个 evaluation items。

应审计：

- 64-item 抽样相对完整评价产生多大偏差或方差？
- repeated responses 如何影响 item-level 可解释误差？
- alignment_faking 的支配性是否掩盖其他 benchmark 的表现？
- 不同预算的收益分布是否提示某种明确的改进机会？

优先冻结预测器，开展有代表性、分层的完整 evaluation-item 对照，必要时再扩展评估范围。

该实验属于：

**Audit Measurement → Confound-Isolating Diagnostic → Reassess Bottleneck**

注意：修正评价并不自动意味着算法获得改进。不得通过选择评价子集制造过 Gate 的结果。

### 高层问题 B：低预算与高预算是否需要不同信息机制？

已有局部 item 方法主要在具备一定已揭示标签后产生收益。需要在 R31 完整 pair 结果上重新进行逐预算分解。

如果预算 0、1 仍几乎没有改善，而预算 7–31 持续改善，应区分：

- 冷启动信息不足；
- 当前局部估计的样本复杂度；
- 可以转移的先验信息；
- 需要在观察标签后才可识别的 item-level 关系。

如果这些机制确实不同，再考虑：

**Diagnostic → Decompose → Differentiated Treatment**

而不是直接给不同预算设置一套未经验证的超参数。

### 高层问题 C：可利用的信息是否已经接近当前方法的上界？

如果新诊断表明当前 item 机制的额外增益空间有限，则使用 oracle diagnostic、可辨识性分析或受控模拟，研究限制来自：

- item 信息本身；
- 表示方式；
- label budget；
- task distribution shift；
- 推理时的信息利用机制。

Oracle 只能用于分析，不能用于实际预测、参数选择或提交。

**上述 A、B、C 是候选研究问题。先根据已有证据和预期信息价值排序，选择一项执行，不要求同时开展。**

---

## 五、Gate、实验纪律与持续控制

沿用冻结的 Research Success Gate，不事后降低标准：

- 相对 smoothed mean strength 2 和训练内选择的强简单基线，绝对 pair-equal ALC 改善至少 **0.005**；
- benchmark-grouped 三折方向一致；
- benchmark-equal ALC 同时改善；
- benchmark-cluster paired bootstrap 95% 差值区间上界小于 0；
- 无未解释的严重 benchmark regression；
- 关键新增机制有受控实验支持。

上述为研究晋级标准。**独立泛化证据**及**提交 ZIP 验收**仍需分别检查。

继续遵循：

- 原有公开 benchmark 已反复参与开发，不能重新称为独立盲测。
- 保持官方六档预算、pair 等权 Brier ALC 和标签隔离。
- 独立 pair 与条件共享 cohort 的结果必须分开。
- 不重复 R1–R31 已经充分失败、且没有新增机制依据的搜索路线。
- AnySearch 调研应解决明确的认知缺口，不是每轮堆砌文献。
- 未经用户授权，不正式提交 Codabench 或公开发布。
- 保留 `dist/smoothed_mean_strength2.zip` 作为 fallback。
- 持续核对官方截止时间与实际执行资源。

若一个方向被否定，应上钻并改变研究动作；不要在同一失效假设下无限调参。

未达到 Gate 且仍有高价值可执行行动时，继续下一轮。遇到实际环境、资源或期限阻碍时保存可恢复状态，不虚构后台持续执行。

---

## 六、每轮的最小交付

每轮只需提供以下六项，避免冗长研究报告淹没核心判断：

1. **Current Bottleneck**：当前最重要的未知。
2. **Upward Update**：这轮更新了什么 high-level 认识，依据是什么。
3. **Operator Choice**：为什么选择这个研究操作或组合。
4. **Downward Experiment**：具体实施了什么判别实验。
5. **Evidence & Gate**：真实结果、机制解释和 Gate 差距。
6. **Next Move**：下一轮继续下钻、上钻还是更换路线，为什么。

保存代码、结果、日志、关键文献和简明断点；避免重复生成全套项目文档。

---

## 七、立即执行

从 **R31 → R32** 开始。

先恢复当前项目与结果，完成最新逐预算及评价覆盖的证据审计，判断 A、B、C 哪个问题最值得优先处理。

必要时使用 AnySearch 验证相关统计理论、方法限制或机制假设。

然后选取具有明确适用理由的 Research Operator，设计并执行一个高判别力实验。

实验完成后上钻更新认识、检查 Gate，并继续下一轮。

**不要先规划几十轮，也不要随机列出十种候选算法。**

始终遵循：

\[
\boxed{
\text{观察证据}
\rightarrow
\text{上钻认识}
\rightarrow
\text{选择算子}
\rightarrow
\text{下钻实验}
\rightarrow
\text{更新认识}
}
\]

**每一轮必须至少产生一种实质进展：更可靠的机制判断、更清楚的能力边界，或经严格验证的性能改进。研究目标未达到时，不把“完成一轮”当作完成整个任务。**