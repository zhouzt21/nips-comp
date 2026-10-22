# Research & Engineering Prompt：PAIEC 竞赛概率预测算法的失效机制诊断、跨 Benchmark 泛化改进与稳健性能优化

## 一、研究角色与核心任务

你是一名兼具以下专长的研究与工程专家：

- **Probabilistic Prediction & Calibration Researcher**：理解概率预测、Brier Score、概率校准、贝叶斯收缩、概率融合及预测不确定性。
- **Out-of-Distribution Generalization Researcher**：研究模型在未知 benchmark 上的泛化失效、特征迁移、分布偏移及负迁移机制。
- **Few-Shot Adaptation & Meta-Learning Researcher**：理解如何结合跨任务学习得到的预测先验与当前任务的少量观测标签。
- **Scientific Failure Diagnostician**：能够从具体实验异常中形成竞争性解释，并通过有判别力的实验定位限制性能的真正原因。
- **ML Algorithm & Experimental Systems Engineer**：能够阅读现有实现，构造轻量算法改进，执行训练、验证、消融、打包与复现。

你的任务不是重新跑通 PAIEC baseline，也不是直接设计一个更大的 MLP。

**核心目标是：基于已经完成的真实数据实验，诊断当前 MLP 在未见过的 benchmark 上无法稳定优于简单统计方法的原因，寻找能够可靠改善 Brier ALC 的算法机制，并通过严格的 benchmark-level 泛化实验验证改进是否成立。**

目标优先级为：

1. 获得在未见过的 benchmark 上稳定优于强简单 baseline 的方法。
2. 理解性能提升或失败的具体机制，确保结果不是偶然数据划分、调参或数据泄漏的产物。
3. 维持轻量、可独立推理、符合 PAIEC 竞赛限制的工程实现。
4. 在比赛截止前获得可以冻结提交的最优证据支持版本，并形成技术报告所需的有效实验结果。

**稳定优于简单方法是优化目标，而不是必须通过选择性汇报得到的预设结论。** 如果证据不足，必须保留简单方法作为提交备选，明确指出阻碍性能改进的瓶颈和已被实验排除的假设。

---

## 二、现有工程与实验材料

### 2.1 工程路径

项目根目录：

`E:\VScode_data\project\nips-comp`

Baseline 及当前 MLP 工程：

`E:\VScode_data\project\nips-comp\paiec_baseline`

现有整体计划：

`E:\VScode_data\project\nips-comp\plan.md`

完整实验报告：

`E:\VScode_data\project\nips-comp\paiec_baseline\reports\mlp_benchmark_holdout_2026-10-22.md`

该实验报告也已作为附件提供。

现有数据：

`D:\paiec_payload\data`

现有实验记录主要分布于：

- `runs/mlp_trials/`
- `runs/mlp_cv/`
- `runs/mlp_split_2026.json`
- `runs/public_data_audit.json`

主要实现：

- `mlp/model.py`
- `mlp/train.py`
- `mlp/evaluate.py`
- `mlp/make_split.py`
- `mlp/make_cv_splits.py`
- `mlp/summarize_cv.py`

最终提交接口为：

`predict(input, labeled) -> float`

请首先阅读实际存在的代码、配置、权重元数据及已有实验结果，不要仅凭报告推测实现细节。

### 2.2 官方竞赛资料

竞赛规范：

https://aims.stanford.edu/competition

官方 baseline：

https://github.com/aims-foundations/paiec_baseline

公开数据：

https://huggingface.co/datasets/aims-foundations/measurement-db

该竞赛的关键特征是 benchmark-level cold start：

训练阶段可以使用公开 benchmark 的合法数据，但隐藏测试来自未见过的 benchmark。

评分采用六档标签预算：

\[
n\in\{0,1,3,7,15,31\}
\]

并对每个 subject–benchmark pair 计算：

\[
ALC=0.1B_0+0.2B_1+0.2B_3+
0.2B_7+0.2B_{15}+0.1B_{31}
\]

最后对 pair 等权平均，越低越好。

所有优化必须服从这一任务与评价机制。

应以官方当前规则为准，重点复核预测接口、allowed evidence、acquisition/evaluation 分离、benchmark inventory、运行环境和截止日期。

---

## 三、已经完成的工作与关键实验结论

以下结果来自前一轮公开数据本地模拟。

**不要重新把这些已完成的工作当作本轮主要任务。**

### 3.1 数据与模型

已经准备约 1,642 万 response rows，其中约 1,629 万条具有二元 outcome。

当前 MLP：

- 使用八个 subject 字段，以及 item text、item features、interactors；
- 将输入通过稳定 signed hashing 编码为 2,048 维特征；
- 使用隐藏层为 128、64 的两层 ReLU MLP；
- 输出 sigmoid 概率；
- 通过当前 subject–benchmark 的已揭示标签进行 residual-based adaptation；
- 使用 pair-weighted Brier loss 和 AdamW 训练；
- 最终可通过 NumPy 独立推理。

已有训练使用每个 benchmark 最多 400 条 response rows 的采样限制；因此实际训练样本远少于全部公开数据。

已有早停模型通常在 epoch 2 停止。

这些设计可能与当前泛化问题有关，但不能在诊断之前直接认定是根因。

### 3.2 开发集结果

| 方法 | Brier ALC |
|---|---:|
| Empirical Mean | 0.236505 |
| Smoothed Mean，strength 2 | 0.205411 |
| MLP prior，不使用标签 | 0.208058 |
| MLP + residual adaptation，strength 4 | **0.191819** |

开发集上 MLP 看起来有明显优势。

但该开发集的 1,061 个 subject–benchmark pairs 中，948 个来自 `afrieval`，结果存在 benchmark 组成不均衡的问题。

### 3.3 本地测试结果

| 方法 | Brier ALC |
|---|---:|
| Smoothed Mean，strength 2 | 0.238503 |
| Smoothed Mean，strength 8（事后诊断） | 0.234650 |
| MLP prior，不使用标签 | 0.253891 |
| MLP + residual，strength 4 | 0.236854 |

这些测试已经经过模型开发过程中的多次查看。

因此，不能将测试集上的事后方法比较当作未受污染的模型选择证据。

特别是 strength 8 的平滑均值属于测试后诊断，不是独立验证得到的最优参数。

### 3.4 三折 benchmark-grouped 验证

| 方法 | Pair-equal Brier ALC |
|---|---:|
| Smoothed Mean，strength 2 | **0.208331** |
| MLP prior，不使用标签 | 0.275033 |
| MLP residual，strength 2 | 0.215339 |
| MLP prior mix 0.25 + residual | 0.208883 |
| MLP prior mix 0.5 + residual | 0.210238 |

三折验证表明：目前的 MLP 先验在未见过的 benchmark 上可能严重失效。

仅将该 MLP prior 统一向 0.5 收缩，并未在已有验证中稳定优于 smoothed mean。

**因此本轮不能把简单重复调整 prior mix 或 residual strength 当作主要算法创新。**

如果再次采用相关思路，必须提出与先前实验不同的机制假设，并设计能够区分它们的实验。

### 3.5 当前提交状态

已有三个通过本地检查的 ZIP：

- `dist/smoothed_mean_strength2.zip`
- `dist/mlp_candidate_internal_alpha4.zip`
- `dist/mlp_full_public_epoch2_alpha4.zip`

其中 smoothed mean 是目前证据相对稳健的备选方法。

全公开数据重训的 MLP 虽然已通过提交接口测试，但这些精确权重并没有独立 held-out benchmark 的性能估计。

32 项本地测试已通过。

因此，**本轮主要目标是算法有效性和跨 benchmark 泛化，而不是重复证明工程可以运行。**

---

## 四、核心研究问题：为什么 MLP 不能稳定优于简单方法？

请首先将当前问题视为一个需要区分竞争性解释的科学诊断任务，而不是默认的超参数优化任务。

你需要从现有实验中分析两个可能相互关联、但不能混为一谈的问题：

**问题 A：为什么跨 benchmark 学习到的 zero-label prior 泛化不可靠？**

**问题 B：为什么已有标签的 adaptation 无法充分弥补 prior 失效，甚至可能带来负迁移？**

先明确在什么意义下二者可以分离，以及何种实验能够区分二者的影响。

### 4.1 建立竞争性机制假设

至少考虑以下候选解释，但允许根据实际代码与结果删除、合并或添加假设。

**H1：特征表达缺乏跨 benchmark 可迁移语义**

当前 2,048 维 signed hashing 混合了 subject 字段和 item 相关文本。

这种表示可能记住训练 benchmark 中常见的词汇与局部统计规律，却不能稳定表达新 benchmark 的 item 难度。

需要区分：

- item text 本身携带的信息不足；
- 当前编码方式没有有效提取信息；
- 特征在训练和测试 benchmark 间存在语义不一致；
- 哈希冲突或高维稀疏结构影响学习；
- 模型捕获了仅在特定 benchmark 中成立的偶然相关。

**H2：模型主要学习了 benchmark-specific correlation，而非可迁移规律**

某些词汇、subject 属性或格式在一个 benchmark 内与正确率相关，在另一个 benchmark 中可能完全不同。

一个在随机样本划分上表现优秀的模型，未必具备跨任务泛化能力。

需要判断模型是否依赖特定 benchmark 的表面特征，及其失效程度。

**H3：训练数据采样与主指标之间存在错配**

当前只有约两万条采样训练 responses，而完整公开数据具有千万级规模。

每 benchmark 固定上限 400 rows 的策略可能丢失了重要的：

- subject 多样性；
- item 多样性；
- 难度覆盖；
- 组内成功率变化；
- 少数但有信息量的 subject–benchmark pairs。

但增加训练数据也可能增加对特定 benchmark 的过拟合。

必须通过受控实验区分数据量不足、采样偏差与本质的跨域泛化问题。

**H4：MLP 概率先验存在明显校准失效或分布偏移**

开发集的 zero-label prior Brier 为 0.208058，而三折验证为 0.275033。

需要诊断：

- 预测概率是否系统性过高或过低；
- 极端概率是否过度自信；
- benchmark 间标签 prevalence 是否发生变化；
- 模型是否仍有 item 级排序能力，只是整体概率校准错误；
- 是否存在只有某类 benchmark 才发生的负迁移。

区分 discrimination、calibration、base-rate shift 与真正的条件预测关系失效。

**H5：当前局部标签融合机制不够合理**

当前 residual adaptation 可能没有充分区分：

- 学到的先验概率；
- 先验的不确定性；
- 当前 benchmark 的整体偏移；
- 已揭示标签带来的抽样方差；
- 当前 item 与 acquisition items 的差异。

某种统一收缩常数可能不适用于预算 1、3、7、15、31 的不同证据量。

但是不能预设更复杂的融合一定有效。

**H6：训练目标没有充分模拟测试时的少标签预测过程**

当前训练主要优化单个 response 的监督 Brier loss，但评测需要在新的 subject–benchmark pair 上结合极少量标签。

两者的统计问题并不完全相同。

需要判断 episodic training、cross-task adaptation training 或其他直接面向少标签预测的目标，是否可能优于独立训练 prior 后再做启发式 residual correction。

**H7：模型选择与验证协议放大了偶然优势**

开发集由单个 benchmark 主导，且 MLP 结构、早停和融合参数经历过多轮选择。

因此，观测到的优势可能部分来自 validation composition 和反复选择，而不是真正泛化。

需要将此类实验设计因素与算法本身的限制分开。

### 4.2 假设不能停留在文字层面

对每个重要假设给出：

1. 支持该假设的已有观测；
2. 与之竞争的其他解释；
3. 当前尚缺失的证据；
4. 最小的判别实验；
5. 预期不同结果分别意味着什么；
6. 假设被否定后应停止哪些优化路线。

不要把所有可能原因一次性铺开并逐个尝试。

**优先选择能够显著缩小算法设计空间的诊断实验。**

---

## 五、第一阶段：利用已有结果定位真正的性能瓶颈

本阶段以诊断为主，尽量避免重新训练大量模型。

### 5.1 分解六档预算下的误差来源

对主要方法计算：

\[
B_0,B_1,B_3,B_7,B_{15},B_{31}
\]

并定义相对于平滑均值的差异：

\[
\Delta B_n=B_n^{candidate}-B_n^{baseline}
\]

综合得到：

\[
\Delta ALC
=
\sum_n w_n\Delta B_n
\]

重点回答：

- 哪些预算造成了当前 MLP 的主要劣势？
- 哪些预算的改进对最终 ALC 最有价值？
- zero-label prior 的失败有多严重？
- residual adaptation 在哪些预算真正提供增益？
- 高预算下不同方法是否已趋于相近？
- 某些预算上的优势是否被其他预算上的损失完全抵消？

同时对每个 benchmark 分别进行这一分解，避免整体平均掩盖相反的失效模式。

### 5.2 区分 prior 失效的不同形式

如果有必要，在现有训练 benchmark 内使用 out-of-fold predictions 进行分析。

应研究：

**整体成功率偏移**

模型预测均值和真实成功率存在多大差异？

**概率校准**

是否存在极端置信度预测导致的 Brier 惩罚？

**item 区分能力**

在同一个 subject–benchmark pair 内，模型是否能够识别哪些 item 相对容易、哪些相对困难？

**跨 benchmark 的特征关系变化**

相同类型的特征是否在不同 benchmark 中呈现相反的预测关联？

应根据实际可获得的数据选择有意义的分析手段，例如 reliability diagrams、Brier decomposition、概率分布、paired residual analysis、within-pair ranking diagnostics。

不要机械生成大量图表。

需要明确哪些指标只是事后诊断，不能直接解释为可部署预测性能。

### 5.3 检查实际训练数据利用情况

详细检查：

- 每 benchmark 400 rows 的采样方式；
- 训练时各 subject–benchmark pair 的覆盖；
- pair weighting 的实际实现；
- 重复 item/response 的处理；
- 正负样本分布；
- early stopping 的触发机制；
- 训练误差与验证误差轨迹；
- 特征在不同 benchmark 的覆盖率；
- 哈希特征的有效维度和碰撞情况。

尤其需要判断：epoch 2 停止究竟表明模型开始过拟合，还是因为验证集不稳定、数据采样不足或学习率等因素导致早停判断失真。

输出一份**有证据排序的瓶颈诊断**，而不是泛泛列举所有可能问题。

---

## 六、第二阶段：从失效机制构造有针对性的候选算法

本阶段允许研究多种方向，但不要求全部实现。

候选方法必须针对前一步确认或尚待判别的具体瓶颈。

每个方向先完成最小可行设计，再决定是否投入训练资源。

### 6.1 方向 A：构造比当前 MLP 更稳健的预测先验

首先回答：

是否真的需要复杂的 item-level 神经网络，才能提供比 0.5 更好的跨 benchmark 预测先验？

可以研究：

- 利用可迁移 subject 属性构造先验；
- 使用分层或带正则化的能力先验；
- 在 subject 信息不可靠时收缩到全局先验；
- 显式估计 prior 的可靠性；
- 使用经过 benchmark-level cross-fitting 的概率校准；
- 条件化选择先验信息，但避免无依据地学习 benchmark 身份。

尤其应区分：

\[
\text{Good prior on seen benchmarks}
\]

与

\[
\text{Transferable prior on unseen benchmarks}
\]

如果全局训练均值或简单 subject 属性先验比现有 MLP 更稳定，应将它们纳入后续融合方案。

不要以神经网络为先验模型的唯一合法形式。

### 6.2 方向 B：改进 item 特征表示

如果证据表明问题来自当前特征表达，则研究更可迁移的 item representation。

可以探索：

- 区分 subject 与 item 的独立编码；
- 对结构化 item metadata 做字段感知编码；
- 更有表达能力的文本表示；
- 去除 benchmark-specific 的表面特征；
- 使用可迁移的 item 类型、结构或难度相关线索；
- 简单稀疏文本模型与 MLP 的比较；
- 在资源及规则允许的情况下使用已有预训练语义编码器。

但须检查三个问题：

1. 新特征是否在隐藏测试的 `input` 中确实可获得？
2. 是否在未见过的 benchmark 上仍然具有一致含义？
3. 额外计算、依赖和模型文件是否适合最终推理环境？

不要未经验证便假定 embedding 一定优于 hashing。

优先研究**特征所表达的知识是否可迁移**，其次才是特征维度或网络大小。

任何预训练模型的使用须符合官方的训练来源、披露和打包规则。

### 6.3 方向 C：改进训练数据组织与目标函数

如果现有采样策略可能是关键瓶颈，则设计有限规模的对照：

例如比较不同训练数据上限、pair-aware sampling 和 benchmark-aware sampling。

分析：

- 增加训练样本是否改善 OOD Brier；
- 训练多样性与训练总量何者更关键；
- 模型是否需要更强正则化；
- pair-weighted Brier 是否符合预期；
- BCE、Brier 及其他合理损失的比较价值；
- 是否需要限制某些 benchmark 对共享特征的支配作用。

应优先比较具有明确假设的少量配置。

训练数据扩大时，确保 held-out benchmark 的标签不参与训练、校准或特征监督。

### 6.4 方向 D：采用更合理的 Bayesian / shrinkage adaptation

由于平滑经验均值已经表现较稳健，该方向有较高研究价值。

考虑模型：

\[
p_0=f_\theta(s,x)
\]

其中 \(p_0\) 是跨 benchmark 学到的先验预测。

当前允许揭示的数据为：

\[
\mathcal D_n=\{(x_i,y_i)\}_{i=1}^{n}
\]

目标是构造：

\[
p_n(x)=
\operatorname{Adapt}(p_0,x,\mathcal D_n)
\]

研究的重点不是简单增加一个混合系数，而是回答：

**什么条件下应该相信跨 benchmark 先验？什么条件下应该更多相信当前 benchmark 的局部标签？**

候选机制包括：

- 基于 Beta-Binomial 或 empirical Bayes 的收缩；
- 先验强度与不确定性估计；
- 带正则化的局部截距 / logit correction；
- 根据预算变化调整先验和局部统计的相对权重；
- 基于已揭示样本预测残差的校准；
- 将 item-level 预测与 pair-level prevalence estimation 分离。

例如可以考虑：

\[
p_n(x)=
(1-\lambda_n)p_{\mathrm{local},n}
+
\lambda_n p_{\mathrm{prior}}(x)
\]

但 \(\lambda_n\) 应具有可解释或可验证的统计依据。

需要与已有固定 mix 和固定 residual strength 区别开来，避免重新执行已被三折结果否定的等价实验。

特别要处理 \(n=0\) 时局部均值未定义、\(n=1\) 时高方差及局部样本与 evaluation items 分布差异的问题。

### 6.5 方向 E：直接优化少标签适应过程

如果证据表明分离训练 prior 和 adaptation 存在结构限制，则考虑模拟竞赛的 episodic training：

在训练 benchmark 内：

1. 构造 subject–benchmark episodes；
2. 将可用 items 分为 acquisition 与 evaluation；
3. 从 acquisition 中采样不同预算的标签；
4. 基于已揭示标签产生 evaluation predictions；
5. 优化六档预算上的加权损失。

可以将目标表达为：

\[
\mathcal L
=
\sum_{n\in\{0,1,3,7,15,31\}}
w_n
\mathbb E_{\text{episode}}
\left[
\operatorname{Brier}(p_n,y)
\right]
\]

该方法的直觉是：训练阶段就让模型学习“拿到少量当前任务证据后应该如何改变预测”。

但必须验证它是否带来真正新增的信息利用能力，而不只是提高了模型复杂度或过拟合训练 benchmark。

在有限资源下，优先尝试小模型、简单 episodic objective 或可解析的低维适应模块，不必直接构造复杂的 meta-learning 网络。

### 6.6 暂不优先开展的工作

除非已有诊断强烈支持，否则暂缓：

- 大规模 MLP 架构搜索；
- 无目标地增大 embedding 维度；
- 引入复杂 LLM Agent；
- 大量重新调节已经失败的固定 prior mix；
- 同时修改特征、训练、适应与 acquisition；
- 利用已反复查看的 local test 进行模型选择。

主动标签选择可以作为后续独立方向，但应先获得一个有合理泛化能力的 predictor，以免同时优化 acquisition 与 prediction 导致无法归因。

---

## 七、第三阶段：设计能够判别改进是否真实有效的实验

这是本轮最重要的约束。

### 7.1 正确处理已经被查看过的验证集

原来的 development 和 local test 均已被分析，不能再次作为未经使用的独立测试集。

原始 training pool 内的三折结果也已参与当前方法的诊断。

因此需要建立新的实验开发协议：

- 在合法的公开训练 benchmark 中开展新的 grouped experiments；
- 训练、内部早停、校准、超参数选择在严格分离的 benchmark folds 中完成；
- 尽可能采用 nested grouped cross-validation 或明确分离的模型选择过程；
- 现有开发集与本地测试只能用于历史诊断，不应反复用其决定新模型；
- 固定随机种子、采样策略和训练配置；
- 如后续出现确实未被用于设计的新公开 benchmark，优先留作独立验证。

需要特别指出：

**重新随机划分同一批已经反复分析的 benchmark，并不自动产生一个全新的盲测集。**

这类实验仍然有助于判断方法的稳健性，但最终性能结论必须承认复用同一公开 benchmark 集合所带来的选择偏差。

不得依靠多次 Codabench formative feedback 反向推断隐藏 benchmark 或进行不透明的 leaderboard-driven tuning。

### 7.2 选择正确的比较单位

主指标必须是：

\[
\Delta ALC
=
ALC_{\mathrm{candidate}}
-
ALC_{\mathrm{baseline}}
\]

负数表示候选方法更好。

主要比较方式应采用相同：

- benchmark folds；
- subject–benchmark pairs；
- acquisition/evaluation item 划分；
- acquisition trajectories；
- item split seeds；
- 评分预算；
- evaluation response 集合。

这种 paired comparison 比分别运行两个模型后比较平均分更有统计效率。

区分：

- Pair-equal ALC：比赛主指标；
- Benchmark-equal ALC：跨 benchmark 稳健性辅助指标；
- Per-benchmark improvement：判断优势是否集中在少数 benchmark；
- Per-budget Brier：定位改善来自何处。

不要因为某个辅助指标更好就改变主目标。

### 7.3 明确定义“稳定优于 baseline”

在开展新的大规模实验前，先冻结胜出判据。

至少需要考虑：

1. 是否降低跨 benchmark 验证的 pair-equal ALC；
2. 改进幅度是否超过合理的统计波动与工程成本；
3. 是否在不同 benchmark folds 中方向一致；
4. 是否有少数 benchmark 出现严重性能退化；
5. 是否优于冻结的 smoothed mean strength 2；
6. 是否还能经受一个仅在训练/内层验证中调节的强平滑均值基线；
7. 不同随机种子下结果是否稳定。

如数据允许，使用以 benchmark 为聚类单位的 paired bootstrap、fold-level 差异或其他适当的分层不确定性分析。

不要将 item split seed 的极小方差误当作跨 benchmark 的泛化不确定性。

有限 benchmark 数量下，置信区间可能较宽，应如实报告。

可以预先设定实际有意义的最小 ALC 改善阈值，但其大小应根据已有波动、benchmark 数量和竞赛目标确定，不能在结果出现后反向修改。

### 7.4 先确定最小实验矩阵

不允许无限组合搜索。

建议按照“一个假设对应一个主要修改”的方式安排：

**Experiment 0：现有模型的失效分解**

仅分析历史结果与已有权重，尽量不重新训练。

**Experiment 1：强简单先验 / 收缩方法**

验证统计建模是否已经能够进一步改善 smoothed mean。

**Experiment 2：特征或训练数据的单一关键修正**

根据诊断选择最有希望的一个，不同时改变多个因素。

**Experiment 3：模型先验与局部标签融合**

只对具有证据支持的先验模型进行适应机制比较。

**Experiment 4：组合与稳健性验证**

仅将已单独显示价值的修改进行组合。

如果 Experiment 1–3 均无稳定改进，应停止盲目堆叠模块，重新诊断假设是否成立。

---

## 八、第四阶段：工程实施与实验执行

完成研究设计后，如果拥有本地工程和终端访问能力，请直接实施，不要停留在文字计划。

### 8.1 尽量复用现有代码

优先扩展：

`mlp/train.py`

`mlp/evaluate.py`

以及必要的特征编码或 adaptation 实现。

可以新增独立实验模块，但应避免破坏原有 baseline、权重及 ZIP。

旧方法必须仍然能够被复现。

### 8.2 运行成本控制

当前主要训练环境为 Windows、Python、PyTorch CPU，推理侧依赖 NumPy。

应尽量采用：

- 有代表性的少量训练配置；
- 分阶段增加数据规模；
- 先小样本验证实现；
- 复用已有数据及缓存；
- 严格控制并行计算与内存；
- 对昂贵实验使用明确的进入条件。

如果 GPU 可用可以利用，但不应以 GPU 或大规模预训练模型为完成任务的必要前提。

### 8.3 训练及验证记录

每个实验必须包含：

- 唯一实验 ID；
- 所检验的机制假设；
- 基础模型与具体修改；
- 使用的训练 benchmark；
- 训练和验证切分；
- 数据采样；
- 随机种子；
- 模型及训练超参数；
- 校准和 adaptation 参数；
- 每预算 Brier；
- Pair-equal ALC；
- Benchmark-equal ALC；
- Per-benchmark 分析；
- 运行时间与资源；
- 权重和代码版本；
- 结论与下一步动作。

对未运行、运行失败、部分完成和已验证的实验明确区分。

不允许填写推测分数。

### 8.4 保证训练和推理一致

对于新特征或新网络，检查：

- 训练与推理编码完全一致；
- 缺失字段与未知值有明确处理；
- 模型文件可以正确导出；
- 不依赖训练时的数据路径；
- 推理可重复；
- 只使用当前调用允许的标签；
- benchmark ID 仅用于合法证据关联，不作为隐藏任务身份记忆；
- 无 NaN、Inf、非法概率；
- 不出现跨调用状态泄漏；
- 所有必要文件可以打包进提交 ZIP。

若采用新的文本编码器或预训练模型，先确认符合竞赛模型打包、许可和计算限制。

---

## 九、第五阶段：根据实验结果迭代，而不是机械执行全部计划

每完成一轮关键实验，重新回答：

**当前发现是否真正改变了对问题的理解？**

按照结果进行自适应研究决策。

### 情况 A：MLP 先验在多数新 benchmark 上继续劣于 0.5

优先放弃继续扩大当前网络的路线。

转向：

- 对特征可迁移性的检验；
- 更简单且稳健的 prior；
- 强收缩或可靠性估计；
- 依赖局部标签的低方差方法。

### 情况 B：更好特征使 B0 改善，但高预算收益有限

说明跨 benchmark 先验可能已经改善，但适应策略尚不能充分利用 item-level prediction。

接下来优先研究如何将局部成功率信息与 item-specific prediction 有效结合。

### 情况 C：B0 没有改善，但标签预算 1–7 获得稳定收益

说明主要价值可能来自局部适应机制，而非纯粹 zero-shot prediction。

应进一步检验该收益是否来自更有效的方差控制或偏移校正。

### 情况 D：增加训练数据改善训练内性能，但 OOD 仍然不佳

优先研究训练 benchmark 和目标 benchmark 的关系，而不是进一步扩大训练数据。

可能需要改变特征不变性、训练目标或数据采样方式。

### 情况 E：方法仅在某些 benchmark 上获得很大提升

应研究能否从输入本身或已揭示证据中可靠识别这些适用条件。

不能利用公开 benchmark 名称推断隐藏 benchmark 身份来选择方法。

如果无法建立可靠的适用性判据，不能将局部优势描述为普适改进。

### 情况 F：改进小于评估不确定性

不应宣布方法成功。

可以保留为研究候选，但提交决策应优先考虑更简单、稳定且已有充分验证的实现。

---

## 十、最终产物

### 10.1 失效机制诊断报告

要求不仅描述现象，还要说明：

- 现有证据支持什么；
- 哪些解释仍然竞争；
- 哪些假设已被实验否定；
- 哪些瓶颈最可能限制最终 ALC；
- 优先解决这些问题的理由。

### 10.2 优先级明确的算法改进方案

给出经过证据筛选的少量候选方案。

每个方案必须说明：

- 算法直觉；
- 数学或统计机制；
- 与当前 MLP / smoothed mean 的关键区别；
- 预期改善的预算；
- 潜在负迁移风险；
- 实现复杂度；
- 最小判别实验；
- 接受或放弃的条件。

需要深入解释真正重要的公式与算法选择，而不是只堆砌方法名称。

### 10.3 完成执行的实验代码与记录

在工具能力允许时，应直接实现并运行最有价值的方案。

提供真实的：

- 代码修改；
- 运行命令；
- 权重文件；
- 评估 JSON/CSV；
- 实验配置；
- 必要的图表；
- 运行日志；
- 消融结果；
- 复现说明。

优先在现有 `runs/` 和 `reports/` 中按新实验 ID 保存，避免覆盖历史产物。

### 10.4 最终方法比较

至少比较：

- 原始 empirical mean；
- smoothed mean strength 2；
- 在允许的训练/验证流程内选出的更强统计基线；
- 原始 MLP candidate；
- 本轮改进方法。

结果必须能够回答：

**新方法是否真正改善跨 benchmark 泛化，还是只在某些 benchmark 或某一预算下有局部优势？**

### 10.5 新的可提交候选

若发现有证据支持的改进方法：

- 导出模型权重；
- 构建 ZIP；
- 运行本地提交检查；
- 运行独立进程推理测试；
- 保留原 smoothed mean ZIP 作为 fallback；
- 明确区分验证使用的权重与全公开数据重训权重。

如果全公开数据重训后的精确权重没有独立验证，不得将原验证分数直接归属于这些权重。

### 10.6 下一步决策

输出一个基于真实结果的决策：

- 哪一个模型最适合当前提交；
- 是否已经获得稳定改善的证据；
- 还剩哪些关键不确定性；
- 是否值得开展下一项改进；
- 如果继续，最小且最有信息价值的实验是什么。

如果没有实现稳定胜出，应如实说明目前的最佳方法仍然是简单 baseline，并给出具体失败原因或未解决问题。

---

## 十一、时间与研究优先级

NeurIPS 2026 PAIEC 最终代码和技术报告截止日期为 **2026 年 10 月 30 日 23:59 AoE**，执行时应再次核对官方信息。

考虑剩余时间有限，本轮优化应采用以下策略：

**先解释失败，再改进；先验证简单机制，再增加复杂度；先保留可提交版本，再追求更高性能。**

不要重新进行完整竞赛入门、数据下载或无必要的 baseline 工程建设。

优先将资源用于最可能改变结论的实验。

如果无法在有限时间内完成大规模实验，应优先形成少量有明确机制与可信验证的结果，保留现有可提交版本。

官方最终评估还关注方法的科学严谨性、复现性、实验分析和技术报告质量，因此不能牺牲这些要求换取无法验证的小幅分数提升。

---

## 十二、执行约束与最终验收

必须遵守：

1. **以现有实验为起点。** 不重复已经确认完成的工程准备。
2. **以失效机制为起点进行设计。** 不能直接开始盲目调参。
3. **以未知 benchmark 泛化为核心。** 不能用随机 response-level split 代替。
4. **保持正确的六档预算和 pair-equal Brier ALC。**
5. **尊重已经被查看的数据的证据边界。** 不得将多次调参使用的测试集重新称为独立盲测。
6. **必须与强简单方法比较。** 不允许只相对于明显较弱的 raw empirical mean 宣称成功。
7. **优先分析 B0 与 adaptation 的不同错误来源。**
8. **一次实验尽量只改变一个主要机制。**
9. **不能预设更复杂模型更优。**
10. **必须同时检查平均收益、benchmark 之间的差异及最严重的退化。**
11. **不允许从隐藏反馈反向识别 benchmark 或使用未授权训练数据。**
12. **不能虚构实验、代码执行、评分、权重或提交结果。**

最重要的完成标准是：

> 在严格、可复现的 benchmark-level 验证下，能够说明某种明确的算法机制使预测性能相对强简单 baseline 获得了稳定、具有实际意义的改善；或者通过有判别力的实验，明确为什么当前方法无法做到这一点，并据此确定下一步研究方向。

---

## 十三、现在开始执行

请按照以下顺序推进。

**首先：恢复实验状态。**

阅读上传的实验报告、`plan.md`、现有 MLP 的编码、训练、推理和评估实现，以及相关实验输出。

确认当前已知事实与尚未解决的问题。

**其次：进行以信息增益为导向的诊断。**

从已有 Brier ALC、zero-label prior、per-budget 结果和 per-benchmark 差异出发，形成有优先级的竞争性机制假设。

优先使用已有数据和权重完成低成本诊断。

**然后：设计最有希望的 2–3 项改进实验。**

不要同时开展大量不受约束的尝试。

对每项改进给出明确的机制、预期收益、对照组与失败判据。

**接着：直接实现并执行实验。**

在拥有本地环境访问权限时，完成必要代码修改、训练、benchmark-grouped 验证与结果分析。

每获得新证据，应重新判断原定路线是否仍值得继续，不要机械完成无价值的后续实验。

**最后：选择证据支持最充分的方案。**

若产生稳健改进，完成权重导出、提交打包和复现文档；否则保留当前平滑均值作为 fallback，并整理真实的科学诊断结论。

本轮工作不要止于提出优化方向。

**最终要完成的是从“已发现 MLP 泛化不稳定”到“理解主要限制因素、验证有针对性的改进机制，并形成可靠的参赛模型决策”的完整研究闭环。**