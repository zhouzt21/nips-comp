# R52: Amortized IRT 原文、代码与 PAIEC 适配边界

核对日期：2026-10-22。用户原始执行要求保存在 `doc/复现与组合.md`，本文不改写该原文。检索使用 AnySearch；论文 PDF 从检索确认的官方入口下载。这里只记录来源、实现合同和小型数值检查，不把尚未运行的模型写成实验结果。

## 1. 已固定的原文与作者代码

| 来源 | 身份与固定版本 | 读取位置 |
|---|---|---|
| Sang T. Truong, Yuheng Tu, Percy Liang, Bo Li, Sanmi Koyejo, *Reliable and Efficient Amortized Model-based Evaluation* | *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267:60238–60265, 2025 | [正式 proceedings](https://proceedings.mlr.press/v267/truong25c.html)，PDF §3、§4.1、§5.1–5.2，论文页 4、6–7 |
| reeval 作者代码 | `aims-foundations/reeval`，旧 `sangttruong/reeval` 重定向到此；commit `726cfe1ccedaa5576073665d9cda812d20a910c5`，2026-05-16 | `calibration/calibration.ipynb` 第 15 个 cell（零起始 index）；`cat/adap_test.py:10`；README 的符号说明 |
| Chris Ge, Daria Kryvosheieva, Daniel Fried, Uzay Girit, Kaivalya Hariharan, *Agent Psychometrics: Task-Level Performance Prediction in Agentic Coding Benchmarks* | [arXiv:2604.00594v2](https://arxiv.org/abs/2604.00594v2)，2026-07-20；首页写明 COLM 2026 | PDF §2.1.2、§3.2.1–3.2.2、§5.3，尤其论文页 5 |
| Agent Psychometrics 作者代码 | `dariakryvosheieva/agent-psychometrics`，commit `1c34d34a0ee82c92b1c63deb834495c5440bd362`，2026-08-30 | `experiment_new_benchmarks/pipeline.py`、`experiment_new_agents/README.md`、`experiment_new_tasks/feature_irt.py`、`experiment_adaptive_testing/cat_simulation.py:99` |

本机固定副本：`D:/nips-comp/paiec-dataset/reproduction_sources/amortized_irt/`。原文文件为 `truong25c.pdf` 与 `agent_psychometrics_2604.00594v2.pdf`；同名 TXT 是 `pdftotext -layout` 的解析副本。两个代码库均 detached 到上述 commit。Agent Psychometrics 使用 sparse checkout，仅检出方法代码、README 和 LICENSE；没有运行其数据下载、训练或特征抽取脚本。

来源 SHA256 和检索 request ID 见本目录 `amortized_irt_sources.json`。AnySearch 的两篇 arXiv HTML 抽取均失败，失败记录保留；因此读取官方 PDF。OpenReview 页 `3jcfZwndGN` 返回浏览器验证页，未绕过，不能据此宣称另行核实正式 proceedings；COLM 2026 状态目前由作者 v2 PDF 首页支持。

## 2. Truong 原文与发布代码的具体区别

原文 Rasch 定义为 `p(y=1)=sigmoid(theta - difficulty)`。作者代码为兼容 R 的 `mirt` 约定写为 `sigmoid(theta + z)`，其中 `z` 是 easiness。我们的实现统一使用 difficulty，因此 `difficulty=-z`；这不是改变预测模型。

§4.1 在边缘/EM 语言下描述校准，§3 使用标准正态能力分布固定位置和尺度。论文 §5.2 使用 Llama-3.1-8B-Instruct 的 4096 维 embedding 和线性 difficulty predictor。§5.1 的新模型能力预测使用训练 FLOPs 的对数线性映射，而非任意 subject 文本哈希。

发布 notebook 的实际代码需要单独追溯。在零起始第 15 个 cell，它执行：

```python
thetas_nuisance = torch.randn(150, n_test_takers, device=device)
z = features_batch @ w + b
probs = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
loss = -(Bernoulli(probs=probs).log_prob(data_batch) * train_idtor_batch).mean()
```

这里省略的是缺少 embedding 时的 `z_free` 分支；PAIEC 输入已为每个 item 提供有限数值特征，因而不需要该分支。每个 subject 的 150 个 ability draws 在同一次拟合中固定，并由该 subject 的所有 item 共用；不能每条 response 都重新独立抽一个 subject ability。

这段代码优化 **expected negative log likelihood**：

`L = mean_{a,s,j}[mask_sj * BCEWithLogits(theta_as - difficulty_j, y_sj)]`。

它不是 `-log mean_theta p(y|theta,difficulty)`，也没有在每次 LBFGS 更新前重新估计 ability posterior 的 E step。两种目标不能混称为同一个边缘似然。本次 primary 候选忠实复现作者发布代码的 expected-NLL 数值核心；报告应明确此论文/代码差异。

拟合 difficulty 后，作者冻结它，用同一训练 mask 对每个已见 subject 的自由 `theta` 最小化 Bernoulli NLL。`cat/adap_test.py` 的 `estimate_theta()` 也只使用已询问标签更新该 theta。原实现优化器为 LBFGS（lr 0.1、max_iter 20、history_size 10、strong_wolfe，外层最多 100 次）。本项目若使用 AdamW 训练线性头，应在运行配置中列为优化适配，不能声称逐设置复现。

## 3. Agent Psychometrics 修正了哪些推荐前提

§3.2.1 采用 **先 SVI 拟合标准 IRT，再对冻结 difficulty 做 ridge regression** 的两阶段训练，主结果不是端到端联合 BCE。作者也提供 `experiment_new_tasks/feature_irt.py` 的 joint-likelihood 对照；其 mean(theta)^2 正则用于可辨识性。

§3.2.2 的 agent ability 为 **LLM 类别参数 + scaffold 类别参数**。新 agent 是已见 LLM 与已见 scaffold 的新组合。论文和 `experiment_new_agents/README.md` 都明确要求两个组件各自出现在训练集合；它不能直接解决任意未见 LLM/scaffold 的冷启动。因此 PAIEC 的 subject256→ability 映射必须作为新适配验证，不能引用该论文当作已证明可迁移的结论。

其任务特征可含 gold solution、tests、repository context。PAIEC 当前方法只使用比赛合法可见 item 输入和 subject 字段；不能为了贴近论文而引入隐藏答案、完整任务执行轨迹或未获准共享标签。

作者 online `estimate_theta_mle()` 名称虽为 MLE，实际目标是 Gaussian MAP；默认 prior sigma=3，且约束 theta 在 [-6,6]。新 agent 实验还用跨 heldout agents 的 theta residual RMSE 设置 informative prior sigma。PAIEC 不复制这种跨外层 evaluation 的 prior 拟合：固定 precision=1，且无额外 [-6,6] 截断。该方差选择属于预声明适配，依据是标准正态定标约定，并非原论文拟合得到的超参数。

## 4. 本项目两条可区分的实现

### Primary: 发布代码的校准核心 + PAIEC 冷启动适配

1. 使用与 CNP 相同的 source seed73/cap25 episodes；只从当前 fit benchmarks 的 query responses 聚合 count/successes。固定 item 特征为 MiniLM384 + metadata256，fit-only 标准化。不能用 heldout benchmark 标签拟合 scaler、difficulty、ability 或回归系数。
2. 按 canonical 合法 subject 字段建立训练索引；索引只用来共享 nuisance draws，不能作为 predictor 特征。一次固定 150 个 N(0,1) samples/subject。
3. `AmortizedIRT(difficulty_bias=True)` 的线性 difficulty 头优化 `reeval_expected_nll_counts()`。固定 nuisance ability 给 difficulty 的位置和尺度提供锚点。
4. 冻结 difficulty，用同一 fit response 池对每个训练 subject 估 ability。`fit_training_abilities_counts(..., prior_precision=0)` 是严格 MLE 根；全 0/全 1 subject 的 MLE 不有限，函数返回 NaN+mask，避免把任意优化停止值当成真值。主协议选择 `prior_precision=1`，围绕 0 做 Gaussian MAP 防止分离发散；这是清楚标注的稳定化适配。
5. 仅对这些训练 subject，用稳定哈希 subject256 回归训练 ability；ridge 系数只能预固定或在 inner fit/development 选择。该映射适配替换了原文的自由已见 subject 参数/FLOPs，原论文没有证实这个哈希特征能泛化。
6. 训练完成后，再通过下述支持 MAP 更新作 B1–B31 推断。B0 不进行在线更新。

对于 repeats，按 `(subject,item)` 保存 count 与 success 是 lossless Bernoulli 聚合。`success*softplus(-logit)+(count-success)*softplus(logit)` 保留每条 response 的贡献；不能 majority vote，也不能先平均后不加 count 权重。

原 notebook 对完整 masked 矩阵取 mean。核心函数允许 `grid_size=subjects*items` 精确复现该分母；默认对实际 response 数取 mean，是固定 mask 下与原目标成比例的归一化适配。加正则、修改采样或优化器时，这种分母缩放对最终参数不一定等价，必须记录。

### Control: 联合训练的 feature-conditioned Rasch

`AmortizedIRT(difficulty_bias=False)` 用 source B0 BCE 联合训练 subject ability 与 item difficulty。difficulty 没有截距，且输入 item 使用 fit-only 中心化，使 fit 集平均 difficulty 固定为 0；ability 的截距承担全局成功率偏置。这排除了两个任意截距一起平移的 gauge。相关特征产生的系数不唯一仍需正则，不能把每个哈希维度系数解读成心理能力。

该 control 更接近一个可分的线性 logistic predictor，并非 reeval 原发布校准流程。它不实现多维能力、discrimination、MoE 或 Agent Psychometrics 的两阶段 SVI/ridge 主模型。

## 5. 推断接口与机制

```python
model = AmortizedIRT(item_dim=640, subject_dim=256, difficulty_bias=True)
model.eval()
logits = model.predict_logits(context_x, context_y, target_x, subject)
probabilities = model(context_x, context_y, target_x, subject)
```

形状分别为 `[B,N,640]`, `[B,N]`, `[B,Q,640]`, `[B,256]`，输出 `[B,Q]`；N 为 0–31，context label 必须是 0/1。B0 输出 `sigmoid(theta0(subject)-difficulty(query))`。非空 context 只在 eval 模式调用，不能误以为当前支持 MAP 在参与元训练反传。

在线求解 `sum_support BCEWithLogits(theta-difficulty_i,y_i) + (theta-theta0)^2/2`。严格凸的一维目标用有理论边界的二分法求解；以 delta=theta-theta0 表示，根必在 `[-N/precision,N/precision]` 内。返回当前预算内标签决定的 theta，query labels 从不进入接口。

**解释性边界：**固定支持 item 集合时，Rasch 的 ability 更新只依赖成功总数及支持难度分布，对 support-label permutation 不敏感；这是 Rasch 的充分统计量性质。因此 CNP 的“打乱标签配对必须损失性能”门槛不能照搬到 IRT。当前实现只整体平移 query logits，没有从具体支持—query 文本相似性学习不同的局部修正。这是它与 CNP/ANP 的机制差异，也是可能无法捕获 R25 局部收益的原因。

## 6. 已执行的核心检查与未完成项

实现：`paiec_baseline/mlp/amortized_irt.py`；检查：`tests/test_amortized_irt.py`。

执行命令：

```powershell
& D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1 -m unittest tests.test_amortized_irt -v
```

2026-10-22 首次结果：8 tests passed，unittest 报告 0.160 秒。补充 Rasch 的 label-only permutation 不变断言后再运行，同样 8 项通过（0.089 秒）。检查覆盖：与固定源码 expected Bernoulli NLL 的 loss/gradient 一致；raw repeats 与 counts/successes 一致；B0 gauge 与训练梯度；在线 MAP 梯度为零；全 0/1 和无观测 subject 的 MLE/MAP 边界；合法预算及 binary 检查；query 集改变不影响已有 query；support 成对重排不变和 label-only permutation 不变；保存重载一致和大有限输入概率有限。

这些是核心数值证据，不是 PAIEC 成绩。外层 B0/B1、六预算 Brier/ALC、0.5/smoothed2/R25 对比、benchmark 多数方向、inner selection、cold-start 与组合门槛由主运行 harness 执行并另行记入 R52 产物。任何外层不增益均不得写为“原文方法已证明在 PAIEC 有效”，也不得直接融合到 R25。
