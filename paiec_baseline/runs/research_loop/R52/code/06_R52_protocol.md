# R52：内容难度校准与 subject 冷启动先验

2026-10-22，候选评分前冻结。继承 `REPRODUCTION_V4_SPEC.md`；完整用户方案为 `doc/复现与组合.md`；原文/作者代码映射为 `literature/reproduction_v4/amortized_irt.md`。本协议未利用 R52 验证成绩选择参数。

## 问题与证据边界

检验原作者的内容 difficulty 校准，加 PAIEC 合法 subject 特征冷启动，能否改善新 Benchmark 的 B0/B1。Primary 复现 reeval commit `726cfe1ccedaa5576073665d9cda812d20a910c5` 的 fixed-normal expected-NLL 核心；这是发布代码的目标，并非负 log-marginal likelihood。MiniLM/metadata、subject 哈希回归、pair 等权、MAP 稳定化是明确的 PAIEC 适配。不称为 Agent Psychometrics 完整复现，也不把其已见 LLM/scaffold 新组合结论外推为任意 subject 冷启动证据。

## 数据与模型

继承 R51 的 source seed73/cap25、316/260/248 pairs、R43 source 内 Benchmark 二等分，每 pool 双向拟合。两个方向 fit/validation 互斥，均不使用 outer heldout 标签；重复 query responses 用整数 counts/successes 保留。只使用 fit episodes 的 query 标签拟合；support 输入可参与无监督标准化，support 标签仅供评价时的预算内更新。MiniLM384+metadata256，标准化只在 fit 输入拟合，std 下限0.05。subject256 使用既有八个合法字段的稳定哈希；subject key 只标识训练能力样本共享关系，不能进入预测特征。

1. 线性 difficulty 含 bias，零初始化。每个 canonical subject 固定150个 N(0,1) nuisance samples，seed=`5207+100*fold+inner`；同 subject 的所有 item 共享这些 samples。目标为每 pair 内 response 等权的 expected Bernoulli NLL，再对 pair 等权。相对原作者 dense masked mean，这是显式重加权，不宣称与原始矩阵的最优解等价。
2. LBFGS lr0.1、max_iter20、history_size10、strong_wolfe，最多100个 outer iterations。无额外 difficulty 正则。达到梯度 infinity norm≤1e-5，或连续3次 objective 改变绝对值≤1e-9 且梯度≤1e-4，才标记数值收敛；达到100次仍未收敛须标记 optimizer-limit，不能误称充分拟合。保存全部 loss、梯度和迭代次数。内存分块计算同一完整目标，不改变样本权重。
3. 冻结 difficulty，对相同 fit query counts/successes 拟合每个训练 subject 的 theta，Gaussian MAP precision=1、prior mean=0，使用严格凸一维根求解。这里保留 response 计数，不把全成功/全失败 subject 的任意有限 MLE 停止值当成真值。
4. subject256→theta 用 ridge alpha=1；不惩罚截距、各训练 subject 等权。完整保存 subject key、观测数量、theta 与回归误差。该回归替代原文 FLOPs/categorical known-subject 输入，是待验证适配。
5. 推断 B0 使用预测 theta；B1–B31 仅用当前预算内标签做 precision=1、以预测 theta 为中心的一维 MAP 更新。无隐藏标签、无跨 subject 共享揭示标签。

CPU4threads、一个重训练进程；每 fit 最多1800秒，包括优化与后处理。达到时间限制写 incomplete 并停止，不能当成完整负结果。无 validation epoch/hyperparameter 选择；固定优化停止后的最终权重接受来源内检验。保存 checkpoint、normalizer、fit contract、依赖 SHA256 与代码快照。跨方法计算预算不同，因此不以此实验声称方法级算力公平排名。

## 消融、指标与进入条件

保留 difficulty、将 subject ability head 置零得到消融（zero-subject-prior），单独报告 B0/B1 及六预算结果；不额外重训，不在两者中事后挑优替代 primary。Rasch 在固定 support item 集上只依赖成功数和难度分布，label-only permutation 理应不变；保存这一数值检查，不套用 CNP 的配对置换增益门槛。

Primary 进入 outer 低预算验证必须同时满足所有三个 source pools：

- B0 Brier<0.25，且每 pool 严格超过一半 Benchmark 的 B0 Brier<0.25。
- B1 pair-equal Brier 分别优于 R25、smoothed2、training-selected smoother；B1 benchmark-equal Brier 也分别优于三者。
- 拟合完整、数值有限；若 optimizer-limit 且未满足收敛条件，单独记为优化证据不足，不可用来否定模型家族。

报告所有六预算 Brier、ALC、固定 p>=0.5 的 response 分类正确率，以及误差分解。来源内结果为开发证据；三个 source pools 存在历史复用，不合并成独立新测试集。

通过后固定同一训练程序在各 outer source 拟合，才运行原5242 pairs/原seeds0,1,2低预算评价。仅在其独立增益成立后，预声明 B0/B1 使用该 prior、其余预算 R25 的组合并接受完整 Research Gate；不能从 outer 挑混合系数。失败则停止该 primary 及 IrtNet 扩容；不扩大联合训练/subject 回归网格。联合 BCE 实现仅作为机制参考，不在此轮追加训练。
