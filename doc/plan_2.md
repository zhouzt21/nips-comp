候选 1：完整训练轨迹与 OOD checkpoint 审计
第一优先 Pilot
Observed Failure → Mechanism。 普通 MLP 最佳 epoch 偏早，现有日志不足以判断后期是否仍有泛化改善。可能是过拟合，也可能是早停截断了其他解。

Literature Evidence。 Kumar 等说明延迟泛化可以在特定条件下出现；DomainBed 说明模型选择标准是跨域泛化实验中不可忽略的一环。两篇文献支持的是“需要测量和控制”，不是“继续训练一定会提高分数”。

proceedings.iclr.cc
+1




最小干预。 在不改变网络和 AdamW 配置的情况下，补全日志并完成固定长度的短训练轨迹。保存每个 epoch 的 checkpoint；原 Early Stopping 仍作为对照选择规则，不能用事后发现的最佳外层 checkpoint 宣称无偏收益。

判别实验。 比较原 patience=5、同训练步数下的最佳来源内选择、以及固定 20/30 epochs 轨迹中的候选。所有 checkpoint 用相同验证 Benchmark 和相同 query 集评估。

快速否定条件。 如果两个或更多独立的来源分组中，后期 Train Brier 下降但 held-out 来源 Brier 持续恶化，没有稳定恢复，停止研究延迟泛化；也不再投入长时间 Grokking 搜索。

收益判断。 这一项最可能避免错误训练决策，但本身未必改善 R25 总 ALC。普通 MLP 的直接收益主要出现在 B0/B1；R43 的轨迹则可检验 B3–B31。
候选 2：让条件模型真正经历不同的支持集
Observed Failure → Mechanism。 R43 学习修正通常不如原始 R25；R51 在固定 episode 上显著拟合，却没有可靠利用 support 标签与 query item 的关系。

Literature Evidence。 CNP/ANP 的核心是对不同观察集合形成条件预测函数；R51 的固定训练任务数量不足，是与既有现象吻合、但尚未被单独证实的候选解释。

Proceedings of Machine Learning Research
+1




最小干预。 优先选择计算更便宜的 R43 小网络。保持模型、特征、公式 \(p=\operatorname{clip}(b+g\,d,0,1)\)、AdamW、Brier loss 及总更新数不变，只在各合法来源 pair 内为训练过程重新抽取 support/query 和预算组合。训练中的标准化仍只能从 fit 数据拟合。

判别实验。 固定 episode 与每周期重采样 episode 两组对照；比较不同预算的条件化 Brier、query-specific 与 episode-only 模型差值，以及正确支持关系被打乱后的性能变化。

快速否定条件。 如果重采样后没有稳定超过 epoch 0 的 R25 回退，或者 query-specific 模型无法优于 episode-only 控制组，则停止该学习修正方向。

预期收益。 只可能改善 R43 实际修正的 B3/B7/B15/B31，不应声称改变 B0/B1。考虑到 R25 的大部分已知增益集中于较高预算，这一机制与剩余优化位置的匹配度较高，但成功概率仍不确定。
候选 3：对齐 Benchmark 而非仅对齐响应行
Observed Failure → Mechanism。 普通 MLP 的训练仍可能偏向具有大量可采样响应或更容易学习的来源 Benchmark，从而在未见 Benchmark 上形成错误先验。

Literature Evidence。 DomainBed 和 GroupDRO 提醒我们重视跨域 ERM 与不同群组的有效权重，同时也提供了“复杂训练目标不一定优于 ERM”的负面证据。

OpenReview
+1




最小干预。 在完全相同的抽样数据、初始化、更新步数下，对比当前 pair-weighted risk 与预先固定的一种 Benchmark-balanced risk。例如：
\[
L_\lambda
=
(1-\lambda)L_{\mathrm{pair}}
+
\lambda L_{\mathrm{benchmark}}
\]

其中 \(L_{\mathrm{benchmark}}\) 是先在每个来源 Benchmark 内平均 pair loss、再在 Benchmark 之间平均；pilot 可预先固定 \(\lambda=0.5\)，不进行大量系数搜索。

判别实验。 同时报告目标 pair-equal ALC、benchmark-equal Brier、各来源的增减，以及样本加权导致的有效梯度变化。

快速否定条件。 若仅改善 benchmark-equal 指标却恶化真正计分的 pair-equal ALC，或与既有 pair-balanced 负实验类似没有稳定收益，则停止。

关键限制。 R22 的扩大训练池和早前 pair-balanced 采样已经提供负面信息；Benchmark-balanced risk 与它们并非完全相同，但不能假装数据重加权是未经尝试的新方向。
候选 4：EMA、SWA 与短窗口概率平均
Observed Failure → Mechanism。 单一 checkpoint 可能对小规模来源验证过于敏感，而邻近 checkpoint 的预测误差可能互补。

Literature Evidence。 SWAD 在多个 Domain Generalization Benchmark 上取得收益，Kaddour 等的系统比较则说明这类策略并非普遍有效。

NeurIPS Proceedings
+1




最小干预。 不增加训练步数，保留相邻 checkpoint。对比单一最佳权重、EMA 权重、短窗口 SWA 权重及相同窗口的概率平均。

判别实验。 只有候选 1 显示存在一段来源验证表现相对稳定的 checkpoint 区间时才启用。不能从出现严重过拟合的全部后期 epoch 随意求平均。

快速否定条件。 若平均后仅在来源内更平滑、跨 Benchmark Brier 没有稳定收益，或明显差于最优单一 checkpoint，停止。

资源判断。 训练端的额外成本很小，但多 checkpoint 推断、转换 NumPy 权重与完整预算评分会增加评估成本。若线上只能承载单一模型，参数平均比多模型 ensemble 更容易部署。
候选 5：控制初始化、学习率和隐式正则化
Observed Failure → Mechanism。 如果候选 1 发现模型长期接近初始化附近的线性化函数，而后期出现与来源外 Brier 相关的表示重构，那么训练起始尺度或更新速度可能限制了有用特征的形成。

Literature Evidence。 Kumar、Woodworth 和 2026 年谱理论提供了机制依据，但应用到 PAIEC 仍需要额外假设。

proceedings.iclr.cc
+2




最小干预。 仅在有明确诊断迹象后，做小型预注册对照：默认初始化与一个明确的输出层尺度条件，保持相同输入、来源 Benchmark、总更新步数及 checkpoint 规则；随后再单独考虑 weight decay。

快速否定条件。 若在匹配计算预算下没有改善来源外 Brier，或者改变初始化只改变训练速度、不改变最终概率质量，则不继续扩大参数网格。

为什么排在最后？ 因为当前主要异常已经有更直接、更便宜的解释。并且调大正则化可能进一步削弱小网络的表达或使训练欠拟合，不能仅凭 Grokking 文献认为它有利。