# R51：CNP 与 attention 核心的来源内进入实验

2026-10-22，评分前冻结。授权原文 `doc/复现与组合.md`；来源映射 `literature/reproduction_v4/cnp_anp.md`；继承 `REPRODUCTION_V4_SPEC.md`。来源内 R25 baseline 已逐预算同协议构造并回放 pooled ALC，误差小于 5e-12。下述候选模型尚未评分。

## 可证伪问题与方法边界

将当前支持集合直接映射为 query 条件函数，能否比冻结 R25 更有效利用预算内标签？CNP mean 与 AttentiveCNP learned dot-product cross-attention 分别测试集合汇总和 query-conditioned 汇总。后者是作者 ANP 仓库的 deterministic 路径，未实现 latent/ELBO；其失败不能否定完整 ANP。

作者仓库固定 `aca7e38ea64b718fbd7f311ccae5d51d73447d15`，原文 CNP §2.2–2.3 与 ANP §2.1–3。原始 TensorFlow 代码逐模块映射，独立 NumPy 方程核对；未运行旧 TF1.13 notebook。Bernoulli 输出由 CNP 原文允许。PAIEC 只训练 disjoint query，不对 context 重构计分，是原文 target 含 context 协议的显式适配。

## 数据、训练和选择

原 R25 三个 outer source pool：seed73、最多25 pairs/benchmark，pair数316/260/248。复用 R43 的 source 内 benchmark 二等分，两方向各训练一次；fit/validation/outer 互斥。R25 semantic ridge=4/4/1，metadata=16；强统计先验 alpha=2/2/4 来源于各 outer training 完整pair率，保持原值。公开 benchmark 的历史适应性复用继续披露。

输入为原始 MiniLM384+metadata256。只用 fit episodes 的 support/query 输入拟合逐坐标 mean/std，std 下限0.05；无标签参与此变换，validation/outer 不参与。context 只使用前 n 个合法标签。query重复响应以counts/successes无损聚合。

CNP 与 AttentiveCNP 固定 H=R=64；参数分别102,913与148,097，同层宽但不等参数，不将所有差异归因为 attention。优化 AdamW lr0.001、weight_decay0.0001、gradient norm上限5，batch8 pairs，每更新累加六预算的官方权重 Bernoulli NLL。每pair内部按原 response 数加权，pair间等权。固定60 epochs；seed=5107+100*fold+inner，两模型同seed规则。CPU4threads，一个训练进程，每个fit最多1800秒；达到资源上限记录incomplete，不把未完整训练的模型作为负结果。全部训练loss、每epoch validation Brier、权重均保存。预测器不接收评价标签。

每个模型在一个source pool的两方向 validation 曲线按pair数汇总，选择同一个最佳epoch（0..60）；epoch0为真实初始化模型，不替换为R25。选epoch是development，不提供无偏泛化估计。检查所选epoch的训练损失与曲线，记录训练是否饱和，禁止见outer后扩大超参网格。

## 判别控制与停止规则

所选epoch在每个inner validation做4次支持标签配对置换，保持标签计数/预算/所有query不变。候选进入outer必须在全部三个source pools同时满足：pair-equal ALC优于R25、smoothed2及training-selected smoother；benchmark-equal ALC优于R25；真实配对ALC优于置换平均。模型间单独判断，不选第三种宽度/attention网格救场。

通过后按选定epoch在各outer source全部重训，锁定权重/变换，才进入全部5242pairs、原seeds0/1/2、64query items/全部responses的外层评价，逐benchmark回放R31基线。仍使用原Research Gate；若可能过Gate再核实全query。未通过则停止该核心设计并保留负结果，不执行不必要的外层评分或完整latent ANP扩容。

主指标Brier/ALC；补充固定阈值p>=0.5的response正确率、pair均值偏差与item协方差分解、配对置换差异。这些诊断不能替代主指标或改变晋级门槛。
