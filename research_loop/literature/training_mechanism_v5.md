# plan_2 原文与作者代码对应表

核对日期：2026-10-22。检索使用 AnySearch（先发现 academic/code 子域，再按标题检索；标题未命中时补一般检索）。下载的是检索定位的一手 PDF，保存在 `D:/nips-comp/paiec-dataset/training_mechanism_sources/`；`papers.json` 记录 URL、SHA256、字节数，`repositories.json` 记录作者仓库 commit。PDF 经 `%PDF` 检查后由本地 `pdftotext -layout` 转为可定位全文。本文的原文页码/章节和作者源文件是依据，搜索摘要只作发现。

## 依据与迁移边界

| 来源和状态 | 原文定位 / 作者代码 | 本轮使用及不能推出的结论 |
|---|---|---|
| Kumar et al., **Grokking as the Transition from Lazy to Rich Training Dynamics**, ICLR 2024；读取 [arXiv v3](https://arxiv.org/abs/2310.06110v3)，2024-04-11 | 原文 §2–4：初始化附近的线性化、特征学习及 NTK 与任务对齐；本地 `kumar.txt`。AnySearch 两次题名/作者代码检索未定位可验证作者仓库，OpenReview 页面返回浏览器验证，未绕过 | 支持测量训练轨迹和表示变化，不支持 PAIEC 必有迟发恢复。Taylor logit 一阶展开由链式法则实现，并与 Torch JVP 数值对照；CKA 是附加诊断。**代码来源缺口明确保留，不称完整 Grokking 论文复现** |
| Gulrajani & Lopez-Paz, **In Search of Lost Domain Generalization**, ICLR 2021；[原文](https://arxiv.org/abs/2007.01434) | §3 模型选择；[DomainBed](https://github.com/facebookresearch/DomainBed/tree/b93c22a1cfc3b2428398272c1a116c8de1f4139e)，`domainbed/model_selection.py:75` 来源验证、`:130` leave-one-out、`:54` oracle；`algorithms.py:92` ERM | checkpoint 选择是算法合同的一部分。内层来源验证可用于选择，外层只能评价。此处保留 PAIEC 的 AdamW/Brier，不把 DomainBed 图像交叉熵实现照搬 |
| Garnelo et al., **Conditional Neural Processes**, ICML 2018；Kim et al., **Attentive Neural Processes**, ICLR 2019 | CNP §2.3 公式(4)，随机 context 条件训练；[作者仓库](https://github.com/google-deepmind/neural-processes/tree/aca7e38ea64b718fbd7f311ccae5d51d73447d15)。完整原文/代码逐块映射复用 `reproduction_v4/cnp_anp.md` | 候选2只借用不同观察集合的训练机制，保留 R43 网络和损失；**R43 重采样不是 CNP/ANP 架构复现**。不将固定 episode 是失败原因当作已证实事实 |
| Sagawa et al., **Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization**, ICLR 2020；[原文](https://arxiv.org/abs/1911.08731) | §2 群组风险，§3.1–3.2 训练最坏群组拟合不保证测试最坏群组泛化；[group_DRO](https://github.com/kohpangwei/group_DRO/tree/cbbc1c5b06844e46b87e264326b56056d2a437d1)，`loss.py:95 compute_group_avg`、`compute_robust_loss` | 本轮仅固定 λ=.5 的 pair/Benchmark 平均风险，明确区别于作者的动态 adversarial group probabilities 和 minimax。群组权重可能改变有效梯度，但不保证真正 pair-equal 计分收益 |
| Cha et al., **SWAD: Domain Generalization by Seeking Flat Minima**, NeurIPS 2021；[原文](https://arxiv.org/abs/2102.08604) | §3.2、Fig.2、Appendix B.4；[swad](https://github.com/khanrc/swad/tree/252190eb881169e736beccf51af5caaa5c5cf666)，`domainbed/lib/swa_utils.py:9,47`、`domainbed/swad.py:83 update_and_evaluate` | 作者逐 iteration 密集收集并依据 loss valley 排除过拟合。本轮预声明来源稳定区间的3个 epoch；参数算术平均复现 AveragedModel 核心公式，但**不是完整 SWAD**。EMA decay=.9 为本任务固定对照 |
| Kaddour et al., **When Do Flat Minima Optimizers Work?**, NeurIPS 2022；[正式全文](https://proceedings.neurips.cc/paper_files/paper/2022/file/69b5534586d6c035a96b49c86dbeece8-Paper-Conference.pdf) | §4 跨任务比较、§5 失败审计、§6；[WASAM](https://github.com/JeanKaddour/WASAM/tree/fd0937d6b5b3b6703b285954a2f56b36bedaaada)，`swa_utils.py:8 MultipleSWAModels`，Torch AveragedModel 参数更新 | SWA/SAM 存在任务依赖与失败例；训练/测试 loss minimizer 不相关会削弱平均效果。本轮不复现 SAM/WASAM，只使用该负面证据控制平均的进入条件 |
| Woodworth et al., **Kernel and Rich Regimes in Overparametrized Models**, COLT 2020，PMLR125；[正式全文](https://proceedings.mlr.press/v125/woodworth20a/woodworth20a.pdf) | §3–4 的初始化尺度、对角模型和隐式偏置；本地 `woodworth.txt` | 是机制理论依据，不将对角/齐次模型结论当作带 bias、sigmoid、AdamW 的 PAIEC 定理。未定位论文作者运行仓库，本轮不宣称重现原理论实验 |
| Pracher et al., **A Spectral Theory of Grokking: Weight Decay induces Feature Learning**, [arXiv:2609.26679](https://arxiv.org/abs/2609.26679)，v1 2026-09-22，**预印本** | 理论假定齐次网络、平方损失、耦合 L2/梯度流，实验含 SGD；本地 `spectral2026.txt:540,679` 明示代码待论文发表后公开 | 这是对用户“2026年谱理论”的题名定位，不能证明其原意唯一对应。2026-10-22 AnySearch 未找到可验证作者代码更新。PAIEC AdamW 为解耦衰减，且 bias/sigmoid 不满足齐次条件，故只作有条件假设依据，不宣称定理复现或以其要求调大 decay |

## 可复核实现

- R54：`mlp/training_mechanism_training.py` 的 `train` / `selections` / `diagnostic`；普通 MLP 与 R43 网络、优化器和损失直接沿用本项目历史 `train.py` / `reliability_training.py`。所有 epoch 使用相同来源验证 query，内层两方向共同选 epoch。
- R55：`mlp/training_mechanism_data.py` 的 `raw_episodes` / `make_bank` / `resampled_banks`，相同 pair 集、行数和更新次数，仅支持/查询身份与标签抽样变化；fit 标准化固定。
- R56：同文件 `row_weights`，显式 N/(P*n_pair) 和 N/(B*P_b*n_pair)，保持样本和初始化；完整 fit 初始梯度对照在 `gradient_audit`。
- R57/R58：按 `TRAINING_MECHANISM_V5_SPEC.md` 条件审计；没有触发时只记录未执行及诊断，不伪造训练结果。

本次为有原文和作者代码依据的 PAIEC 机制迁移。论文原数据成绩、完整算法配置和原作者 checkpoint 均不是本轮声称的复现对象；来源缺口不能由本地实验结果补成“已复现”。

## 搜索可追溯标识

主要 AnySearch request_id：Kumar `84b58f62-4652-408b-8cb2-cf424e262210`、作者代码补查 `65926f20-60c3-4b7a-b1c7-274559ecc5b8`；DomainBed `735c9dba-38a8-40eb-86dc-c88b124de946` / `b8474ea0-a4b6-46f0-94e8-bb4c5cea8660`；GroupDRO `b0ceb921-42be-488b-8133-5db4a012ffee`；SWAD `1b9df206-fbe2-4683-8533-0bf2546e773e`；Kaddour `a3f5f69d-6d6b-4aee-8c49-7370d2458704`；Woodworth `d5a9f45f-69e2-45f2-aa30-56bbe1117bbe`；谱理论 `17878afc-4109-4cd5-9004-f90ce9f992b5` / 代码补查 `b2b9e6d8-4bfe-4392-b458-9d997e270c45`。嵌套双引号导致的两次 CLI 参数错误未产生搜索证据，后续已用普通字符串重试。
