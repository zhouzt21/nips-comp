# R59–R61 原文与代码依据

核对日期2026-10-22。本轮是冻结PAIEC预测的机制诊断与一个无训练选择器的迁移试验，不是重跑下面论文的全部原始实验。

## 检索与原入口

使用anysearch skill，已确认安装脚本与维护仓库副本SHA256均为`f029d4841d856085e27ebd226d36714bfc45898ee1f60f210d04195fca868a7d`。先发现academic子域，academic.search两次标题检索未命中目标；补一般标题检索后定位原文，再用extract核对NeurIPS和作者仓库页面。未使用其他搜索引擎。

- Ovadia一般检索request_id `72c3667d-8924-4e55-9e4e-23f4df7247a3`。
- Gneiting/Raftery一般检索request_id `b3188e87-ca56-49df-b341-e1376a6e3835`。
- academic未命中request_id `299e2dc7-d002-40c1-8045-974ef07feff1`、`51bae3fc-9676-4ee8-be9d-5e2c7c9ff919`；未以无关结果作为证据。
- 首次双domain参数被PowerShell包装为空格而失败，改为单academic发现；带内嵌引号的两次命令参数错误后改为普通标题查询。属于检索调用异常，不是实验结果。

原始PDF均通过`%PDF`检查、SHA256记录，再由本机`D:/texlive/2025/bin/windows/pdftotext.exe -layout`读取全文；现有迁移Python未安装PDF包，使用已有TeX Live工具，无新依赖安装。副本和manifest位于 `D:/nips-comp/paiec-dataset/low_budget_prior_sources/`，下载脚本为同目录研究代码中的 `fetch_low_budget_sources.py`。

## 理论依据与精确边界

1. Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*，JASA正式论文，[作者全文](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)。第3节、期刊p.363 Example 1给出quadratic/Brier scoring rule，采用分数越大越好的负平方误差约定；本项目采用损失越小越好。它支持以概率平方误差检验预测质量，而非只看分类正确率。

2. Ovadia et al. (2019), *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*，NeurIPS正式论文，[proceedings](https://papers.nips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift)、[原文](https://papers.nips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf)。主文p.4、§3 Eq.(1)给出按类别数归一化的Brier；同页定义置信度选择评测；§4/Fig.2–3讨论shift下校准和置信度。论文支持单独检查来源外可靠性，未证明本任务的confidence、disagreement或一个支持标签足以识别优劣。

本任务的四项恒等式是对每pair有限response分布直接展开二次损失：`Var(Y)+(mean(P)-mean(Y))²+Var(P)-2Cov(P,Y)`；它**不是**论文的分箱uncertainty-resolution-reliability估计，也不能把其中的预测方差叫作跨训练样本的估计方差。以重复response计数计算，逐pair等式独立数值核验。

## 作者代码与评分核对

原论文p.2脚注给出 [google-research/uq_benchmark_2019](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019)。本机固定到commit `5b09c22d73a9d35eb6c5d2a99b95677a45053466`，保存README、metrics_lib.py、calibration_lib.py及SHA256；这是核对时仓库版本，非声称2019原提交未变。

- `metrics_lib.py:189` 的 `compute_accuracies_at_confidences` 从预测概率求置信度，再按阈值筛选评测。本轮仅借用“合法信号需要以评分核验”的思路，不复制阈值扫描。
- `metrics_lib.py:218` 的 `brier_scores`实际返回`sum(p²)-2p_y`，没有`+1`且不除类别数。二分类PAIEC须用`(author_raw+1)/2`才等于`(p-y)²`。直接复制原函数原值会错标绝对Brier与ALC。
- `metrics_lib.py:267` 的 `brier_decomposition`使用预测类别分区并说明估计误差；本轮不以它替代用户指定的精确四项恒等式。
- `audit_low_budget_composition.py`只提取并执行原文件的NumPy `brier_scores`函数，在实际预测/成功失败计数上核对上述归一化；不加载TensorFlow，也不运行原仓库的训练/下载脚本。

## R60的来源映射

新候选由R59诊断提出，**不是文献中已有且保证有效的算法**。B1选择条件`(prior(s1)-y1)²<.25`是将当前允许的一条response用于与恒定.5的proper-loss比较；.25为代数常数，无阈值搜索。它仍存在单样本误判风险，R61实际数据才是有效性证据。

更多文献、温度校准、deep ensembles、网络重训、固定mix扫描均未进入本轮。
