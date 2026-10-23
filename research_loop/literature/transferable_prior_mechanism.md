# R65–R68 原文与作者代码依据

核对日期：2026-10-22。检索使用AnySearch skill；检索摘要仅发现线索，实验依据为原文和作者源码。

## DomainBed：来源隔离和模型选择

Gulrajani & Lopez-Paz，*In Search of Lost Domain Generalization*，ICLR2021，同行评审。[原论文](https://arxiv.org/abs/2007.01434)。已读取本地全文`D:/nips-comp/paiec-dataset/training_mechanism_sources/domainbed.txt`第161–229行（§3–3.1），SHA256 `46cc5143a336fd351eaa78d9bc5ea0c413c8ddac0206be76fdfa594c5c266e53`；作者代码[model_selection.py](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/model_selection.py)第130行起的LeaveOneOutSelectionMethod，固定commit `b93c22a1cfc3b2428398272c1a116c8de1f4139e`。

原文强调域泛化算法需明确选择规则，来源域验证与目标oracle选择不同。R66沿用PAIEC已有两个Benchmark分组、Brier选epoch，再全来源refit；没有冒称完整复现DomainBed的架构、数据或严格逐域LOO。所有选点和外层公开数据被反复使用的限制均披露。

本轮AnySearch academic查询request_id `ea631ff6-9119-45a8-a124-84d69254e721`、`333cb517-32ce-4b8b-8745-418503c3f154`没有准确返回目标论文，未用于支撑结论。后续`extract`核对原论文入口和固定代码URL，保存`transferable_prior_domainbed_landing.txt`及`transferable_prior_domainbed_code.txt`。尝试HTML版本入口提取失败记录保留；全文证据来自上列可复核本地原文。

## 自定义机制消融的实现归属

R65平方损失分解是逐response的代数恒等式，不是假借外部论文的方法。R66 namespace删除直接基于现有`mlp/model.py`原encoder、`mlp/train.py`原采样器、`mlp/training_mechanism_training.py`原网络/AdamW/选择程序；item-only从原有item namespace的零向量起点累加，不给旧权重遮盖输入。R67是项目原先验残差公式的最小消融组合，不声称外部论文已证明其PAIEC收益。

## ADeLe：忠实输入准备仍不等于效果复现

Zhou et al.，*General scales unlock AI evaluation with explanatory and predictive power*，Nature652:58–67（2026-04-01在线发表，正式同行评审）。[正式全文](https://www.nature.com/articles/s41586-026-10303-2)。本轮再次读取原文p59关于18维rubric需求标注的定义，结合已存补充材料的annotation方法；文献与版本清单见`literature/reproduction_v4/adele_sources.json`。

[作者仓库](https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation/tree/816321f327dfe66910b72b514156aaa296b15f00)，固定commit `816321f327dfe66910b72b514156aaa296b15f00`。本轮实际读取`src/adele/annotation/prompts.py:16,62`、`src/adele/rubrics/catalog.py`以及AS/MCr/QLl/KNf完整rubrics，并对512请求重新调用作者builder，逐条prompt与body完全一致。只验证输入忠实性，没有调用judge、生成标签、训练或效果分数。长度等CPU特征不冒充原rubric标注；没有可验证的本地生成annotator。

## 官方运行时合同

2026-10-22通过AnySearch提取[官方竞赛页面](https://aims.stanford.edu/competition)，快照保存`R68/official_interface.md`及原始extract。Predictor Interface/Test-time Adaptation明确单个input、`labeled`多pair结构、预算限制和匿名benchmark_id。跨pair可见记录受到当次传入证据限制；文档并未给出固定cohort4、稳定item ID或跨调用状态保证。

[官方baseline仓库](https://github.com/aims-foundations/paiec_baseline/tree/cc88a959c1e5e7462f31a9215baba066ce6d6e2b)通过AnySearch发现并核对，git ls-remote确认HEAD，固定下载README与`empirical_mean/model.py`到`R68/official_baseline/`。代码按完整subject属性及benchmark_id过滤显式labeled，无持久状态，不能替代hosted runner实现。该官方版本没有本项目新增的smoothed_mean路径；一次404被记录为入口假设修正。

Codabench17828的AnySearch提取仍失败，记录`transferable_prior_starter_access.txt`。这不是AnySearch整体不可用。未知的是实际peer组成、调用顺序/隔离及可靠item匹配；“多pair证据有文档依据”与“旧cohort4评分等于线上信息”不可混同。
