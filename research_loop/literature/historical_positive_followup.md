# R62–R64 原文与作者实现依据

核对日期：2026-10-22。检索使用 **anysearch skill**；没有以搜索摘要代替论文/代码证据。没有扩大为新的算法调研。

## DomainBed：模型选择属于算法合同

- 原文：Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization*, ICLR 2021（已同行评审）；[arXiv 2007.01434](https://arxiv.org/abs/2007.01434)。本地 `D:/nips-comp/paiec-dataset/training_mechanism_sources/domainbed.txt`，SHA256 `46cc5143a336fd351eaa78d9bc5ea0c413c8ddac0206be76fdfa594c5c266e53`。
- 精确定位：§3、§3.1，正文第3–4页；本地提取文本第161–229行。原文区分训练域验证、留一域验证与目标域oracle选择；留一域选择之后再对全部训练域重拟合。本文使用的是这一选择/隔离原则。
- [作者仓库](https://github.com/facebookresearch/DomainBed)，固定commit `b93c22a1cfc3b2428398272c1a116c8de1f4139e`；[`domainbed/model_selection.py`](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/model_selection.py)，`IIDAccuracySelectionMethod`第75行起、`LeaveOneOutSelectionMethod`第130行起、`OracleSelectionMethod`第54行起。本地文件SHA256 `717e0e78a1a1b5bf97f2ecda080c34f1c016d5d120668c04543f3ec796c8eee4`。
- 本轮AnySearch academic搜索request_id：`5ccca24a-a226-4970-b508-e5c4f5c0f67a`；另通过AnySearch `extract`读取上述固定commit的代码URL，核对目标域不参与来源验证的实现。关键依据来自已读取的原文正文和实际作者源码。
- 迁移边界：没有声称复现DomainBed全部benchmark/架构/指标。PAIEC仍使用原Brier、AdamW及原网络，并使用既有两个Benchmark分组内层验证而非严格leave-one-domain-out。原文不是本项目分数提升的证据。

## 历史候选实现依据

| 对象 | 实际实现/选择依据 | 本轮处理 |
|---|---|---|
| R54普通MLP | `mlp/training_mechanism_training.py`、`training_mechanism_data.py`；R54每fold的`mlp_analysis.json` | 固定来源选择epoch [1,2,1]，全来源refit |
| R56 Benchmark风险 | 同上；R56每fold的`selection.json`，risk=.5pair+.5benchmark | 保留原数据/初始化/风险；固定epoch [1,2,1] |
| R61门控 | `research_loop/run_low_budget_composition_pilot.py`第34行起 | 沿用首支持prior平方误差<.25，仅B1 |
| R43 query | `mlp/run/run_reliability_round.py`、`reliability_training.py`、`evidence_reliability.py` | 已有合法全来源权重复用；epoch [0,0,20] |
| R55重采样 | `mlp/training_mechanism_training.py`、`training_mechanism_data.py`及R55选择文件 | query [0,0,17]、episode [1,20,17]；固定原episode seeds及fit标准化 |
| R42模板 | `mlp/context_tokens.py`、`context_features.py`、`run/run_context_pilot.py` | 项目自定义模板算法；不假托外部论文原算法 |
| R46平均 | `mlp/run/run_complementarity_pilot.py`和R46协议/评分记录 | 固定.5/.5，只替换词法分量，原组合为强对照之一 |

这些候选最初的外部方法依据继续使用已核对的 [training_mechanism_v5.md](training_mechanism_v5.md) 与 [low_budget_prior_composition.md](low_budget_prior_composition.md)。R64不会产生新的MiniLM向量，也不训练新编码器。

## 解释边界

R62–R64都是反复使用的公开开发数据上的探索性证据。训练/模型选择的Benchmark隔离可以阻止直接泄漏，但无法撤销过去的自适应选题偏差。三组来源池互有重叠。三item-split seeds不等于三次随机初始化复现。

R64置换只在clip前保持item等权残差均值，未保证response计数加权均值；clip还可能改变均值。必须报告clip次数，并把它作为诊断对照，不能把分数差直接解释为纯item对应关系的因果效应。核心残差信息检验采用response加权的精确平方损失差分解。
