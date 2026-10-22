# 公开二元标签的语义边界

2026-10-22 实际核对本地 `benchmarks.parquet`，版本 `54074453da7b9911be0f48e4c560be547bc84d21`：61个目录中54个 `higher_is_better`、4个 `lower_is_better`、3个 `unordered`。完整记录：`runs/research_loop/R15/label_semantics_inventory.json`。

例如 alignment_faking 的1是分类器检测到alignment-faking reasoning；braveguard的1是guard将轨迹判为harmful；emoji_attack的1是成功jailbreak。它们不能自动解释为同一种“模型回答正确”或一维通用能力。即使方向同为higher，也不保证任务测量同一种能力。

官网既使用success/correctness表述，也明确要求预测response1；官方准备代码按binary+item筛选，没有方向转换。尚未核实hosted ingestion是否另有映射。因此保留原始标签、全部历史评测对象和主门槛，不擅自翻转0/1，也不把非higher任务删除后宣称提升。

历史Brier计算仍对应合法的原始二元outcome预测；此前报告中的跨任务“success/ability”表述应按本说明收窄，不能当作统一能力测量。隐藏测试的具体标签转换仍是协议证据缺口。

## 直接判别结果 R16

仅使用higher方向的**外层训练benchmark**重新拟合R15相对能力模型，其他条件不变，外层全部737pairs继续按原始标签评分：

| 方法 | cohort4三种子ALC |
|---|---:|
| 简单peer | 0.204959 |
| 原能力修正 | 0.204983 |
| 仅higher训练的能力修正 | 0.204982 |

简单方向过滤没有恢复该能力模型。不能将失败全部归于事件标签混合。

在625个higher方向的外层pair中，旧MLP的B0仍为0.274379，差于0.25；其ALC0.217277也差于同条件平滑均值0.210323。R7在这个子集中有较小增益（三种子0.209073 vs0.209700）。这些都是事后分层诊断，不是新的独立测试集。

来源：[竞赛规则](https://aims.stanford.edu/competition)、[公开数据README](https://huggingface.co/datasets/aims-foundations/measurement-db/blob/main/README.md)、固定revision的本地metadata。AnySearch提取原文保存在 `research_loop/literature/raw/`。不使用metadata反查隐藏benchmark。
