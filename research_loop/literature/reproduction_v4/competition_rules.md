# 本轮竞赛规则来源

核对日期：2026-10-22。检索使用 AnySearch。查询 `site:aims.stanford.edu/competition PAIEC 2026 pre-trained models inference API budgets Brier`，request_id `f8793d0f-2664-42b7-82fd-001783ccfb08`。据结果继续 extract 官方页面 [The Predictive AI Evaluation Competition at NeurIPS 2026](https://aimslab.stanford.edu/competition)，完整返回保存在 `D:/nips-comp/paiec-dataset/reproduction_sources/rules/official_competition_anysearch.md`。

核对的原文位置：Predictor Interface、Test-time Adaptation、Evaluation / Metric、Participation / Training data & Runtime、Technical / Pretrained models & Compute limits。

- 六档预算为0/1/3/7/15/31，原子split为distinct item，所有重复response留在同一侧。每个pair有相同权重；Brier ALC权重为.1/.2/.2/.2/.2/.1。此次继承这些协议，仅保留明确披露的64评价item公开代理上限。
- 当前predict接口合法subject字段为8个已有配置字段，item字段为item_content/item_features/interactors/benchmark_id。匿名ID只作证据分组，不作新模型特征。
- 预训练模型可以使用并须披露。比赛专门训练/微调数据限制于benchmark inventory；此次只使用已有measurement-db冻结revision，作者原始实验数据不用于PAIEC训练。
- 评分阶段网络限列明提供商的inference APIs，禁止search/browsing/grounding；模型随包或models.txt预取，评分时不可任意下载。此次使用本地已有MiniLM和CPU模型，不调用付费推理服务。
- 页面注明final code/report deadline为2026-10-30 23:59 AoE，每run 8小时（含setup）。本轮源内CPU训练采用更小的独立预算，不能把原文允许资源上限解释为当前机器可用算力。

该页面用于规则核对；方法性能证据来自本地受控实验。没有正式提交或访问隐藏反馈。
