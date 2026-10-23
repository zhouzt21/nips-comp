# R73：原始 ADeLe rubric 的 Luna 标注与质量边界

日期：2026-10-23。原始方案：[plan_8](../doc/plan_8.md)。本页只报告输入、教师标注和复标，不报告尚未完成的 PAIEC 效果。

## 已完成的工作

恢复 R53 的 128 个 distinct items，来自 outer fold 0 的 25 个训练 Benchmark。四个维度保持 AS、MCr、QLl、KNf，512 个原始 prompt 经作者 builder 重建逐条一致。四个互不读取结果的 `gpt-6-luna` 子代理各标注 32 items；另一个没有接触主标注的 Luna 子代理对预先随机抽取的 16 items 独立复标，共 64 条。所有代理仅获准读取隔离的 item/rubric 文件，不获得 subject response、成功率或预测误差。

主标注共 512 条：436 条整数评分、76 条无法判断；108 个 item 的四维完整。98 条包含歧义说明。无法判断保留 `null`，不转成零分。原始响应、理由、item identity、rubric、模型别名和原始 prompt hash 均已保存。两名代理将文件写入了隔离目录内的 `annotations` 子目录，整合时保留原文件，并做逐字节相同的规范路径副本，来源见质量 JSON。

|维度|可评分 / 128|均值（可评分）|0 / 1 / 2 / 3 / 4 / 5 分计数|
|---|---:|---:|---|
|AS：Attention and Scan|109|1.0917|47 / 24 / 20 / 17 / 1 / 0|
|MCr：Identifying Relevant Information|108|1.5926|29 / 21 / 30 / 22 / 5 / 1|
|QLl：Logical Reasoning|109|1.4771|31 / 23 / 30 / 22 / 3 / 0|
|KNf：Knowledge of Formal Sciences|110|0.6273|81 / 8 / 9 / 5 / 7 / 0|

## 复标结果与解释

64 条复标中，48 条双方均给出数值；其余条目涉及无法判断。双方可评分部分的完全一致率 **68.75%（33/48）**，差异超过一级 **2.08%（1/48）**，平均绝对分歧 0.3333。是否可判定的一致率 **93.75%（60/64）**。AS、MCr、QLl 的完全一致率各 58.33%，KNf 为 100%；后者不能单独证明高等级识别能力，主标注的 KNf 主要集中在零分，复标样本也很小。

唯一超过一级的分歧是 item 79 的 MCr：主标注 3、复标 1。题目是稀疏矩阵最小权匹配，分歧来自是否把算法选择过程计入相关信息辨识，不能用该题模型成功率来裁决。item 93 只出现时序信号占位符 `<ts>`：主标注四维为 null，复标尝试按题面类别推断需求。主线程复核题面后保留原始分歧，未改主标签、未根据效果选择标注者。

这是**同模型复标的一致性诊断**，没有人类参考等级，不能表述成原论文的人类 Delphi 验证或证明 rubric 的外部效度。各 Benchmark 的分布已存质量 JSON；分布差异可能反映真实任务需求、输入缺失和标注偏差，当前数据不能将它们分离。

## 输入覆盖与 CPU 学生限制

- 6 个 item 达到旧 R53 的 20,000 字符上限，其中仅 1 个四维完整；这些长文本常是缺少任务指令的论文正文。
- 59/128 个 item 超过项目 MiniLM 实际的 126 个内容 token 窗口（前96＋后30，再加 CLS/SEP，总长最多128），教师可见上下文比学生文本编码更长。对应 52 个 item 四维完整。
- 69 个 item 在该窗口以内，56 个四维完整。原始文本不一定包含图片、时序、交互环境；缺失模态不能靠评分器猜出。
- 这 128 个 item 原本从 query 池取样，不构成每个 pair 的完整 31-label support 标注。教师纯信息诊断与学生完整六预算实验必须分别记录。

因此，接下来需要分别检验“是否可可靠描述需求”“描述是否有跨 Benchmark 预测增量”“学生能否保留增量”。标注完成本身不构成 PAIEC 提分证据。

长度审计修正记录：初版误用通用256窗口；阅读冻结的 `mlp/semantic_features.py::encode_many` 后改按实际126内容token规则重算。旧审计保存在 `*_superseded_window256.json`，不用于结论。原始标注文件 SHA256 不变。

## 原文、代码及本轮改动

基于 [Nature 原文](https://www.nature.com/articles/s41586-026-10303-2) 的 Methods “LLM annotators and inter-rater analysis”“Assessors and metrics”及 Supplementary Table 23；作者代码固定在 [816321f327dfe66910b72b514156aaa296b15f00](https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation/tree/816321f327dfe66910b72b514156aaa296b15f00)。原文使用 GPT-4o-0513、temperature 0 和 1,000 输出 token；预测器使用 18 个需求维度加 UG 的 subject-specific RF，并以 AUROC/ECE 评价。

本轮遵照用户指定使用原四维 rubric 和原任务文本，通过原生 Luna 子代理分片执行。平台只暴露 `gpt-6-luna` 别名，未暴露固定模型快照、温度或作者 API 参数；多个任务共享代理上下文，保留简短判定摘要而非详细思考链。后续 CPU ridge 与 PAIEC Brier 也是明确的方法适配。因此本轮是**原始 rubric 驱动的受控适配实验**，不声称复现原论文的精确模型、完整 assessor 或数值结果。

AnySearch 本轮学术检索重新定位到 Nature 正文；代码垂直检索未返回目标实现，未将其结果用作证据。实际实现依据已固定的作者 checkout，原论文 PDF/补充 PDF 的 SHA256 本轮重新核对一致。检索原始记录在 `research_loop/literature/dual_track_demand/`。

## 可复核文件

- [标注合同](../paiec_baseline/runs/research_loop/R73/annotation_contract.json)：原始输入 hash、分片、独立复标随机种子、方法差异。
- [完整主标注](../paiec_baseline/runs/research_loop/R73/final_annotations.jsonl)：512 条，未根据结果修改。
- [质量汇总](../paiec_baseline/runs/research_loop/R73/quality_summary.json)：逐 rubric/Benchmark 分布、文件 hash、异常计数。
- [逐条复标比较](../paiec_baseline/runs/research_loop/R73/repeat_comparisons.json)、[长度审计](../paiec_baseline/runs/research_loop/R73/item_length_audit.json)。
- [准备脚本](../research_loop/prepare_demand_annotation.py)、[整合与质量脚本](../research_loop/audit_demand_annotations.py)。准备脚本禁止覆盖已冻结输入；质量脚本不会调用标注模型。

教师特征不是 response 真实标签；原始 R53/R68 资产未改写。未调用额外收费 API；原生子代理的实际计费、token 用量没有可用测量，不能报告为零费用。
