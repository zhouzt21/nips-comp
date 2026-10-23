# PAIEC runtime evidence contract — R69

## 官方客户端补证（初始条款之后取得）

公开repo确实包含 `tools/streaming_ingestion.py`。已读取并保存原文件与hash（R69/runner_retrieval.json）：文件首部说明coordinator持有证据、每checkpoint重建evaluation workers；`run_streaming`将同一evaluation event的`labeled`传给所有distinct target输入。它加强了“一个事件共享当前证据”的代码依据，但私有coordinator如何抽peer、跨subject item划分、实际部署版本仍未取得。没有真实Hosted运行记录，仍不标`verified_runtime`。

当前机器入口为 `PAIEC_RUNTIME_EVIDENCE_CONTRACT_FINAL.json`，已合并两阶段证据并从保存全文重新生成准确行号。初始 `PAIEC_RUNTIME_EVIDENCE_CONTRACT.json` 与 `PAIEC_RUNTIME_EVIDENCE_CONTRACT_CLIENT_ADDENDUM.json` 被不同实验阶段冻结，保留不变。下表为初始审计快照，其行号以FINAL.json为准；cross_call_state须结合新增官方客户端证据阅读。

核查日期：2026-10-22T16:11:02.418721+00:00。官方repo `cc88a959c1e5e7462f31a9215baba066ce6d6e2b`；原文副本与hash见R69/retrieval.json。

## 条款状态

|条款|状态|证据/边界|
|---|---|---|
|multi_pair_labeled|verified_official|labeled contains entries across sampled subject–benchmark pairs; official_interface.md:131|
|visible_subject_fields|verified_official|Exactly eight plain-string subject fields, missing empty string; official_interface.md:95|
|visible_item_fields|verified_official|item_content, item_features, interactors, anonymous stable benchmark_id; item_id absent; official_interface.md:108|
|budget_zero|verified_official|labeled is empty at B0; official_interface.md:120|
|budget_nested|verified_official|0,1,3,7,15,31 per pair; nested trajectory; same evaluation items; official_interface.md:120|
|acquisition_separation|verified_official|50/50 items per pair; all repeated responses stay with item; evaluation outcomes never revealed; official_interface.md:118|
|exact_input_prediction_reuse|verified_official|Identical subject/item input reuses one probability for repeats; official_interface.md:87|
|visible_subject_match|verified_official|Official empirical mean matches complete subject dict and benchmark_id; official_baseline/empirical_mean/model.py:26|
|actual_hosted_peer_count|unknown|No executable hosted runner or actual evidence sample obtained; 无可核验runner条款|
|hosted_peer_selection|unknown|No guaranteed same-benchmark peer cohort or selection distribution; 无可核验runner条款|
|own_plus_peer_per_call|inferred_not_guaranteed|Multi-pair schema permits it, but actual calls and timing not observed; 无可核验runner条款|
|target_query_peer_acquisition_overlap|unknown|Per-pair split is explicit; cross-subject item overlap rules are not; 无可核验runner条款|
|cross_call_state|unknown|Do not assume call order or persistent state, including across budgets; 无可核验runner条款|
|same_text_means_same_item|unsupported|Identical text can differ in features/interactors; internal item_id unavailable; 无可核验runner条款|
|peer_labels_without_visibility|unsupported|Never infer unseen labels from historical IDs or hidden evaluation outcomes; 无可核验runner条款|

## 保守实验合同

仅使用当前调用的允许标签；B0空；不依赖历史调用、缓存标签或匿名ID解码。同题匹配使用 benchmark_id/item_content/item_features/interactors 四字段完全一致；文本同而metadata不同不合并。完整subject字段相同不能借内部subject_id当作另一个peer。多对一的输入映射单独报告，无法从可见输入确认的匹配不赋予“同一内部item”语义。

R72将运行0/1/3peer的条件模拟，peer由合法可见配置的确定性排序选择，未保证Hosted具有这种组成。一个peer的评价响应永不揭示；它的acquisition item可以在另一subject的query中出现，这是明确标注的跨subject重叠假设。自己的query响应不得进入任何拟合或校准。

## 有界检索及证据缺口

AnySearch检索与官方页面提取成功；官方GitHub完整tree和原baseline已取。AnySearch Codabench提取失败，公共API TLS EOF；未绕过认证或TLS。没有真实Hosted运行记录，故没有任何 verified_runtime 条款。未把unknown写成禁止共享。

## 给组织者的问题草稿（未发送）

1. predict的labeled是否同时含同一benchmark中所有抽中subject的当前预算标签？可否提供脱敏输入样例/runner版本？
2. 同benchmark subject数量/选取方式是什么？是否可能没有peer？各budget是否对全部pair同步截断？
3. acquisition/evaluation划分是每subject独立还是benchmark共享？某subject acquisition item是否允许出现在另一subject evaluation？
4. 对item_content/features/interactors完全相同但内部item_id不同的情况，应该如何理解？是否会提供稳定item标识？
5. predict调用顺序和进程状态是否隔离？是否保证每调用提供完整的当前允许证据？

官方入口：https://aims.stanford.edu/competition/；repo：https://github.com/aims-foundations/paiec_baseline 。
