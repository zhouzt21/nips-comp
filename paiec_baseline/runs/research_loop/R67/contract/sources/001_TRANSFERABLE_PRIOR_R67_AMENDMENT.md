# R67 pre-outer candidate freeze

R66真实结果：subject分段ΔR25=[−.000051779,+.000936464,+.000009019]，不通过；item分段=[+.000218970,−.003078693,−.002826517]，通过事前2/3来源池、相对常数、合并pair/BM及移除alignment描述性方向条件。没有根据新的外层消融结果选候选。

## Mechanism entry judgment

- 合法可观测信息：item_content、item_features、interactors。不是纯文本，不含benchmark ID或subject描述。来源分组结果支持“item字段足以保留来源内低预算均值信号”，不支持“必须依赖subject能力或非加性交互”。
- 相对R25：新增的无支持标签概率先验，局部证据不足时可能互补；不是首标签可靠性门控。
- 负迁移解释仍为可证伪假设：来源中content/metadata/interactor模式与prevalence关系会改变，同样的item输入也无法表达subject间响应差异。R65的enginemt_qa/benger低估、appworld/healthadminbench高估是反例，合法低维距离未提供稳定识别规则。此解释足以进行一次低成本探索，不能声称已可靠识别未知负迁移。
- 不训练可靠性网络；不扫描mix、预算、阈值，不用目标标签决定是否启用。

## Frozen method and evaluation

唯一方法R67_item_segment。每fold在全部合法source inner_halves按原顺序refit item-only MLP，seed2026/risk mixture/AdamW/网络不变；来源已选epochs=[1,2,2]。B0=p_item(query)，B1=clip(p_item(query)+(y_support0−p_item(support0))/3,0,1)，B3–31使用原R25。同一实际predict调用可实现，仅使用当前query和首支持输入/标签；整组query统计不作为输入。

外层严格复用R62的support/query IDs、variants、counts与原seeds0/1/2：5242pairs、36BM、15726轨迹、6预算。直接回放R62的R25/R56_segment/smoothed2/training_selected_smoothed概率，不重抽样/重生成embedding。index身份和原分数必须重算一致。模型预测只由item字段、允许标签产生，query outcomes仅在评分阶段读取。

报告pair/BM ALC、六预算Brier/预测正确率、三fold、三itemseed、Benchmark-cluster bootstrap10000次seed20261022、alignment与其他贡献、最大退化、相对R56分段及两强均值、训练/推理CPU时间。模型初始化仅一次；没有新独立Benchmark，不作盲测或无偏确认。

预算：3个短refits，CPU4threads，最多30分钟外层；保存每个chunk概率及身份。若超时中断留痕，不换候选或缩减总体后宣称完成。Research Gate保持原样，提交/fallback不变。
