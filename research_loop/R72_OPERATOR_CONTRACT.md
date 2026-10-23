# R72 输入覆盖审计后的冻结算子合同

时间：2026-10-23，读取覆盖率后、任何R72预测分数之前。与初始 `LEXICAL_PEER_EVIDENCE_SPEC.md` 联用。

## 观察与选择

已完成5242pairs×3seeds输入覆盖审计，不用query outcomes选择cohort。3peer情景平均可见2.9611个peer；同输入覆盖率（pair内response加权后pair平均）B1/3/7/15/31约0.3377%/0.9730%/2.2550%/4.7811%/9.6104%。这支持同时测量稀疏同输入残差和更广泛的异输入共享方向。不能只测命中样本然后宣称全体收益。

## 可见证据与情景

- 以完整subject8字段哈希排序，每4个配置为一个固定cohort。重复可见配置的所有相关subject只做own-only；不利用内部ID区分peer。预算独立，只给每个成员前n个acquisition标签；B0空。自身query与support内部item无交叠，peer的query标签永不输入。
- 0peer无共享；1peer取冻结cohort中第一个其他成员；3peer取所有其他成员，不强行补满。cohort大小最多4，最多4×(31+64)=380个distinct subject/item input预算设计；正式Hosted组成仍未知。
- 匹配完整benchmark/content/features/interactors四字段。每peer当前前缀中重复的可见key不参与same-input残差；不同metadata/interactors不匹配。离线发现不同内部item对应相同可见key单列报告，不声称它们必为同一道潜在任务；模型只声称同可见输入。此保守相等定义依赖正式输入字符串序列化，需真实runner确认。

## 固定算法，无新训练或outer调参

1. **R72_same1/3**：每peer在当前允许前缀内中心化 `r=y−mean(y_peer)`，对当前query相同完整key的独立peer残差求和，修正 `clip(R25+Σr/(4+m),0,1)`。tau=4预先固定，不扫描；n≤1残差0。均值中心化控制peer整体成功率，不保证能完美估计潜在能力。
2. **R72_pooled1/3**：沿用R36 within-subject centered ridge机制，输入升级为与R25完全一致的MiniLM+metadata缩放；每组独立中心化X/y后合并，ridge=1，保留own smoothed2均值。不是声称复现外部论文新结果。
3. **R72_diff3**：对每个单独query，仅移除peer前缀中与该query完整key相同的记录，重新计算peer中心和ridge；own支持保持。无同输入匹配时严格等于pooled3。与pooled3差异为同输入可见证据的条件贡献；shared slope仍受有限样本估计影响。
4. **R36_own_pooled3 / R20_peer_pooled3**：相同新cohort下重放历史原始384维语义within-peer共享方向，使用原R20的每fold来源选择ridge，分别使用own/peer层级均值。逐BM首例与原 `conditional_components.components` 数值核对。历史R36原cohort总体分数仅另列，不作新场景的配对差值。
5. **R72_morelabels_mean3**：在R25上加上合并全部可见标签的smoothed2均值与own smoothed2均值之差并clip，辨别“仅增加标签”的成功率效应。
6. **两组peer支持标签置换**：seed7201/7202，每peer的每个合法前缀独立置换；own标签保持；重新中心化、同输入残差、pooled/diff及R36全部重算。RNG仅依赖benchmark/target/peer/split_seed/budget，不含query outcomes。预算置换是独立诊断，不要求其构成嵌套的可部署轨迹。

所有无peer路径严格返回冻结R25，包括历史控制；保留frozen smoothed2、training-selected mean和R56_segment配对参照。B0应对所有方法一致。

## 评价与停止

完整5242pairs×3seed×6预算，保存概率/索引、响应计数、准确率与Brier，CI用BM cluster。该批为明确条件模拟，不能报告为Hosted效果。所有候选一次性冻结探索，不因外层结果改tau/peer分组/公式。若same-input正面效应仅来自少数命中BM，只报告条件收益与全体增量；若pooled不优于R25或置换不损失，不扩展共享网络。Research Gate与原计划相同；即便数值通过，也仍需独立核实真实runtime输入再讨论部署。
