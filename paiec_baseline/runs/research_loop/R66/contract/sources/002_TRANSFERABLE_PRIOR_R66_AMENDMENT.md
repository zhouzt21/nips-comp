# R66 pre-fit amendment

R65已完成15,726轨迹无新训练诊断。R56 B0=.243287225，R54=.255450074；仅来源mixture常数=.251947015，未解释R56的全部收益。R56平均概率.385996，来源常数平均.507462，目标均值.430466；不能简单声称网络输出等于全来源prevalence。H1/H2/H3及局部来源模式仍不可辨识，故进入特征删除消融。

历史等价性：`reports/mlp_benchmark_holdout_2026-10-22.md`的A1为batch512/pair-risk/旧一次划分及早停；R15/R16为标量能力估计；R52为不同IRT结构。均非当前R56严格等价对照。原R56完整输入组直接复用6份完整30epoch轨迹，不重训。

冻结候选只有两种：subject-only（原8字段namespace，彻底不编码item）；item-only（仅原item3字段namespace，不添加missing-subject常数）。通过原encoder的私有函数副本/按原加法次序获得向量，从epoch0按对应特征训练，禁止对旧权重遮盖输入。不改变2048→128→64→1容量、seed2026、AdamW .001/weight_decay .0001、batch256、mixture风险、400rows/BM采样及30epochs。相同源行和同一个顺序；数值回放full编码与原缓存必须array_equal。

3来源池×2互斥Benchmark方向×2消融=12新fits，各30epochs。原full组复用，不算新fit。逐epoch来源验证与原协议相同，以两个方向pair加权B0曲线epochs1..30的argmin选共同epoch；记录各方向结果及训练拟合误差。CPU4线程、串行，单fit1800s历史stop，预计新增训练/评估不超过45分钟；若超限保存未完成，不自动扩网格。

比较R25、原R56及仅fit样本mixture加权常数。常数采用同一strength2更新。B3–31的分段诊断固定回到R25，比较低预算信息增量；完整MLP6预算也保存。来源选择使用同一验证标签，故所选epoch分数属于source selection evidence，不能说成未参与选择的独立验证。

R67探索触发预先限定：某消融的B0/B1分段至少2/3来源池相对R25改善、合并pair与BM方向均改善，且至少2/3来源池优于fit-only常数；移除alignment后的描述性源结果仍改善，并可说明主要负迁移与合法特征含义。数值满足只触发机制审查，不自动晋级；禁止按outer结果选消融。若无候选通过，R67 condition_not_met，R68继续。

源池重叠不作独立三重复；单一模型初始化，不把三item seeds当训练seed。此阶段不运行新全外层；只有R67条件满足才另冻结预测设计及完整外层合同。
