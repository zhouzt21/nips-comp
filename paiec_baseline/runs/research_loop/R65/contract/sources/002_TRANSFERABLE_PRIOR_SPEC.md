# R65–R68 transferable prior research contract

控制原文为[plan_5.md](../doc/plan_5.md)，逐字保存的用户计划以该文件为准；本文是补充执行合同，不替代或改写原计划。

## Research spec

Question: R56的B0/B1均值收益是可迁移subject/item信息，还是训练来源prevalence匹配与风险权重变化？贡献类型为机制诊断、受控复现与条件干预，全部公开外层证据为探索性。

H1 subject prior；H2 item distribution prior；H3 interaction；H4 source prevalence matching；H5 risk weighting/calibration。可共同成立。反证包括来源隔离验证失败、常数先验相当或更好、优势只在已知主导Benchmark、误差关联跨折反转。

非目标：新超参数网格、目标标签门控、扫描mix、扩大网络、重复R43/R55/R46/R61原配置、自动提交。保持R25、R56_segment记录与fallback。

## R65: scoring contract (before new diagnostics)

- Dataset: measurement-db revision `54074453da7b9911be0f48e4c560be547bc84d21`；MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。
- 直接读取R62已存概率、index、paired_scores、models及来源文件，不重新抽样。5242 pairs × seeds0/1/2，36 Benchmarks；重复response计数保留，pair等权。六预算权重.1/.2/.2/.2/.2/.1。
- R25/R54/R56的B0/B1精确四项分解；B1相对B0的平方误差位移分解，并扣除R25相同更新收益。验证代数恒等式与原始分数。
- 训练来源常数：用原训练样本及pair/mixture权重估计，严格只用对应外折训练来源；仅作诊断比较，不调权重。
- 合法单输入描述：subject字段缺失/长度，item文本长度/字符结构/metadata数量，冻结语义向量低维统计，模型概率/不确定度，已揭示首标签。整组query均值仅离线诊断，禁止进入预测器。
- 逐Benchmark与alignment/其余分开报告：预测均值、真实均值、偏差、方差、协方差、特征分布差异和误差关联。相关性不等于机制识别。
- 训练源prevalence、输入统计仅来自训练源；评价标签只进入计分/诊断，不进入特征、模型选择或门控。
- 保存code/data/model hashes、命令、环境、时间、原始分解及聚合。CPU4threads；R65不训练。

## Conditional stages

R66先查历史严格等价性。若H1/H2/H3仍不可辨识，冻结少量从训练开始删除特征的匹配消融；同来源样本、mixture risk、初始化2026、2048→128→64→1、AdamW、更新预算与benchmark-disjoint内层checkpoint规则。具体候选及计算预算在训练前另写amendment。

R67仅在合法含义信号、来源分组正面证据、负迁移解释、R25互补、无需目标标签选择全部满足时进入；否则condition_not_met。若进入，先冻结设计，再运行原全外层协议与cluster bootstrap。模型seed不等于item-split seed。

R68检查128items×4rubrics原始准备、资源及特征可得性；保持用户ADeLe等待资源约束。共享标签必须由官方调用合同确认，否则blocked_by_runtime_contract，不构建依赖隐含状态的模型。

Research Success Gate保留≥.005双强基线改善、三折改善、BM等权改善、cluster CI上界<0及无未解释严重退化；探索进入与提交选择独立。未触发实验不记为失败训练。
