# R59–R61：评分前诊断与条件组合合同

2026-10-22。用户完整研究计划以 `doc/plan_3.md` 为准，原文保持不变。本文件仅补充执行参数，不替代、压缩或修改用户计划。

## Research spec

- Question：R54/R56的B0/B1局部优势能否由预测时合法输入识别并成为R25的稳定增量？贡献类型为既有方法的实证诊断与受控迁移，非算法新颖性主张。
- H1：有效item差异；H2：pair平均概率；H3：来源偶然匹配；H4：R56改变均值偏差或预测方差。均由response计数加权的精确Brier恒等式检验，不把相关性解释为因果。
- 可能推翻局部优势的证据：跨Benchmark方向不一致、去除单Benchmark后收益反转、优势依赖评价标签oracle、合法信号与优势方向无法迁移。
- 非目标：大模型重训、固定混合系数/阈值网格、架构扫描、ADeLe、正式提交；不得以诊断重组冒充独立验证。

## 冻结输入、选择与协议

沿用V5 dataset revision `54074453da7b9911be0f48e4c560be547bc84d21`、MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`、三个重叠来源池316/260/248 pairs、Benchmark互斥两方向、seed73、每Benchmark至多25pairs、64query items及全部重复responses。预算0/1/3/7/15/31，ALC权重.1/.2/.2/.2/.2/.1，pair等权；另报Benchmark等权。

R54采用既存 `fold*_mlp_analysis.json` 的best30；R56采用既存 `fold*_selection.json` 的epoch。不能在本轮重新按评价结果选择epoch。来源选择本身已看过这些Benchmark的标签，故本轮所有诊断保留选择偏差，不能称为完全独立OOF验证。R25沿用各池semantic_ridge与metadata_ridge16，强基线为smoothed2及历史training-selected smoothed。冻结依赖/权重/输入哈希，保存精确代码快照与环境、完整命令；不改变原模块以免使历史缓存失效。

## R59 执行与产物

1. 回放冻结权重和R25。恢复每个query item及interactor variant的实际response计数/成功次数，对齐support/query身份和特征。R25的item概率广播至相同variant；不平均掉重复response的Brier贡献。逐pair/预算分数须与V5匹配，容差1e-7；失败先修实现，不解释效果。
2. 保存response-group级预测、counts/successes、support/query身份、分组/权重路径。对每pair六预算计算Var(Y)、均值偏差平方、Var(P)、−2Cov(P,Y)、Brier和p≥.5预测正确率；逐Benchmark和来源聚合。Var(Y)不是不可约噪声。
3. R54/R56相对R25以及R56相对R54的配对差值逐项分解；低预算权重.1/.2不重新归一化，因而直接等于预算分段对ALC的贡献。另报全六预算。
4. 诊断预算分段：B0/B1分别取冻结R54或R56，B3–31用R25。二者均为预指定诊断对照，不选择来源最佳后再报总体。保留完整基线、预测正确率、pair/BM等权、各Benchmark增减和移除单Benchmark敏感性。
5. 同pair预测均值替代和去中心variation成本为代数诊断，允许看query预测本身；真实成功率与oracle选型只用于诊断，绝不能部署。计算pair/Benchmark事后最佳选择的乐观空间，明确标注oracle。
6. 预指定合法信号：MLP预测均值距.5、预测方差、R54/R56预测分歧；B1另含第一个合法support标签对MLP原始prior的平方误差。按Benchmark等权做Spearman方向审计，并报告不同来源池一致性；仅相关不能证明有效门控。不得用subject/Benchmark ID或query真实成功率作门控。
7. 每来源池/方法的ALC配对差用Benchmark-cluster bootstrap（seed5907，2000次）给95%区间；这是已查看公开来源的描述性不确定性，不是多次自适应选择后的确认性显著性。

## R60/R61 条件与停止

R59为诊断，不以oracle本身触发候选。预算分段若在三个池的pair ALC均改善且BM等权不退化，可直接冻结为R60最小候选。若分段不稳，仅在预指定合法信号与收益的Benchmark级关系在三个池方向一致、具备可说明的信息来源且明显不是来源ID/标签泄漏时，才讨论一个机制明确的条件候选；方向一致只是必要条件，还需检查来源关联、支配Benchmark和可实施的独立fit选择。未满足则R60/R61记condition_not_met，不扫描阈值或训练模型来追逐当前分数。

若需学习门控，须另行冻结训练Benchmark内独立分组/cross-fitting合同，检查基础模型训练和checkpoint选择是否见过该门控评分Benchmark，不能简单将跨半折预测拼接后随意leave-one-Benchmark-out冒充无泄漏门控。新增训练上限仅用于获准的门控，CPU4threads、单fit1800秒；R59无新fit。

R61来源门槛按用户原文：三个池pair ALC均改善、BM等权不退化、机制方向一致且无未解释严重Benchmark退化。严重回归预警设为单Benchmark ALC增加≥.005（报告而非据此调参）。不通过就停止，不反复调整。只有来源通过才冻结候选并运行原5242pairs×3种子外层，检查原.005Gate、三折方向、BM等权和cluster95%上界<0。保留R25及fallback，不自动提交。

## 文献与计算边界

AnySearch用于发现和原入口提取，关键理论与实现使用原文/作者代码。精确Brier分解直接由代数及原始响应展开核验，不借搜索摘要代替证据。不宽泛重做文献综述。实际耗时和CPU环境记录在run证据；本轮脚本和缓存读写不计用户时间。
