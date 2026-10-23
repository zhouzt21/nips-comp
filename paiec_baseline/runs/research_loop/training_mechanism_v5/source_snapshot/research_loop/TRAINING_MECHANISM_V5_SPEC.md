# plan_2 五候选的评分前执行合同

2026-10-22。用户完整授权原文 `doc/plan_2.md` 保持不变；本文件补充可执行参数和证据边界，不代替或删改原文。轮次R54/R55/R56/R57/R58依次对应候选1/2/3/4/5。R53仍为用户确认等待资源的ADeLe准备。

## 问题、假设与证据

问题是已有负迁移来自训练截断、固定支持集、来源风险权重、单checkpoint噪声，还是有诊断依据的初始化动力学限制。分别以一个因素的匹配对照检验，不预设任何方向会提高成绩。重要主张须映射原始曲线、checkpoint、六预算指标和机制诊断；不以训练loss下降替代OOD收益。

AnySearch用于检索，原文PDF与作者仓库固定在 `D:/nips-comp/paiec-dataset/training_mechanism_sources/`。Kumar/DomainBed/GroupDRO/SWAD/Kaddour/Woodworth及2026谱理论分别提供诊断、选择、加权或平均的依据。本轮为PAIEC受控迁移，不重跑论文全部原始任务；Benchmark混合风险不是完整GroupDRO，短窗口SWA不是完整SWAD，Taylor/CKA是本任务诊断，不宣称直接验证谱理论。原文/代码不具备的条件或代码公开缺口必须记录。

## 共同数据与测量合同

- 数据revision `54074453da7b9911be0f48e4c560be547bc84d21`，MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。沿用R25三个source pools及R43的Benchmark互斥二等分，两方向分别拟合。各source pool共316/260/248个合法pair；source pool之间有重叠，不能称为三次独立盲测。outer heldout不参与拟合、选择、条件触发或本轮诊断。
- 固定验证seed73、每Benchmark至多25pairs、每pair至多64query items，全部重复responses保留。六预算0/1/3/7/15/31、ALC权重.1/.2/.2/.2/.2/.1、pair等权。所有epoch使用同一验证Benchmark与query集。训练拟合变换仅接触fit输入。
- 普通MLP评价使用原2048合法字段编码；相同item但interactors不同的响应分别编码并保留计数，不把它们错误合并。六预算沿用 `evaluate.py` 的residual strength2预测，B0是原prior；不是更换部署默认常量。分类正确率固定p≥0.5，属成功预测正确率，不是subject本身能力分数。
- 保存六预算Brier/ALC、benchmark-equal、逐Benchmark与pair指标、分类正确率；核对R25和两种强均值。只在source条件成立后，才固定程序进入原5242pairs/三个原种子外层并接受原0.005 Research Gate。否则保留负结果停止；不自动替换fallback或提交。
- CPU4threads，训练串行；每个fit最多1800秒，超时/非有限数写incomplete，不能记为完整负结果。保存epoch0及每epoch权重、训练步数、loss、导出误差、数据与代码SHA256。已完成目录只按完全一致contract恢复。

## R54 / 候选1：完整短轨迹

普通MLP固定2048→128→64→1、ReLU、sigmoid、AdamW(lr.001,wd.0001)、batch256、pair-weighted Brier；复用 `train._load_examples` 每Benchmark最多400response行、sampling=response、seed2026。每方向30epochs，不受patience中断。与历史普通MLP相比，来源划分统一为上述R43内层，选择分数统一为固定query的B0；保留的是原网络/优化/采样配置和patience规则，不声称旧checkpoint逐位复现。

R43保持16→16→8→1、g=4sigmoid(log(1/3)+MLP)、输出层初始化0、p=clip(b+g*d,0,1)、原16统计特征、AdamW(lr.001,wd.001)、batch1024、Brier+0.01(g−1)^2、seed4307。query-specific与episode-only均训练30epochs；训练数据固定seed73，原特征标准化/clip不变。epoch0精确R25。候选2只比较前20epochs，以匹配原R43总更新数。

每模型/方向记录全部训练Brier、验证六预算、epoch0参数及固定fit probe（最多128行）的logit一阶Taylor误差与hidden CKA。Taylor基于初始化精确方向导数，不用参数距离冒充线性化误差；这些probe诊断不接触outer。

选择规则分别保存：原patience5/min_delta1e-5（普通MLP从epoch1开始，R43允许epoch0）；同早停停止步数内的精确最佳；best≤20、best≤30；固定epoch20/30。每source pool选公共epoch时，两个方向验证按pair数合并；不用outer选epoch。训练步数不同的选择规则只作轨迹诊断，不称为匹配算力算法排名。

延迟恢复预定义：早停停止之后存在连续3epochs验证指标至少比早停最佳低0.001。若同一pool两块互斥验证Benchmark组均表现为晚期(21–30)训练Brier均值比早期(1–5)低≥0.001，晚期验证均值比早停最佳高≥0.001且无上述恢复，则停止延迟泛化/长Grokking搜索。未出现恢复不证明任意长训练都不会恢复。

## R55 / 候选2：只改变支持集经历

固定episode组复用R54前20epochs；resampled组每epoch使用seed5501..5520，在相同合法source pair的完整item/response池内重新划分support/query并为每support item随机取一个真实response。pair采样集合保持相同；各pair item总量固定，所以每预算query item数量及每epoch更新次数一致。每epoch仍覆盖3/7/15/31四档全部组合，预算分布/权重不变。拟合变换固定来自相同seed73 fit参考episode，防止变换拟合也成为干预因素。query与episode-only使用完全相同重采样池及batch顺序。

保存训练episode身份/内容hash、support/query不交断言、每周期暴露量及步数。可预生成逐epoch数据，但不按模型分数选择或重采样。验证始终固定seed73，选best≤20；同一pool双方向合并选择同一epoch。

选中模型做4次固定seed支持label-only permutation，重新计算R25基方向d和全部z；同时评分置换后的R25。只看原始配对减置换会混入R25自身的对应关系收益，故另外报告 `(model−R25)_true − (model−R25)_permuted`。B0/B1严格不变。

进入外层要求三个source pools中resampled query都比R25、fixed query、resampled episode-only及两种强均值的ALC低至少1e-5，benchmark-equal优于R25，且配对净增益诊断方向正确。若不成立则停止该配置，不加宽/换loss/扫描采样策略。

## R56 / 候选3：固定λ=.5的Benchmark混合风险

普通MLP完整复用R54训练行、初始化、batch顺序、30epochs和选择规则。唯一变化为训练行权重：pair风险每行N/(P*n_pair)，Benchmark风险每行N/(B*P_b*n_pair)，取二者均值；两种权重全数据平均均为1。不是最坏群组minimax，也不是重抽数据。

报告各Benchmark总风险质量、有效权重范围、初始化完整fit梯度的cosine/norm ratio和训练中gradient norm，以及逐Benchmark增减。相对pair风险和R25分别比较，benchmark-equal改善不抵消pair-equal退化。只有三个source pools都在pair-equal上优于匹配pair风险、R25和两种强均值至少1e-5，且benchmark-equal不退化，才进入outer。

## R57 / 候选4：仅对稳定区间平均

读取候选1曲线，普通MLP用B0、R43用ALC。各pool选择最早符合条件的连续3epochs窗口（从epoch1起）：窗口max−min≤0.001，且全部≤patience最佳+0.001；三个pool都存在窗口才启用该模型族。该阈值是预声明工程准入，不是论文定理。若无窗口则记录条件未满足，不在过拟合后期强行平均。

同一窗口对比：原单一best≤30、窗口内最佳单一、参数EMA(decay=.9，初始化为窗口首权重)、等权SWA、三个checkpoint的最终预算概率等权平均。无新增训练更新；变换固定不平均，所有线性层weight/bias平均。不是SWAD逐step dense loss-valley复现。只有相对两种单一基线和R25/强均值的pair-equal ALC在三个pool均稳定改善至少1e-5，benchmark-equal改善，才考虑outer。多checkpoint推断成本单列。

## R58 / 候选5：诊断触发后才改变初始化

普通MLP固定fit probe上，若至少两个source pools的两个方向均满足：(a)epochs1–10中位Taylor相对误差≤.1且CKA≥.9；(b)epochs15–30出现Taylor相对误差≥.3或CKA≤.8；(c)后期连续3epochs来源验证B0比early-stop最佳低≥.001，才触发初始化实验。指标阈值为本任务操作化假设，不等于已证明lazy/rich机制或因果关系。

触发后只比较默认初始化和输出层weight/bias×0.1，其他与R54相同、30epochs、同选择规则；这不是Kumar的完整输出重参数化，也不直接测试耦合L2谱定理。未触发则记录实际诊断与未执行状态。无论结果如何，本轮不接续learning-rate/weight-decay网格；后续weight decay须另有证据和合同。
