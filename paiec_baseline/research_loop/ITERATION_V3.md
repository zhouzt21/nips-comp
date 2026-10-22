# Loop v3 实验闭环

## R32 — Measurement：64-item上限是否解释未达门槛？（已完成）

1. **Current Bottleneck**：R31改善0.003563，尚差0.001437；全部5242pair中5224的评价池被截到64items。
2. **Upward Update**：覆盖造成的混杂必须先排除，之后才能归因给表示或适应机制。R31的77%收益来自15/31预算；0/1无收益是当前局部估计器的构造，不能据此宣称输入没有冷启动信息。
3. **Operator Choice**：Audit Measurement → Confound-Isolating Diagnostic。固定模型、参数、支持标签、种子和pair权重，只恢复完整评价items并保留每条response权重。
4. **Downward Experiment**：预先按计数分层选125pair，36benchmark，三个原种子；另外将占81.6%权重的alignment_faking扩展为全部4278pair的完整评价。R32耗时1534.5秒，离线补135433个向量；R32c耗时150.8秒。精确Brier分解误差≤4.4e-16。R32的11个可直接对应历史记录的检查及R32c全部旧pair/seed/budget回放通过；其他部分抽样benchmark没有历史逐pair文件，不冒充逐pair已回放。
5. **Evidence & Gate**：主导benchmark R25完整ALC=0.190666939，均值=0.194777782；收益0.004110844，原截断收益0.004121535。用主导benchmark精确差值替换分层估计后，总体cap效应为+0.000018690，近似设计区间[-0.000160791,+0.000198172]。把它加到R31实际差值，估计完整池收益0.003544746，仍差0.001455254。这里是**差值修正估计**，不是全5242pair完整评分；该区间描述固定公开总体内pair抽样，不能代表未知benchmark泛化。重复响应经验方差会在两个Brier差异中抵消，也不能直接视为总体Bayes误差。研究Gate仍未通过。
6. **Next Move**：停止将评价截断作为主要剩余差距的解释。进入R33：区分现有局部方向的信息有限与幅度不足。原分层样本的绝对分数外推与完整主导benchmark结果均保留，不选择性替换原实验。

证据：`runs/research_loop/R32/measurement_conclusion.json`、`R32/summary.json`、`R32/replay_checks.json`、`R32c/summary.json`。原始分层cap效应=-0.000240702；补足主导benchmark后改为+0.000018690，说明小规模pair抽样本身也有不确定性，不能单凭外推点估计改写研究结论。

## R33 — 固定局部方向的幅度上界（已完成）

1. **Current Bottleneck**：同一R25修正方向是否有足够的额外风险下降空间？R17旧oracle只覆盖向均值收缩，不能回答放大问题。
2. **Upward Update**：区分方向中存在信息与当前证据足以确定幅度。R32已削弱主要测量混杂，但仍不能直接推断信息已用尽。
3. **Operator Choice**：Audit → Limit，固定一条概率修正方向，分别界定收缩、适度放大、无界放大的经验容量。
4. **Downward Experiment**：全部5242pair、三种子、原64item评价；主导benchmark另外完整评价。解析分段二次损失最小化，保留每条response权重；四项数学测试通过，所有36benchmark的R25/base逐预算均值均回放到1e-7，主导benchmark完整评分也回放通过。耗时121.8秒。
5. **Evidence & Gate**：baseline ALC=0.198687363；实际R25=0.195123926；收缩oracle=0.194532758；max4 oracle=0.189299771；无界oracle=0.187694709。允许≤4放大后潜在收益0.009387591，超过仅收缩0.004154604。主导benchmark完整评价max4 oracle收益0.010048943。说明幅度路线尚不能被排除；**不是可部署收益、未过Gate**，oracle会利用评价集噪声。n0/1方向为零，仍没有这两个预算的新增信息。
6. **Next Move**：进入R34，用支持集留一预测估计尺度，并加入训练来源内选出的固定尺度控制。若支持标签无法识别幅度，不能以oracle上界继续宣称可行，也不扩大系数网格。

## R34 — 只用当前支持标签估计修正幅度（已完成）

1. **Current Bottleneck**：R33中潜在的适度放大能否由允许的少量标签可靠识别。
2. **Upward Update**：容量上界不等于证据可辨识性，必须把oracle替换为只看支持集的估计器。
3. **Operator Choice**：Adapt by Conditioning，并用来源选择的固定尺度与支持对应关系打乱作对照。
4. **Downward Experiment**：当前支持集精确留一预测，带向1收缩的标量尺度；三折来源选tau=[1,10,10]、固定尺度=[1,1,0.5]，全部选择完成后评价5242pair/三seed。三项数值/无状态测试与所有原R31模型/统计baseline回放通过。耗时113.6秒。
5. **Evidence & Gate**：adaptive=0.195075364，固定尺度=0.196571739，打乱尺度=0.195139067，R25=0.195123926，smoothed2=0.198687363。新增收益仅0.000048562，对R25有一折退化，benchmark聚类区间[-0.000059454,+0.000024109]跨零；打乱对照的差值区间也跨零。只能报告小点估计，不能声称稳健新机制。相对均值总收益0.003611998，Gate未过，未晋级/打包。
6. **Next Move**：先诊断支持尺度与oracle的关联、来源目标是否存在一致的低/高预算冲突，不追加尺度网格。

## R35 — 支持信息与来源目标分解（已完成）

1. **Current Bottleneck**：弱改进是支持统计无信息，还是统一收缩压制高预算信号？
2. **Upward Update**：分开“统计关联存在”和“足以形成低风险的概率修正”；不能靠oracle潜力跳过这一判断。
3. **Operator Choice**：Diagnostic → Decompose，先检查预算与来源目标，再决定是否有理由区别处理。
4. **Downward Experiment**：逐pair/seed/budget合并R33/R34记录；回放三折原来源episodes，分解原四个tau的逐预算损失，ALC均与原记录一致。没有新外层tau搜索。耗时25秒。
5. **Evidence & Gate**：预算3/7/15/31的支持尺度与截断oracle尺度秩相关为-0.031/0.042/0.234/0.475。来源最优tau在15预算三折均为不调整；31预算为1/10/1，低预算偏好也不一致。因此，信息随预算增加，但“不够大胆仅因低预算压制”的统一解释不成立；相关性不能证明可实现oracle收益。未新增候选、Gate未过。
6. **Next Move**：停止本轮尺度参数优化。复核R29/R20的既有信息来源证据，再决定是否测试不同任务方向或当前许可的跨subject证据；不默认扩大网络/历史先验。

## R36 — 条件共享中的均值与item效应（已完成）

1. **Current Bottleneck**：共享subject均值与共享item方向是否具有不同的适用条件？已复核R29无源协方差迁移收益，故没有直接开发历史方向bank。
2. **Upward Update**：R20旧alignment_faking的均值共享退化提示exchangeability不能统一假定，需拆开验证。
3. **Operator Choice**：Decompose → controlled factorial，均值×方向独立改变，固定原来源正则与强own方向对照。
4. **Downward Experiment**：实际5242pair/三seed/cohort4/64queryitems，旧100pair配对保留、新pair另行分组。两个数值等价测试、全部旧benchmark子集的五个预测器回放通过。耗时172.3秒。27个每seed额外行与同cohort其他行共享完整可见subject键，原逻辑按可见配置合并；因此该条件base不能直接当独立pair分数。
5. **Evidence & Gate**：own均值+pooled方向=0.195253399，peer均值+pooled=0.197450494，own均值+原own方向=0.195325060，own mean=0.198841312，peer mean=0.200993288。取消均值共享的总体收益被主导benchmark驱动；4BM改善、26退化、6持平，两个fold退化，区间跨零。pooled相对同正则own方向改善0.002233240且打乱对照变差，但相对原强own方向仅0.000071661，一折退化、区间跨零。相对own mean仅0.003587913，Gate未过，不能选择性对较弱peer mean宣布胜出。
6. **Next Move**：不统一关闭均值共享，也不追加item pooling复杂度。检查显式独立备选与当前证据决定的依赖结构，固定原own方向；继续承认cohort4是条件情境。

## R37 — 包含独立备选的当前计数分组（已完成）

1. **Current Bottleneck**：显式独立备选能否缓解不适当的跨subject均值依赖？
2. **Upward Update**：区分单变量先验形状与组间依赖，避免把换先验误当可靠结构识别。
3. **Operator Choice**：Encode independence by construction → condition partition structure。均匀单变量先验不随浓度改变。
4. **Downward Experiment**：至多4个可见subject的15种分区，后验权重与先验固定权重各自来源选择；对照旧R11层级均值，全部搭配原own item方向。三折后验alpha=1、固定权重alpha=4，选择先于外层。四项数学测试、R36四个参照模型回放通过；耗时154.8秒。
5. **Evidence & Gate**：partition+item=0.196281688，固定分区=0.197090012，旧hierarchy+item=0.195102817，own+item=0.195325060。新方法比own差0.000956628、比旧hierarchy差0.001178871；相对固定权重的区间也跨零。未达Gate，不能把更复杂的后验分组当作可靠收益。`R37_protocol_erratum.md`纠正冻结说明中4/26/6的计数，不改变实验。
6. **Next Move**：停止分区浓度/似然变种；检查输入配置关系是否提供计数之外的异质性线索。

## R38 — 输入配置关系的可用信息（已完成）

1. **Current Bottleneck**：可见配置相似度是否能识别应当借用的peer？
2. **Upward Update**：源任务绝对能力先验失败不等于当前任务内的配置关系无信息，但关系必须直接验证。
3. **Operator Choice**：Audit available fields → relation representation → shuffled-correspondence diagnostic。
4. **Downward Experiment**：冻结排除ID/路径/标签等字段的配置Jaccard；固定R36cohort，对比同名、全部peer及8次配置映射打乱。使用全公开response估计pair率作为诊断代理，不作为可部署输入或Brier分数。无新训练，耗时2.45秒。
5. **Evidence & Gate**：最相近配置的peer-rate距离0.123158低于全部peer0.135880；但其peer均值到目标率的平方偏差0.109737高于全部peer平均0.090696，三个fold差值[-0.000726,+0.001455,+0.022352]。配置关联确实优于打乱，但硬选邻居损失了平均化收益；预先设定的三折进入条件不满足。不能将关联改写为低预算预测增益。
6. **Next Move**：不调整字段/Jaccard权重或直接开发该邻居预测器。回到独立pair，检查固定线性概率近似与Bernoulli曲率之间的差别。

## R39 — 匹配曲率的局部Bernoulli求解（已完成）

1. **Current Bottleneck**：线性概率近似与硬截断是否限制了当前二元证据的利用？
2. **Upward Update**：固定表示后，把局部目标换成匹配曲率的Bernoulli似然，可分离输出链接与迭代求解的价值。
3. **Operator Choice**：Audit objective → matched-curvature solver comparison；不增加正则网格。
4. **Downward Experiment**：R25原特征/折参数/支持标签/均值offset，比较线性、一次Newton和收敛logistic。5242pair/三seed/64queryitems；三项测试与所有旧模型逐预算回放通过。耗时78.1秒；最多7次迭代，最大梯度9.995e-9，无失败或fallback。
5. **Evidence & Gate**：Bernoulli=0.195129601，一步控制=0.195144159，R25=0.195123926，base2=0.198687363。对R25差值+0.000005675，区间[-0.000068113,+0.000013872]；两折退化。对一步控制的区间也跨零，不能归因于迭代求解。Gate未过。
6. **Next Move**：停止局部loss/link/截距调参。检验当前输入几何是否支持少数可辨认的item组；区分输入分组的作用与简单扩大局部修正。

## R40 — 当前输入分组与标签条件化（已完成）

1. **Current Bottleneck**：输入的主要变化方向是否定义了成功率不同的item组？
2. **Upward Update**：输入几何不能自动解释为标签结构；必须用实际预测和对应关系控制检验。
3. **Operator Choice**：Input-only partition → count-conditioned model averaging。不借用其他pair标签。
4. **Downward Experiment**：第一主方向中位分组，精确比较共享/独立Beta率，先验赔率1:1。5242pair/三seed；四随机方向与四标签置换控制。评分前修复中位边界的符号对称性，四项测试通过；原R31回放通过。耗时151秒。
5. **Evidence & Gate**：分组0.200578409，随机方向0.200551033，打乱标签0.201521788，R25 0.195123926，base2 0.198687363。对R25三折均差，差值区间[+0.001813,+0.005972]；对随机方向区间跨零。正确对应关系有小幅信息，但这个高方差分组规则不能转化为强预测。Gate未过。
6. **Next Move**：停止分组数/阈值/先验网格。先检查原表示在提取、截断及metadata筛选后造成的输入等价类合并及经验损失，区分信息丢失与不合适的几何。

## R41 — 输入等价类的信息损失审计（已完成）

1. **Current Bottleneck**：确定性的输入处理是否合并了有不同结果的官方输入？
2. **Upward Update**：将表示碰撞与表示的几何质量分开；复用R18计数审计，新增响应加权的嵌套等价类损失。
3. **Operator Choice**：Audit representation → nested equivalence classes；只作诊断，不训练预测器。
4. **Downward Experiment**：5242pair/三seed原64queryitems；每级用评价标签拟合组内最佳常数，仅计算经验逼近代价。两个解析测试、嵌套损失非负检查通过。耗时52.6秒；分词器提示原串超过512是截断前计数警告，本轮没有向模型传入这些长串，也没有进行模型前向。
5. **Evidence & Gate**：raw/canonical/token/R25信息键经验损失为0.001976/0.002236/0.002305/0.003051，总碰撞代价0.001076；benchmark等权0.012822。alignment_faking代价0。benger、edu_circuit_hw占主要部分。原始键还含source_task_id、图片路径等，原始oracle可能仅靠身份区分；阶段归因依赖合并顺序，不能宣称恢复这些标识会有预测收益。benger原问题正文不同但保留窗口基本相同；edu主要缺图像像素。不是总体Bayes噪声下界或可实现ALC改进。
6. **Next Move**：不恢复opaque ID或重开长度网格。检验当前支持输入的共用模板能否被安全移除，让同一126token预算保留更多实际问题内容；只从支持输入估计模板，先来源内小规模实测，不改编码器。

## R42 — 当前支持上下文中的共用模板（已完成，未进入外层）

1. **Current Bottleneck**：保留窗口被共同模板占用，是否限制当前证据关系的表达？
2. **Upward Update**：区分内容缺失与表示几何；固定模型/窗口，仅按当前支持输入移除共同前后缀。
3. **Operator Choice**：Condition representation on support context → preserve variable content。与R18加长窗口不同，不增加token预算。
4. **Downward Experiment**：seed73/max25pair来源内pilot，三来源316/260/248pair；全部原R25来源分数回放通过。三项模板隔离测试通过；未变窗口MiniLM前向差<1e-5；离线编码4451个新窗口，耗时91.4秒。
5. **Evidence & Gate**：语义版三来源差值[-0.000001265,-0.000004189,+0.000004170]，第三组pair/BM等权均退化，未通过全三组改善的进入条件。词法控制pair差值[-0.000144193,-0.000203181,-0.000001557]，但前两组BM等权退化，不能事后换主候选宣称成功。来源内结果不是OOD证据，未运行外层。
6. **Next Move**：停止该前后缀规则及长度/边界网格。重构为“学习当前证据的可靠性”，检验源任务训练的查询条件化MLP与仅episode条件化控制；保留原特征，不重开绝对成功率先验。

完整tools回归于R42执行期间实际通过90项（8.46秒），日志`runs/research_loop/v3_regression_tests.log`。这是该时点的测试数，不自动包含后续新增实现。

## R43 — 学习当前证据的条件可靠性（已完成，未进入外层）

1. **Current Bottleneck**：学习当前证据的可靠性，是否比学习绝对成功率更能跨任务迁移？
2. **Upward Update**：分离episode整体幅度与query-specific可靠性，并让完全不修改R25的epoch0参与选择。
3. **Operator Choice**：Property-targeting episodic supervision → query-information ablation；16维不含身份的互补对称统计，16/8隐藏层MLP。
4. **Downward Experiment**：三外层来源各两折benchmark内层验证；query/episode两个控制；固定20epoch，全部6个来源重拟合权重已导出，NumPy/Torch逐epoch误差<1e-5。两项互补/无状态/空预算测试通过，原来源分数回放通过；耗时41.3秒。
5. **Evidence & Gate**：query选择epoch[0,0,20]，episode[0,0,9]。前两组选择原模型；第三组内层query0.212692920、episode0.212781164、R25 0.213091212。训练损失在所有内层下降，但前两组多次验证退化，不能归因于训练未运行；第三组正面结果不能替代全三组进入条件。admit_outer=false，未评价外层，来源内分数不是泛化估计。
6. **Next Move**：停止该MLP可靠性设计的宽度/特征/损失网格。复查R6负证据适用范围：其使用词法核，加入结构项同时改变了边际方差；用冻结参数的表示×方差分配对照检验这两个混杂，不重新调eta。

## R44 — 旧协方差负证据的适用范围（已完成，未进入外层）

1. **Current Bottleneck**：R6的失败是否依赖词法表示或结构项改变边际方差的混杂？
2. **Upward Update**：固定二元边际矩只能给出模型假设，不能保证泛化或概率校准。
3. **Operator Choice**：Audit assumption → representation×variance factorial；保持原eta/核宽度。
4. **Downward Experiment**：原R3词法缓存只读；来源316/260/248pair。词法/语义×原/固定对角，另设均匀相关控制。两项矩阵/单标签解析测试通过；原R6和R25来源分数回放通过。耗时6.3秒。
5. **Evidence & Gate**：语义固定对角三来源ALC[0.218476,0.202493,0.214519]，原语义协方差[0.218000,0.202418,0.212854]，R25[0.217718,0.201942,0.213091]。三组均退化；不能以矩约束合理替代实测收益。未进入外层。
6. **Next Move**：停止该方差分配路线。检验现有语义与词法局部预测是否包含互补误差信息；固定等权组合，用精确Brier恒等式与词法对应关系破坏控制，不搜索混合系数。

## R45 — 既有局部估计器的互补性（来源内已完成，进入外层）

1. **Current Bottleneck**：语义平均更好，是否意味着词法估计器没有增量信息？
2. **Upward Update**：区分单独性能与误差互补性；两个估计器共享同一窗口和当前支持标签，不混合历史zero-shot先验。
3. **Operator Choice**：Complementary evidence → exact Brier decomposition → correspondence destruction。
4. **Downward Experiment**：原128窗口、既有正则、固定1:1概率平均；来源316/260/248pair，四次词法query置换。精确Brier恒等式测试、全部R25/R42词法来源分数回放通过。耗时30.8秒。
5. **Evidence & Gate**：三来源组合ALC[0.217561630,0.201926364,0.212284758]；对R25变化[-0.000155949,-0.000015556,-0.000806454]。三组BM等权均改善且胜打乱对应关系控制。达到pilot进入条件，尚不是OOD收益，更未过最终Gate。
6. **Next Move**：进入R46，所有方法/权重冻结，实际5242pair/三seed外层配对评价，不调混合系数。

## R46 — 固定语义/词法组合的外层评价（已完成）

1. **Current Bottleneck**：来源内的互补性是否能改善外层比赛主指标？
2. **Upward Update**：互补信息存在，不保证组合后的偏差/方差在新benchmark仍合适；辅助指标不能替代主指标。
3. **Operator Choice**：Freeze complementary estimators → paired outer replication；不调权重。
4. **Downward Experiment**：5242pair/三seed/64queryitem及全部响应，固定1:1平均；原R31各预算与Brier恒等式回放通过。耗时147.5秒。
5. **Evidence & Gate**：组合0.196033067，词法0.197343986，打乱0.196670087，R25 0.195123926，base2 0.198687363。组合对R25差值+0.000909141，区间[-0.000363944,+0.001086625]，两折退化。23/36benchmark改善，BM等权0.212476689好于R25的0.212920227，但pair等权变差。对打乱控制三折改善且区间全负，支持对应关系信息；不支持算法晋级。Gate未过。
6. **Next Move**：停止混合系数/子集选择。先确认多模态信息是否在官方推理环境可用，避免继续在未确认信息契约上开发。

## R47 — 多模态信息可观测性（公开资料审计已完成；运行时契约未确认）

1. **Current Bottleneck**：公开训练资产可下载，是否意味着隐藏评测中也可读取像素？
2. **Upward Update**：必须区分数据存储契约与predict接口契约；相对图片路径不是图片内容。
3. **Operator Choice**：Audit available information → runtime-contract boundary。
4. **Downward Experiment**：读取官方页面、当前HF revision、ai2d/hallusion/edu资产表目录及官方curation schema；只读取公开入口，不访问隐藏数据。公开起始包页面/API的requests访问返回SSLError，web工具亦无法读取API。没有下载像素或新编码器。
5. **Evidence & Gate**：三个assets.parquet分别45,514,552 / 142,851,270 / 292,507,671 bytes；schema通过asset_manifest关联。当前HF revision与本地相同。官网列出的四个item字符串字段没有明确提供asset_manifest/挂载解析契约。这不证明像素绝对不可用，但不足以支持可部署的像素方法。Gate状态不变。
6. **Next Move**：像素分支等待公开运行时说明或可访问的起始包；不关闭TLS验证绕过错误。继续独立可执行诊断：R33幅度oracle是否主要受同样本拟合乐观偏差影响。

## R48 — 理想幅度空间的跨面板稳定性（已完成，仅诊断）

1. **Current Bottleneck**：R33头部空间中，有多少可以从另一组同pair标签复现？
2. **Upward Update**：R34/R43的估计器失败，可能来自有限证据不可辨认，也可能来自原oracle的样本内乐观性；两者尚未分开。
3. **Operator Choice**：Audit diagnostic measurement → split fitting from assessment → reassess identifiability。
4. **Downward Experiment**：5242pair原cap，以及4278主导pair完整评价池；按distinct item对分评价池，双向拟合/评分，固定c≤4。添加校准对应关系打乱与额外标签均值控制。7项相关测试通过；初次测试仅测试预期值传list给旧数值helper失败，已改为array后通过，尚未评分时修复；未修改原helper。代码、协议、日志见R48。
5. **Evidence & Gate**：耗时197.1秒。capped跨面板0.191438453，对R25改善0.003685473，但一折退化、benchmark聚类区间[-0.004323704,+0.000135055]跨零；对打乱控制三折改善且区间全负。完整主导benchmark跨面板0.185349420，样本内oracle0.184728840，R25 0.190666939。幅度信息并非全由同样本拟合制造，但没有跨benchmark一致性，更没有预算内可实现性保证。两类oracle之间差值还含拟合样本量变化，不能全部归因为纯乐观偏差。额外标签均值0.171105458在所有预算不变，尤其低预算已经使用额外标签；不能与合法方法进行性能晋级比较。
6. **Next Move**：用现有允许标签分配方向拟合和校准角色，检验独立校准收益能否补偿方向拟合样本减少；不重开幅度参数网格。

## R49 — 预算内分离方向与幅度证据（已完成，未进入外层）

1. **Current Bottleneck**：把有限标签分成独立角色，能否实现R48提示的可迁移修正？
2. **Upward Update**：有额外标签时的可识别性不能替代固定预算下的可实现性；需要同时计入方向估计损失。
3. **Operator Choice**：Separate evidence roles → matched split-head control。
4. **Downward Experiment**：n≥7时四个固定平衡分割、双向共八个head；每个head仅用拟合半边确定均值/方向、另半边校准c≤4。n≤3保留R25。两项互补对称/确定性/query隔离测试通过，三来源316/260/248pair旧分数回放通过。耗时18.8秒。
5. **Evidence & Gate**：三来源校准ALC[0.217961434,0.202762320,0.213416317]，R25[0.217717579,0.201941920,0.213091212]，全部主指标退化。前两组胜同样split的未校准控制，第三组反而退化；来源进入条件失败。不是外层成绩，不导出新候选。
6. **Next Move**：停止分割比例/次数/尺度网格。先检查剩余未利用的官方interactors字段是否在pair内部随item变化；只有实际存在变化才讨论增加编码。

## R50 — 未利用interactors字段的覆盖审计（已完成）

1. **Current Bottleneck**：当前局部模型是否遗漏了能区分item的官方输入来源？
2. **Upward Update**：R32同item内部无interactor变化，不等于同pair不同item之间也不变；先区分这两种计数。
3. **Operator Choice**：Audit available information → variation before representation。
4. **Downward Experiment**：原36benchmark/5242pair，只统计输入字段非空及组内变化，不计算其与结果的相关性；不添加模型或下载数据。
5. **Evidence & Gate**：8.6秒完成5242pair审计。只有emoji_attack的10pair非空且组内变化，每pair最多11种。基于已有R31分数，若只改变这10pair、其他预测保持不变，即使把该组全部误差降至零，整体改善上限也只有0.000403392，小于剩余0.001436563。这个窄修复上限不是一般信息论界，也不能外推隐藏benchmark。
6. **Next Move**：不优先开发该字段的编码器。可观测信息扩展中，像素路径仍需公开运行时契约；requests的两个官方域名均报SSL unexpected EOF，Windows原生curl也握手失败，不能仅归咎于Python证书包。远程web与AnySearch文本提取也未取得起始包页面；后者返回提取失败而非API连接失效。待公开资源契约可核实时再进入像素对照，不关闭TLS验证、不访问私有数据。现有已否定的幅度、split、词法混合与协方差网格保持关闭。见`reports/research_loop_v3_findings.md`的证据边界和进入条件；当前没有后台任务，研究Gate仍未达到。
