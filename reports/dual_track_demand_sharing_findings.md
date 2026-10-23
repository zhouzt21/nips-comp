# PAIEC plan_8：R73–R76 双路线结果与研究认识

日期：2026-10-23。原始任务：[plan_8.md](../doc/plan_8.md)。两条路线已执行至各自的预设停止点；没有正式提交或恢复 Hosted 工作。所有分数都是重复使用公开 Benchmark 的探索性结果。

## 1. 结论与实际候选

|路线|实际完成|结论|保留方案|
|---|---|---|---|
|No-Peer|R73：128 items × 4 rubrics，512主标注、64独立复标；R74：CPU学生和三个来源池的教师/学生/置换/局部对照|未证明四维需求标注提供超出现有表示的可迁移增量；按合同停止，不执行新的完整外层|R25，既有完整外层 ALC **0.195123926**|
|Shared-Peer|R75：共享方向诊断、表示/中心化/ridge消融；R76：5,242 pairs × 3 seeds × 六预算，全场景及原概率重计分|固定 partial8 比同一 peer 组成的完全共享有小幅增量；仍有严重负迁移和可见性限制|研究候选 partial8，ALC **0.190547856**；仅 `conditional_simulation`，不晋级部署|

历史 No-Peer 点估计最低参照 R56_segment 为 0.193805399，但此前未通过全部 fold、Benchmark-equal 和区间条件；本轮没有将它升级为默认方案。新共享候选与 No-Peer 使用不同总标签信息，不能将二者排成相同预算下的公平算法榜。

## 2. 从哪里阅读、复核与复现

|内容|存储入口|说明|
|---|---|---|
|本轮原始计划与冻结合同|[plan_8](../doc/plan_8.md)、[总合同](../research_loop/DUAL_TRACK_SPEC.md)、[A合同](../research_loop/demand_pilot_SPEC.md)、[B合同](../research_loop/SHARED_SLOPE_CONTRACT.md)|假设、来源、进入/停止条件和适配范围|
|标注与 QC|[标注报告](demand_annotation_findings.md)、[R73](../paiec_baseline/runs/research_loop/R73)|原始子代理响应、rubric/prompt hash、最终标注、复标差异和长度审计|
|No-Peer 指标与解释|[R74主报告](demand_pilot_findings.md)、[全表](demand_pilot_tables.md)、[复现](demand_pilot_reproduction.md)|Brier、准确率、grade误差、CI、失败切片及停止决定|
|Shared-Peer 指标与解释|[共享主报告](shared_slope_findings.md)、[全表](shared_slope_tables.md)、[复现](shared_slope_reproduction.md)|0/1/2/3 peers、缺失/不均衡/其他组成、各预算、fold、回归与实际成本|
|模型、概率及身份|[R74](../paiec_baseline/runs/research_loop/R74)、[R75](../paiec_baseline/runs/research_loop/R75)、[R76](../paiec_baseline/runs/research_loop/R76)|fit身份、源代码快照、配置、预测、counts/successes和重计分|
|新 item 的 CPU 实现|[特征提取器](../research_loop/dual_track_cpu_features.py)、[需求学生](../research_loop/demand_student_cpu.py)、[共享输入适配器](../research_loop/shared_slope_adapter_cpu.py)|本地 MiniLM/metadata、NumPy heads、当前输入标签、保守回退|
|审查与交付核对|[只读结构复核](dual_track_structural_review.md)、[交付验证](../paiec_baseline/runs/research_loop/dual_track_v8/delivery_validation.json)、[文件hash](../paiec_baseline/runs/research_loop/dual_track_v8/delivery_manifest.json)|区分科研证据、代码检查、合成 smoke 与 Hosted 验收|
|全局状态|[HIGH_LEVEL_STATE](../research_loop/HIGH_LEVEL_STATE.md)、[历史信号登记](../research_loop/HISTORICAL_POSITIVE_SIGNAL_LEDGER.md)|最新决定置顶；旧轮次叙述保留为历史|

机器可读分数位于 `R74/{teacher_summary,deployable_summary,grade_summary}.json`、`R75/factor_comparisons_by_selection_fold.json`、`R76/full/{summary,additional_comparisons,mechanism_comparisons,gate}.json`。表格 CSV 与 Markdown 同目录，原始概率保存在各轮 `predictions*.npz`，须结合相邻 `index.json` 的 response counts/successes 重算。

## 3. 指标与证据合同

R76 覆盖36个 Benchmark、5,242个 distinct Benchmark–Subject pairs、种子0/1/2，共15,726条轨迹、149个chunk、1,005,741条 query variant 记录。六预算为0/1/3/7/15/31，ALC权重0.1/0.2/0.2/0.2/0.2/0.1；沿用每轨迹最多64个query item及原重复response计数。主要指标是 pair-equal Brier ALC，越低越好；另报 Benchmark-equal 和10,000次 Benchmark-cluster bootstrap。不是全部query item的无上限评分。

本文“准确率”是以预测概率 `p >= 0.5` 判断单个 response 是否成功的分类准确率，按原 response counts、pair及seed权重汇总；不是被评测 Subject 自身的正确率，也不等于 `1−Brier`。R25 B0恒为0.5，43.0466%的准确率受阈值平局规则影响。

R74为来源 OOF pilot：三个来源池824次pair暴露、412个不同pair，池之间重叠。教师 complete-case B0仅176次pair暴露、88个不同pair；不能直接将其数字与学生全来源或完整外层比较。来源拟合的标准化、学生监督和response拟合严格限定fit部分；外层标签未进入选择。

## 4. No-Peer：需求标注能否带来新信息？

### 标注质量

沿用R53的128个item和原作者四维 `AS/MCr/QLl/KNf`，四个原生 `gpt-6-luna` 子代理按item分片，另一个独立子代理复标16个item。512主记录中436个数值评分、76个有理由的null，108个item四维完整。原作者parser恢复全部436数值评分，无解析冲突；null保留，不按成功率补全。

复标64条中48条双方均有效：等级完全一致33/48=**68.75%**，分歧超过一级1/48=**2.08%**。这是同模型复标一致性，尚无人工标准答案校准；不能称“标注准确率68.75%”。唯一大分歧保留两份判断，没有用response outcome仲裁。

学生沿用原MiniLM的126个内容token窗口（head96+tail30，含特殊token总长128）。59/128条文本超窗口，完整四维样本中52条超窗口。早期256窗口的审计已明确作废并留档；标注值不变。教师可看到较长合法文本，学生可能丢失证据，这是一项待验证的替代解释。

### 教师价值、学生质量与实际 Brier 必须分开

|实验|现有输入对照|新增需求特征|差异与判断|
|---|---:|---:|---|
|教师 complete-case B0 Brier|0.249250842|0.337674293|退化 **+0.088423451**，95% CI **[+0.037788915,+0.128479434]**；未通过进入条件|
|CPU学生预测grade的MAE|fit均值0.999971917|0.991676014|仅微小改善；置换学生0.995583311，不能称高保真需求提取|
|学生先验，完整来源 ALC|0.210674592|0.210595327|增量 **−0.000079265**，CI **[−0.000295235,+0.000130880]**，跨零|
|局部需求特征，完整来源 ALC|R25 0.211347421|0.211345789|增量 **−0.000001632**，CI **[−0.000005246,+0.000001212]**，跨零|

学生先验来源准确率B0/B1为54.442%/63.281%，同来源R25为50.666%/62.992%；这不能替代配对Brier增量判别。完整表包括历史R56/R66先验、教师置换、学生置换、需求单独局部模型及各Benchmark失败切片。

R53只标了query item，缺少相应support教师标注；因此没有伪造纯教师B1或教师局部对应成绩。可部署学生则可编码support和query，已实际评价六档预算。形式上微小学生点估计通过宽松来源规则，但**联合合同要求教师独立价值也通过**，故最终 `stop_at_pilot_keep_R25`。没有新增完整外层、扩大到18维或继续超参搜索。

**认识：** 本轮没有建立“需求表示改善跨Benchmark预测”的证据。教师特征本身已退化，不能仅归咎于学生蒸馏；同时四维、小样本、null选择、文本窗口及固定线性映射都限制结论。这是本轮适配的负结果，不是否定ADeLe整体。学生是现有MiniLM/metadata的确定性函数，它能引入监督或正则偏好，却没有在推理时获得新的原始信息。

## 5. Shared-Peer：可共享的是哪一种关系？

### 来源机制对照

在完全相同的acquisition、query与peer组成下，纯384维语义的 pooled 相对 own-only 在三个来源池分别改善ALC **0.001159/0.001242/0.001345**；语义+metadata pooled相对R25改善 **0.001370/0.001183/0.001641**。所以跨Subject当前证据的作用不依赖“新增metadata”这一唯一解释。

加入metadata点估计改善 **0.000595/0.000118/0.000543**，但证据强度不等于普遍必需。改成global centering则退化 **0.000791/0.000410/0.001170**：保留各Subject自己的平均成功率有作用。R36/R25的ridge在来源池0/1本来相同，差值为零不是两次独立正则证据；池2的R36尺度改善pair-equal而恶化Benchmark-equal，说明表示、尺度与数据权重有交互，不能把旧R36和R72差异全部归因于metadata。

R75使用support两半的预测关系与32个拟合共同斜率Bernoulli null对照；三个重叠来源池诊断19/16/15个Benchmark，去重后4个Benchmark触发异质性信号。该预设经验门槛允许一次partial实验，不是经过多重比较校准的总体显著性检验，更不等于辨识了真实能力结构。

部分共享采用固定目标 `SSE + ||beta_shared||² + 8 Σ||delta_s||²`，保留own均值并对subject偏移强收缩。三个来源池partial相对完全共享均小幅改善，随后冻结唯一候选做完整评价；没有外层挑系数。

### 完整外层结果

|方法/场景|Pair-equal ALC ↓|说明|
|---|---:|---|
|Frozen smoothed mean 2|0.198687363|原强均值基线|
|training-selected smoothed|0.201101132|原来源选择均值基线|
|R25 / 0 peer / 配置歧义回退|0.195123926|own-only|
|1 peer|0.193428907|冻结随机cohort，额外标签|
|2 peers|0.191976827|同cohort|
|3 peers，完全共享|0.190728761|同cohort，主要机制对照|
|部分peer缺失|0.192753990|预定缺失规则|
|peer覆盖不均|0.192681858|预定预算前缀规则|
|另一套冻结cohort，完全共享|0.190667308|敏感性分析，不按结果替换主cohort|
|**partial8**|**0.190547856**|固定收缩，主cohort|
|partial8，peer标签对应置换|0.196799019|破坏局部对应后收益消失|
|旧R72 pooled3|0.190683911|旧hash组成，保留原概率|
|旧R72 diff3|0.190885868|剔除同输入peer的历史强对照|

Partial相对**同一cohort完全共享**的ΔALC为 **−0.000180905**，cluster CI **[−0.000202247,−0.000038056]**，三个fold均改善，Benchmark-equal差异 **−0.000105771**。这是最直接的结构增量证据。

相对**旧R72 pooled3**的点估计差异为−0.000136055，但CI **[−0.000185504,+0.000273115]**跨零、Benchmark-equal略差、两个fold略差；此对照同时改变cohort，不能声称稳定支配历史R72。只对完全共享跑了第二套cohort，未证明partial在任意组成下稳健。

### Brier 与准确率（完整外层）

|预算|R25 Brier|partial8 Brier|R25准确率|partial8准确率|
|---:|---:|---:|---:|---:|
|0|0.250000000|0.250000000|43.0466%|43.0466%|
|1|0.222119481|0.222119481|66.6975%|66.6975%|
|3|0.198842814|0.196816347|68.9930%|69.1457%|
|7|0.181623410|0.176358241|71.2369%|72.6223%|
|15|0.169054986|0.159669390|73.7518%|76.0043%|
|31|0.157957879|0.145551645|76.3170%|79.1862%|

B0/B1不变是冻结算子的构造事实；B1即便可见peer标签也不使用其拟合斜率。

### 收益集中与严重回归

Partial相对R25改善0.004576070，CI **[−0.005238271,−0.000250061]**；18个Benchmark改善、10个不变、8个退化。约**96.98%**的pair加权改善来自 `alignment_faking`；去掉它并重新归一后改善0.000751736，不能用整体区间替代该子集的不确定性。

最大回归 `dave` 相对R25 **+0.006632866**。同cohort完全共享为+0.006927732，partial仅缓和约0.000295；旧R72为+0.002515300。分解恒等式 `ΔBrier = 2E[(p_old−y)Δp] + E[(Δp)²]` 按六预算加权为ALC后，dave新partial的误差对应项+0.001628050、概率位移平方成本+0.005004816。后者在JSON中叫 `variance_cost`，**并非严格的中心化方差或方差增量**。该分解说明当前修正未对齐误差且成本偏大，尚不能提供部署时可靠的回退判据。

相对两种均值基线的整体改善分别0.008139506和0.010553276，三fold、Benchmark-equal及CI等四项数值条件通过；但严重回归仍需解决，不能因事后给出解释就宣称晋级。相对training-selected均值的cellverse退化是原R25已存在的问题，dave则有新增共享负迁移。详细分解保存在 `R76/full/severe_regression_*`。

**认识：** 可共享的是当前任务内、控制Subject均值后的item–response局部关系，不是可直接套用的统一能力标量或共同成功率。额外peer越多，本次总体改善越大；不同Benchmark和组成并非单调安全。保留较强subject偏移收缩比完全强制共同斜率更合适，但当前增量很小，尚未解决负迁移识别。

## 6. 何时共享、何时回退？

- **Own-only**：peer不可见、没有当前own支持、匿名分组token缺失、完整可见Subject配置不唯一/缺失、可见重复证据有冲突，均关闭共享。不存在真实可见roster时不根据隐藏ID补造身份。
- **完全共享**：保留为简单强基线；来源证据不足以支持subject偏移时使用这个机制基线。当前结果不提供逐请求识别“共同斜率必然正确”的可靠规则。
- **部分共享**：本轮来源诊断满足预设条件时固定partial8，作为下一次独立确认的候选；不按外层Benchmark名或误差即时选择。不能把事后dave名单写成线上门控。

新CPU适配器读取当前 `input/labeled/visible_subjects`，只保留公共字段，忽略隐藏item/subject ID；匿名Benchmark token只作相等分组，不作模型特征。没有跨调用标签状态。不可见配置碰撞本身无法从相同可见字典识别，这是输入信息的边界。

适配器按可见配置排序选最多三个peer，R76评分用预先冻结随机cohort；当前 smoke 证明算子和回退路径可运行，**没有证明该输入流端到端重现R76分数**。默认mode保留pooled；partial为显式研究选项。真实Hosted合同仍未核实。

## 7. CPU资源与实际验证

- R74训练与来源评分 **32.52秒**，2 CPU线程；训练峰值内存未实测。一个fit-split的四个压缩学生head合计33,941字节。实际新item特征提取+学生：8条固定真实输入，warm单条中位65.29ms、batch8中位310.01ms；冷加载1.966秒、首次warmup5.560秒单列。缓存向量的1000条四维学生推理32.106ms。这些是本机microbenchmark，不是Hosted SLA。
- R75来源实验 **91.27秒**；R76完整预测 **794.73秒**、概率重计分 **52.06秒**，4 CPU线程。Windows观测R76进程lifetime峰值工作集 **3550.81 MiB**。原进程内ctypes记录0无效，已明确排除。
- 3-peer主场景B31平均额外peer标签约91.79，总可见标签约122.79，最大124；缺失peer约45.65额外标签。不是“总预算31”的算法。精确逐预算/Benchmark/请求成本在 `label_cost_*.csv`。
- 求解矩阵最大124×124，float64核矩阵本体123,008字节；算法约 `O(N²d+N³)` 时间、`O(Nd+N²)` 工作存储，另含MiniLM模型及整批评价存储。不能用核矩阵大小代替实际进程内存。
- R74保存4,394,670个概率值重计分，Brier/accuracy最大误差0；R76最大Brier误差8.88e−16、accuracy误差0。真实有限样本R25/R36/R72几何及primal/dual核对通过。
- 16项合成协议smoke，加新文本本地MiniLM CPU wrapper smoke通过；新文本请求涵盖共享、无roster回退、不完整配置及隐藏ID忽略。参考语义向量最大误差7.45e−8；R74另核对8条真实metadata输入。非全语料逐位一致，也非Hosted验收。

## 8. 原文、作者代码与适配范围

检索使用AnySearch，查询和提取记录保存在 `research_loop/literature/dual_track_demand/` 与 `R75/literature/`。搜索摘要只用于发现；关键依据为原文PDF/补充材料和固定commit源码。

- ADeLe：[Nature正式论文](https://www.nature.com/articles/s41586-026-10303-2)，Methods PDF pp.13–15、补充Table23；[作者仓库](https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation)，commit `816321f327dfe66910b72b514156aaa296b15f00`。复用原rubric与builder并逐条重建512个prompt。作者GPT-4o-0513、18维+UG和RF assessor与本次Luna四维/ridge不同；原生工具未暴露不可变模型快照/temperature，短判断依据也不声称复现作者CoT。详见[来源定位](../research_loop/literature/dual_track_demand/source_evidence.md)。
- Shared slope：[Evgeniou & Pontil, KDD 2004作者全文](http://www0.cs.ucl.ac.uk/staff/m.pontil/reading/mt-kdd.pdf)，Section2共同参数+任务偏移；[MALSAR作者仓库](https://github.com/jiayuzhou/MALSAR)，commit `fb9751594983df020ddc4f7e4a40520ee7c37989`，manual4.4和`Least_SRMTL.m`。原SVM目标、本次Brier平方损失、中心化和固定收缩的转换及源码0.5*SSE约定均写入[映射说明](../research_loop/SHARED_SLOPE_SOURCES.md)，不是复刻原论文实验成绩。

数据revision `54074453da7b9911be0f48e4c560be547bc84d21`；MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；code HEAD `fd5b06df51233392ea9d482732af8063f4222ec1`，工作区含未提交研究代码，精确内容以各轮冻结source hash为准。运行时Python3.12.7、NumPy1.26.4、SciPy1.13.1、torch2.9.1+cpu，本机Ryzen7 4800H/16GB。

额外外部收费API请求为0；原生Luna代理的计费/完整token用量未暴露，不能写成“模型调用费用为0”。未更改原始用户计划、旧模型或fallback；全局状态/ledger按任务要求追加新状态并单列为有意更新。

## 9. 下一步的价值判断

1. **No-Peer：停止本轮四维模型扩张。** 若重新开放，应先在合法来源验证教师定义/缺失上下文/文本窗口中哪一项造成失败，或获得人类校准与新的独立需求证据；单纯再标更多同类样本、换大MLP没有当前证据支持。R25保持。
2. **Shared-Peer：保留partial8做确认，不继续扫描收缩系数。** 最有价值的问题是仅凭当前支持证据，能否预先识别错误共享方向，并在独立数据/组成下减少dave式退化。必须先冻结新合同，不能用已见外层负例直接设计并宣称验证成功。
3. **工程：保持Hosted暂停。** 用户将来恢复该事项后，再用实际可见输入/peer选择规则回放冻结候选；本轮CPU adapter和合成smoke为准备材料。需求特征没有独立收益，未与共享模型组合。

本轮完成了条件允许的实验和停止决定；总体研究晋级与真实线上有效性仍未完成。
