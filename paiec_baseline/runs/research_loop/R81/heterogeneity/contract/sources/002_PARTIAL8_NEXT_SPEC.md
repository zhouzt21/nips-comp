# R80–R83 最小研究与执行合同

2026-10-23，原始完整授权见`../doc/plan_10.md`。本文件在新评分前记录实现选择；不替代或改写用户原文。贡献类型：机制归因与收益符号可辨识性审计，公开数据探索性研究，不作新SOTA/Hosted claim。

## 冻结问题与反证

R80比较样本量失配、有限样本风险噪声、support/query表示差异、真实异质性。若LOIO/5fold只改变相关性而不能改进收益符号/有害组判别，不进入R82。R81比较结构化收缩与Subject-specific方向：只有匹配平方成本后的独立收益、真实偏移可重复性等一致支持，才强化方向主张。合成共同斜率下也改善会削弱“改善必然来自真实异质性”的推断。

不新增神经网络、不扫描阈值/Partial8罚项、不中途改cohort，不恢复Hosted。R80/R81必须完成；R82/R83按下述条件执行。

## 复现与共同评价

沿用R75/R78合法来源名单、R64 query身份与response counts，R72 acquisition，R76真实算子。data revision54074453da7b9911be0f48e4c560be547bc84d21；MiniLM1110a243fdf4706b3f48f1d95db1a4f5529b4d41。Python3.12.7/NumPy1.26.4/SciPy1.13.1/torch2.9.1+cpu，CPU<=4线程。源码commit+dirty快照、输入SHA和命令在各contract。原R1–R79及fallback保留，全局state/ledger仅追加。

R80三个来源池与outer disjoint，来源池之间重叠；描述按每BM首个合法pool去重，不能把多pool当独立样本。R81完整36BM/5242pairs×3item seeds，五cohort7501/7502/7701/7702/7703，六预算0/1/3/7/15/31；ALC权重.1/.2/.2/.2/.2/.1，原64query item cap及重复response计数。BM cluster bootstrap10000次seed20261023，保留pair/seed/cohort依赖；样本/feature统计不作为Benchmark名称预测特征。

## R80：固定主cohort来源审计

主7501、R78来源每BM至多8个唯一可见配置、seeds0/1/2、B15/B31。先运行main，不扩B3/B7。来源query full-support风险/概率与R78 main逐元素核对。

估计器：

1. half：原R78两个随机二分重复，7801/7802及原split_masks；同group无重复时逐元素回放。若support重复item出现，整item分组，不混用重复响应。
2. LOIO：按support item ID作审计分组，同一item全部holdout；item ID不输入模型。每item只留出一次，重复response整体离开fit。
3. grouped 5fold：两次固定随机group分配，8001/8002、BM相等token/可见subject配置/item seed/预算；group打乱后round-robin五折，所有方法同划分。不是调K，不扫折数。

每mask拟合own/pooled/partial后同时预测heldout support及固定query，保留全部概率。隐藏标签不进入fit均值/center/solver；逐实例首mask翻转heldout y应使预测不变。peer只用当次合法预算前缀。保留support ID/group、fit/hold indices、peer前缀、query counts/successes。

关键失配分解：`z_cv−Δfull_query = (z_cv−Δcvfit_query)+(Δcvfit_query−Δfull_query)`。Δcvfit_query是每mask模型在同query上的风险均值，不是先平均概率再计风险。该分解的query结果仅离线诊断。另记录线性核support/query均值距离（固定特征几何、均无标签）与估计误差的BM内关联，相关不证明分布失配是原因。

统计：分别按预算报告排序rho、符号balanced accuracy/accuracy、z−Δ偏差/MAE/RMSE、mask重复符号稳定性。request单位pair×itemseed×预算；另报seed平均pair诊断，不能混淆运行时可用证据与跨seed聚合。中性|Δ|<=1e−12不计入二元方向BA。

不确定性：每请求将OOF逐item风险差按item聚合后做1000次item bootstrap（固定预测，seed8005+可见配置/预算）。这是条件评分敏感性区间，因拟合重叠不宣称普适95%CV风险置信保证。LOIO无mask随机性，不能将单个确定值称零估计方差。另在每BM来源首个可见Subject、seed0、B15/B31做32次own支持item bootstrap重拟合，以原观测item为group并在out-of-bag own items评分；peer固定前缀。这是有限经验分布下的refit稳定性，不提供新独立标签或可推算无限预算需求。

## R80 → R82 进入规则（评分前固定）

候选顺序LOIO、5fold。仅一个估计器达到以下条件才继续：

- B15和B31的request级符号BA均>=.60；至少一预算相对half提升>=.03，且BM-cluster配对ΔBA的95%下界>0；另一预算不降低BA。
- 在上述主要改善预算，至少2/3合法source pools相对half的BA同方向改善。
- 主改善预算中支持bootstrap下界>0的“有害”请求至少覆盖10distinctpairs、5BM，且其实际full-query Δ的BM-cluster均值CI下界>0。支持区间的覆盖有效性仍由来源外校准验证，不能凭名义95%直接部署。

未通过则`sharing_sign_not_identifiable`，不创建R82/R83实验目录；不是声称不存在任何风险信息。该门槛是本项目可操作证据标准，不是文献保证。R82若触发，每outer仅用其合法pool嵌套fit估计预算条件的线性风险校准与BM残差区间；不从union选好变量再回填outer。R83只能在嵌套来源有害组风险确认且策略优于partial/匹配收缩后执行；候选/不确定性具体合同必须在其评分前另冻结，不根据outer修改。

## R81-A/B：冻结预测的方向与幅度

重用R77五cohort保存的所有概率；为审计clip前值，仅新增不改变代数的raw算子包装（原center/kernel/coef/base），逐元素验证clip(raw)=原冻结概率。B0/B1、无peer保持原R25。

每query定义d_pool=p_pool−p_own，d_partial=p_partial−p_own。报告同向、反向、同向缩小/放大、原pool为0而partial新增修正、raw/clip前后边界比例；阈值1e−12为数值零，不是学习阈值。

对照：

- 原R77 source alpha投影收缩，参数不重拟合。
- point-direction：`clip(p_own+sign(d_pool)*abs(d_partial),0,1)`。保留partial逐点幅度和pool逐点符号，边界/零pool方向可能无法匹配，明确计数和成本。
- exact-cost pooled：`clip(p_own+a*d_pool,0,1)`，每request/预算用二分求a>=0，使response加权平方位移尽可能等于partial；只用预测与counts，不看successes。报告不可达成本。此控制使用整组query/response权重，**仅离线机制诊断，不作合法部署方法**。匹配平方范数仍不能唯一分离向量方向、逐item相对幅度和真实Subject能力。

各对照按原六预算保存概率与重新计分；给出pair/BM/fold/CI、alignment/rest、最大回归、平方成本、`2*residual*correction+correction²`恒等分解。跨五cohort先在pair/seed平均风险，非概率ensemble。

## R81-C：偏移与已知机制

用R77事前固定的每BM/cohort首四人组中的主7501组，seed0，B15/B31，不按结果选group。真实实验：两个R78二分重复、互补mask；Partial核对应的own偏移项`(q-center)@X_own.T@coef_own/8`与其共享成分分开。检验重复mask偏移方向、heldout own残差增量；固定peer标签置换8103做负控制。只用合法已知标签，query outcomes不作选择。

合成实验复用同组真实feature几何/当前预算。固定两个世界：fitted共同latent线性斜率；相同共同斜率加各Subject独立高斯方向偏移（全anchor标准差固定.15）。共同斜率在全anchor上标准差固定.15，各Subject截距为本预算own平滑均值；真概率clip至[.01,.99]并记录clip率。共同latent斜率不意味着clip后概率完全线性，明确该限制。

每组/预算32个Bernoulli support重复，seed8105；query真实概率已知，用expected Brier检验own/pool/partial/成本匹配，不能当真实PAIEC成绩。保存生成参数、支持label、query真概率与预测、重复指标。BM聚合为主，rep不当独立BM。合成仅校准机制预期，不能证明真实能力恢复。

## 资源与停止

目标R80<=45分钟、R81完整诊断<=60分钟、R81C<=20分钟，均为事前估计；超过时记录真实coverage/速率，不裁剪冒充完成。预计全任务可能超过60分钟，Dtodo已登记；每<=60分钟写回。保存实际CPU执行墙钟与Windows进程峰值、错误批次、原概率replay。任何实现修复保留旧合同/失败记录。最终报告Q1–Q5、正确率、机制限制和条件未触发原因。
