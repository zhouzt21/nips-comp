# R77–R79 最小研究合同（plan_9）

2026-10-23。原始完整授权保存在 `../doc/plan_9.md`，不修改或代替原文。本文件补充可执行选择，必须在新结果评分前冻结。任务类型：假设驱动的机制诊断与公开数据探索评价。无 Hosted、提交、收费生成API、神经网络或partial罚项扫描。

## 问题、claim与反证

- R77：partial8相对同cohort pooled的微小增量是否跨组成保持？竞争解释为稳定结构、组成依赖、可重复异质性与纯幅度收缩。负方向在多个组成消失/翻转或新增严重回归会削弱稳定结构claim。
- R78：当前支持交叉预测风险是否在未见来源Benchmark上预示query共享收益？竞争解释为真实方向信息、均值相似、小样本噪声与不可辨识。跨Benchmark关联/预定分组不能重复则停止。
- R79：只有来源可靠性和固定最小策略有增量时，验证其是否优于partial与同平均强度收缩，并减少错误共享。平均分改善不证明严重回归被解决。
- 非目标：发现最佳随机peer集合、将已见dave外层误差写成线上规则、原论文数值复刻、新Benchmark盲测或证明真实Hosted标签可用。

## 固定数据与复现

复用R64 query/acquisition身份、R72当前标签与可见配置、R75选择记录及R76概率。数据revision54074453da7b9911be0f48e4c560be547bc84d21；MiniLM1110a243fdf4706b3f48f1d95db1a4f5529b4d41。R25特征几何/subject内中心化/own平滑均值/clip/partial8由原`shared_slope_model.py`调用，不重写替代。无peer/身份歧义回退原R25，不对单subject额外施加partial自由度。

外层36BM、5242pairs、item seeds0/1/2、最多64query items，原response variants/counts；B0/1/3/7/15/31，ALC权重.1/.2/.2/.2/.2/.1。各pair/seed等权，另报BM等权、fold、budget、准确率(p>=.5)、回归。10000次BM cluster bootstrap seed20261023；同BM所有pair/item seeds/cohorts一同重采样。多cohort平均先在同pair内平均，不能增加BM样本量。全部仍为自适应公开探索。

## R77：先补第二组成，再运行扩展场景

1. `alternative`阶段先完整评价seed7502 partial，与已存R76_alt3逐元素匹配信息比较。
2. `full`阶段冻结cohort seeds `[7501,7502,7701,7702,7703]`，直接调用原cohort_map；不查看新成绩后修改。seed7501/7502已存pooled与seed7501 partial显式复用。后三套同时预测pooled/partial。
3. 主7501复用peer0/1/2/3、hash7505缺失、uneven前缀n,n//2,n//4和强制歧义规则，每个场景同信息比较pooled/partial。0peer/歧义精确R25。B0/B1相同。
4. 幅度解释控制：每个outer fold只由对应R75来源预测估计每预算 `alpha=clip(E[d_pool*d_partial]/E[d_pool²],0,1)`，权重同来源pair/response。只用预测几何，不用query y；固定后用于五cohort `R25+alpha*d_pool`。它是source拟合的简单收缩，不是outer最优幅度。另报partial与pooled位移大小/方向及残差分解；平方位移成本不是中心化方差。
5. 组成稳健保留标准：至少4/5cohort点估计改善，cohort平均Δ的BM-cluster CI上界<0，且没有任一cohort新增BM Δ(partial−pooled)>=.005。否则根据方向翻转记录`partial_sharing_composition_sensitive`或证据不足，不扫8附近参数。不将最佳cohort选为部署规则。原Gate及真实Hosted条件保持。

## R78：来源支持交叉预测，先冻结再看结果

- 使用原R75来源fold各BM的固定最多8个唯一可见Subject名单、seeds0/1/2；来源fold与其outer disjoint。三来源池有重叠，综合描述按每BM首个合法source fold固定去重。基线几何沿用已冻结来源参数；不称全新的独立数据。所有新的规则/校准仅用来源，不读取目标outer query来拟合。
- 固定B15/B31；本轮不自动扩展B3/B7。场景为来源7501完整、7502完整、7501 missing、7501 uneven、7501 peer标签置换。cohort由来源选中的合法8个配置构造。query共享收益使用全部当前own标签；诊断拟合只能使用当次未遮挡own标签。
- 两个独立预定随机二分重复，每重复做互补两半，共4个mask。随机性由固定7801/7802、BM、完整可见subject key、item seed、预算决定，不用response或包含outcome的identity。所有方法用同mask；每个own支持标签每重复恰好一次作为heldout。隐藏own y不得进入均值、center、fit、正则或预测；peer仅用当前预算前缀，允许不同Subject合法相同item证据。
- 保存mask indices、fit/holdout身份、预测与二元truth、每次训练own mean和合法peer数，独立重算support Brier。没有训练来源标准化；原特征提取固定。
- 事前变量仅四类：`z_partial`、`z_pooled`为支持Brier(shared)−Brier(own)；`z_mean`为仅用按Subject等权peer平滑均值预测相对仅用fit-own平滑均值预测的Brier差（两者都无局部方向）；`z_combined`为partial斜率+同一peer均值截距相对R25的Brier差。另记录own/peer均值绝对差；combined无额外拟合参数。后两项为机制控制，不自动认为接近均值就可靠。
- 预定可靠性组：z<0、z>=0，零归不可靠。target为full-support query的partial−own（pooled另单列）。报告grouped Spearman、方向balanced accuracy/普通accuracy、每BM的rho与组内真实风险、mask重复一致性、有效标签、alignment/rest、dave、missing/uneven/perm。
- 关联统计先平均同pair的两个预算及item seeds，再算各BM内Spearman；常数/不足3pairs的BM标不可估，不填0。两种权重同时报告。重复mask用于稳定性诊断，不当独立样本。
- R79诊断入口：候选顺序固定partial、pooled、mean、combined；只在主cohort的来源数据选择第一个满足者：至少2/3来源fold可估BM平均rho>0；去重BM平均rho的95%cluster CI下界>0；预定z<0组相对z>=0组的query Δ在pair和BM汇总均更低，且去除alignment仍同方向；两个组各至少10distinctpairs、至少5BM。另外报告其他场景的反例，不把其漂亮切片用于选择。任何未达项为停止，不提高模型复杂度。

## R79：仅条件触发，先来源策略再完整外层

若R78无指标通过，不创建R79实验目录，只在DECISION记录`support_consistency_not_predictive / R79 condition_not_met`。若通过，固定二值`g=1[z<0]`，只控制B15/B31，其他预算保持原partial/R25规则。无阈值扫描、MLP或Benchmark名字。

在每个source pool按Benchmark固定hash两半进行策略验证：z无拟合，matched constant g只用另一半的平均共享开启率拟合，应用于当前验证half；另保留same-cost常数作为离线诊断，不能从外层调参。来源策略进入full的条件：至少2/3池同时优于partial和matched-mean常数，综合pair/BM Δ均负，去alignment仍同方向。若失败停止。若通过只冻结一个候选，outer每fold常数由全部合法来源拟合；完整5242×3×6，report错误关闭成本与严重回归。其他cohort不用于选最有利门控。

## 资源、检查与停止

CPU最多4线程，优先R77顺序执行；目标R77<=60分钟、R78<=30分钟、条件R79<=45分钟，时间为估计，不是结果。每阶段超过目标先记录已完成coverage和真实速率，不裁减必要protocol冒充完成。实测Windows process lifetime peak working set，拒绝0伪内存。保存命令、git HEAD+dirty source hashes、环境、cohorts、raw概率、query counts/successes、计分与独立replay。原R72/R76/两fallback与R62的183冻结资产保持。

数值检查包含旧7501/7502回放、无peer/歧义/B0/B1回退、mask heldout y变动不影响fit预测、同mask比较和当前预算前缀断言。源码冻结后若实现修复必须保留失败批次/合同，不能隐匿或事后调参。运行超过60分钟写回D:/todo；AI时间不记为用户投入。
