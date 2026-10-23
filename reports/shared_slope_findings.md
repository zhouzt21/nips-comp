# R75–R76：共享斜率、随机 peer 组成与部分共享

日期：2026-10-23。范围：`doc/plan_8.md` Track B，纯 CPU，**conditional_simulation**。没有恢复 Hosted 审计，没有正式提交，也没有确认真实运行时可以提供这些 peer 标签。

## 1. 执行结论

完成来源诊断、受控消融、一次固定 partial pooling、全部 5,242 pairs × 3 seeds × 六预算、输入适配器及本地 smoke。

**最强的本轮预注册主 cohort 候选为 R76 partial8，ALC 0.190547856。** 相对同一 cohort 的 fully pooled 0.190728761，改善 0.000180905，Benchmark-cluster 95% CI 为 [-0.000202247, -0.000038056]，三个 fold 均改善。这支持保留该结构作为研究候选，不需要继续扫描个体偏移系数。

**它没有解决严重负迁移，未通过完整 Research Success Gate。** `dave` 相对 R25 仍恶化 0.006632866；相对旧 R72 pooled3 的总体改善还混合了 cohort 更换，CI 跨零且 Benchmark-equal 与 fold0/1 更差。因此不能宣称已经可靠替代旧 R72，更不能部署或当作 Hosted 成绩。

完整数值见 [shared_slope_tables.md](shared_slope_tables.md)，路径与命令见 [shared_slope_reproduction.md](shared_slope_reproduction.md)。

## 2. 实验合同与输入边界

- 使用 R64 冻结的 query/support 身份、全部 response variant、counts/successes 和 seed0/1/2；预算为 0/1/3/7/15/31，ALC 权重 .1/.2/.2/.2/.2/.1。
- 只使用当前 own/peer acquisition 标签计算预测。query outcome 只进入离线评分；cohort、missing mask 和标签置换 RNG 不使用 outcome、评分 identity hash 或测试表现。
- 主 cohort seed7501、第二组成 seed7502，在每个 Benchmark 的完整可见唯一 Subject 配置上预先随机分组，每组最多四个 Subject。重复完整配置沿用 R72 的排除规则。内部 ID 只定位离线冻结数据，不作为预测特征或线上身份依据。
- 0/1/2/3 peers、随机缺失、预算不均、第二组成和强制身份歧义 fallback 全部预先冻结。没有按外层表现选择有利 cohort。
- R72 same3/pooled3/diff3 保留的是**旧 hash cohort**，从原始概率重用；它们没有被冒充为新随机组成。
- 共享分数使用额外标签，不能与 own-only 当作相同总标签预算的算法排名。

原始合同：[SHARED_SLOPE_CONTRACT.md](../research_loop/SHARED_SLOPE_CONTRACT.md)。

## 3. 原文与作者代码依据

AnySearch 完成检索与仓库发现，随后读取作者 PDF 和作者仓库代码：

1. Evgeniou & Pontil，KDD 2004，*Regularized Multi-Task Learning*，Section 2、equations (1)–(10)：共同参数加任务偏移，以及偏移收缩。[作者原文](http://www0.cs.ucl.ac.uk/staff/m.pontil/reading/mt-kdd.pdf)
2. Zhou/Chen/Ye 的 MALSAR 作者仓库，commit `fb9751594983df020ddc4f7e4a40520ee7c37989`：手册 4.4 节与 `Least_SRMTL.m` 的 mean/graph quadratic penalty。[作者代码](https://github.com/jiayuzhou/MALSAR/blob/fb9751594983df020ddc4f7e4a40520ee7c37989/MALSAR/functions/SRMTL/Least_SRMTL.m)

本轮为**机制复现与 PAIEC 适配**：原 KDD 主实验使用 SVM/hinge loss；此处采用 centered squared loss、own smoothed mean、clip 和固定偏移罚项8。MALSAR 实际源码为 `0.5*SSE`，与手册简写不同；已经换算正确系数并用 primal/dual/graph penalty 等价性检查，最大预测误差 7.77e-16。没有声称复现原论文的基准分数。

具体映射、版本和文献 hash 见 [SHARED_SLOPE_SOURCES.md](../research_loop/SHARED_SLOPE_SOURCES.md) 与 R75/literature/manifest.json。

## 4. R75：什么构成了共享收益

每个 outer fold 仅使用其合法 source Benchmarks，每 BM 最多8个完整配置唯一 Subject。实际评分 BM 数为 **25/23/22**，可进行四人斜率诊断的 BM 数为 **19/16/15**；与各自 outer heldout 交集均为0。1146条 source 轨迹是重复来源的暴露数；按固定首个合法 source context 对 Benchmark 去重后为 **573轨迹、191 pairs**。不能把三个重叠 source pool 当作三个独立数据集。

以下是去重 source 的配对 ALC 差，均为方法减对照：

|干预|ΔALC|95% cluster CI|结论|
|---|---:|---|---|
|语义+metadata：pooled−own|−0.001572158|[−0.002423617,−0.000760986]|同一表示下，共享标签有收益|
|仅语义：pooled−own|−0.001399997|[−0.002202391,−0.000691373]|共享收益不依赖 metadata 才存在|
|pooled 加 metadata|−0.000514629|[−0.001149584,+0.000007801]|有弱增量，不能断言已经稳定证明|
|global centering−Subject内centering|+0.000537407|[+0.000072797,+0.001005954]|去除 Subject 内中心化会损失收益|
|partial8−fully pooled|−0.000047830|[−0.000096251,−0.000003286]|来源上存在很小的增量|

正则因素必须单独解释：fold0/1 原 R36 ridge=4，R25 semantic scaling 对应 ridge=4，因此换到同一特征单位后两者均为 ridge1，数值完全相同。**去重表中的 ridge 差为0不是正则强度无效的证据。** fold2 才形成实际 ridge1 vs ridge4 对照：语义+metadata 的 pair-equal Δ=-0.000736403，但 Benchmark-equal Δ=+0.000339786，CI跨零；仅语义也类似。它未给出稳定的全局改参依据，本轮保持冻结的 R25 正则配置。

解释：R72 的主要机制是其他 Subject 当前标签对局部 item–response 方向的补充。Subject 内中心化避免直接把不同能力水平混在同一斜率里；metadata 可能补充表示，但其独立证据弱于共享本身。不能把旧 R36 与 R72 的总体分数差全部归结为某一特征或 ridge。

## 5. 异质性与 partial pooling 的实际证据

支持标签 B31 分成16/15两半；在相同无标签 anchor 输入上，检验各 Subject 的半样本斜率预测差是否可重复。与拟合的共同斜率、独立 Subject 截距所生成的32次 Bernoulli null 比较。

三个 source 选择环境中，超过 null95 且半样本可重复的 BM 数为4/2/2；去重只有 **beavertails、bertaqa、cmmlu、dave**。三个环境的平均 H 也越过预注册 null95，故执行一次固定 partial8。来源 ALC 分别改善 0.000056679 / 0.000039939 / 0.000122923，允许进入完整探索评价。

这是**小样本、模型依赖的探索入口**：32次 null 的分位越界不等价于经多重比较校正的显著性；fold2 两个 BM 的离散上尾 p 为0.0606。拟合 null 的正确性、条件 Bernoulli 独立性及每 BM 仅一个四人组都限制解释。未发现异质性也不能证明各 Subject 斜率完全相同。

固定模型求解 `SSE + ||beta_shared||² + 8 sum ||delta_subject||²`；保留 own 平滑均值并对每个 Subject 的 X/y 分别中心化。没有使用 query 标签拟合参数，没有扫描8附近的系数。

完整评价中，partial8 对同 cohort fully pooled 的增量：

- Pair-equal ΔALC **−0.000180905**；Benchmark-equal Δ **−0.000105771**。
- 三fold Δ：**−0.000055299 / −0.000059981 / −0.000203741**。
- peer 标签置换后 ALC **0.196799019**；正确关系相对置换 Δ **−0.006251162**，CI [−0.006900443,−0.001882192]。这支持当前 item–response 对应关系的重要性。
- partial8 对 R25 的收益约 **96.98%** 来自 alignment_faking 的 pair 权重贡献；partial8 相对 fully pooled 的额外收益约 **93.79%** 也来自该 BM。它不是均匀分布于所有任务的提升。

## 6. Peer 数量、组成、缺失与不均衡

|场景|ALC↓|B31 平均额外peer标签|B31 平均总标签|
|---|---:|---:|---:|
|0 peer / R25|0.195123926|0|31|
|1 peer|0.193428907|30.7043|61.7043|
|2 peers|0.191976827|61.3022|92.3022|
|3 peers，主组成|0.190728761|91.7936|122.7936|
|随机缺失部分peer|0.192753990|45.6543|76.6543|
|peer预算不均（31/15/7）|0.192681858|52.3949|83.3949|
|3 peers，第二组成|0.190667308|91.7936|122.7936|
|partial8，主组成|0.190547856|91.7936|122.7936|

1→2 peers 的 Δ=-0.001452080，CI [−0.001673419,−0.000013057]；2→3 的 Δ=-0.001248066，CI [−0.001411739,−0.000195370]。主 cohort 的 0/1/2/3 场景总体随信息增加而改善；第二 cohort 仅验证 3-peer 完全共享的组成敏感性，未做其完整数量曲线或 partial pooling。这不意味着任意 peer 集合、任意 Subject 都有收益。

missing 和 uneven 都仍优于 R25，CI 上界分别 −0.000047002 和 −0.000005873；它们也明显不及完整3 peers。预算不均不只是平均标签数变化，还改变了共同方向的有效估计权重，因此不能从这两个场景精确分离“数量”与“覆盖质量”的因果份额。

第二组成相对主组成 Δ=-0.000061454，CI [−0.000440306,+0.000017434]，没有稳定优越的证据；不据此选 seed7502。B0/B1 与 R25 完全相同，因为各 Subject 只有0/1标签时中心化局部斜率为0；本轮方法不声称改善低预算先验。

## 7. 严重回归、旧 R72 比较与 Gate

`dave` 的 ALC：R25 **0.238811814**；旧 R72 pooled3 **0.241327114**；主随机 fully pooled **0.245739546**；partial8 **0.245444681**。部分共享减轻了主随机组的退化约0.000294865，剩余退化仍为 **+0.006632866**。

精确 Brier 分解相对 R25：`d=p_new-p_R25`，差值为 `2E[(p_R25-y)d] + E[d²]`。dave partial8 的第一项 **+0.001628050**，第二项 **+0.005004816**。两项均为损失贡献。JSON 中第二项历史字段名 `variance_cost` 实际是**概率位移平方成本**，没有减去均值，不能解读为中心化方差或纯方差增量。旧 R72 的第一项为负而新组成变正，提示 peer 组成能改变修正方向；这仍是公开数据的事后机制描述，不能据此按 Benchmark 名称设置在线白名单。

partial8 对旧 R72 pooled3 的 Δ=-0.000136055，但 CI **[−0.000185504,+0.000273115]**；Benchmark-equal **+0.000169721**，fold0/1 分别 **+0.000140851/+0.000053171**。这是**不同 cohort**比较，不是 partial 的纯因果效应。对 R56_segment 的 CI 也跨零，fold0 更差；不能仅凭总体 ALC 宣称全面替代既有强方案。

partial8 相对两种均值基线的 ≥0.005、三fold、Benchmark-equal 和 cluster CI 四项数值条件通过，但 severe regression 条件尚未解决。对 training-selected mean，cellverse/cruxeval/dave/egoschema 有 ≥0.005 回归，详见原始 gate.json 与回归分解。因此本轮结论为：**保留研究候选；完整 Gate 未通过；不替换冻结 fallback、不自动部署。**

## 8. 可运行输入适配器与资源

使用 [shared_slope_adapter_cpu.py](../research_loop/shared_slope_adapter_cpu.py)。它调用 root 的 `CPUItemFeatures` 从本地 MiniLM 在 CPU 编码新文本，纯特征/权重状态不保存标签；过滤匿名当前 Benchmark token 只用于同组隔离，不使用 Benchmark 名称、隐藏 item ID 或隐藏 Subject ID。

- 当前明确可见的 Subject roster 缺失、全配置重复/不完整、同可见输入重复或标签冲突：own-only fallback。
- flat labeled 无法证明两个相同可见配置是否为同一个真实 Subject；适配器不会借隐藏 ID 消除这种歧义。
- 无 peer / own B0：回退 own-only；`own_predictor` 可注入另一个单独验证的 No-Peer 模型。
- 适配器按当前完整配置排序取最多3 peers；R76 评价使用冻结随机 cohort。因此 smoke 证明输入边界和算子行为，**不证明适配器端到端复现 R76 的 cohort 分数**。
- 旧 `shared_slope_adapter.py` 的 `make_r25_encoder` 仅能查历史 cache，不可用于新 item；它保留作为首轮 smoke 证据。新 item 必须使用 `shared_slope_adapter_cpu.py` 的 factory。

16项首轮合成 smoke 通过；新文本 CPU smoke、无 roster fallback、incomplete-config 下隐藏 ID 无影响也通过。真实 source 上新求解器与 R72/R36 原公式最大误差2.22e-16。全部是本地验证，不是 Hosted 验收。

实际资源：source预测/诊断91.27秒；完整 R76预测794.73秒；独立重计分52.06秒。完整批处理 OS lifetime peak working set **3550.81 MiB**；原内置 ctypes 计数返回0，已作为无效仪器值拒用，source进程峰值无法恢复。

单请求最大 own31+peer93=**124标签**，特征维数640，dual kernel 124×124，float64 单矩阵123008 bytes；求解复杂度约 `O(m²d+m³+qmd)`，还需模型、feature与输出内存。额外 CPU 新文本实测：2线程、124标签单请求 **2.381秒**，进程峰值 **384.58 MiB**；0peer31标签首次请求 **4.635秒**，含首次算子初始化，不能据这两个单次时长比较0/3 peers的稳定延迟。模型构造0.537秒，不含所有Python导入；这些短合成文本结果不能外推真实长输入/Hosted时限。

## 9. 对共享问题的理解与下一步

可共享的不是其他 Subject 的总体正确率，而是同一当前任务环境内、经 Subject 内中心化后的局部 item–response 关系。完整共享利用更多标签降低估计的不稳定性；部分共享允许自己的支持标签修正共同方向。本轮同 cohort 的小幅改善支持这一机制，但没有给出“任意 peer 都安全”的结论。

何时使用：没有可信 peer 身份/当前可见性证据就回退 own-only；共享信息契约成立时 fully pooled 是强简单对照；有来源异质性与同信息配对收益时 partial8 是值得保留的研究候选。当前不能依据隐藏 Benchmark 身份或观察到的外层成绩动态挑 peer/强度。

最有价值的下一步是**只基于合法 source 和当前 support，预注册一个能识别有害共享并回退的风险诊断**，检验它能否减少 dave 类回归且保留收益。不要继续选幸运 cohort，不扫描个体罚项，不把尚未验证的需求特征直接与 shared 模型混合。所有公开 Benchmark 已反复使用，本轮证据仍为探索性。
