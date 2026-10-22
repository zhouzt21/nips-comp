# Research operators v2

## Loop v3 当前操作

| Round | 操作链 | 证据与转向 |
|---|---|---|
| R32 | Audit Measurement → Confound-Isolating Diagnostic | 去cap差值效应估计+0.000019，区间约±0.00018，不能解释剩余0.00144；保留全量与分层结果的证据边界 |
| R33 | Audit → Limit fixed prediction family | ≤4放大oracle潜在收益0.00939，仅收缩0.00415；支持进一步判别，不能部署oracle |
| R34 | Reframe → Adapt by Conditioning → matched coefficient control | 新增0.000049，聚类区间跨零，未晋级 |
| R35 | Diagnostic → Decompose budget/objective | 高预算关联增强但来源预算偏好不一致；不支持继续尺度网格 |
| R36 | Separate prevalence/item exchangeability → controlled factorial | 共享方向胜同正则own，但未稳定胜原强own；均值共享有异质性 |
| R37 | Uniform marginals → explicit independent partition → posterior structure | 新分区方法输原own/旧hierarchy；停止该结构模型 |
| R38 | Input relation audit → matched correspondence destruction | 有配置关联但邻居均值代理两折退化；不开发该邻居方法 |
| R39 | Audit objective → curvature-matched nonlinear solver | 对R25差值+0.00000568、区间跨零；停止loss/link变体 |
| R40 | Input-only partition → count evidence → correspondence controls | 三折输R25，未胜随机方向；停止分组网格 |
| R41 | Audit deterministic representation → nested equivalence classes | 经验碰撞代价0.001076；主导benchmark无精确碰撞，不推论噪声下界 |
| R42 | Condition representation on current support context → remove shared template | 来源内进入条件失败；不恢复身份字段或扫边界 |
| R43 | Learn evidence reliability → query/episode information ablation | 两组内层选择epoch0；不进入外层，不扫MLP结构 |
| R44 | Audit covariance assumption → representation/variance factorial | 固定对角语义三来源均退化；停止方差分配网格 |
| R45 | Complementary estimators → exact Brier identity | 固定平均来源内通过，进入R46；不是OOD证据 |
| R46 | Freeze combination → paired outer replication | 对R25 pair等权退化+0.000909，尽管23/36改善；不换主指标 |
| R47 | Audit information → distinguish storage/runtime contracts | 训练资产存在；公开起始包TLS错误，运行时像素契约未确认 |
| R48 | Audit oracle optimism → disjoint calibration/assessment panels | 主导full池空间保留，跨BM一折退化；额外标签不可晋级 |
| R49 | Separate direction/calibration roles → same-split controls | 三来源均输完整R25；停止分割及幅度网格 |
| R50 | Audit omitted official input → scoped repair ceiling | 仅10/5242pair；即使其误差归零最多增益0.000403，不足补门槛 |

六字段记录：`ITERATION_V3.md`。下方v2内容为历史记录。

<!-- current-v2-r29 -->
## R22–R29 操作及退出依据

| Round | 操作链 | 结果/停止条件 |
|---|---|---|
| R22 | 隔离覆盖与每周期计算 → 真实重训 → OOD检验 | 新MLP仍输均值；停止该池扩张 |
| R23 | 分解subject/item → 软约束 → mean-only控制 | 两折退化；停止该近似 |
| R24/R25 | 审计字段 → 信息/正则归因 → 分块惩罚 | 小幅增益三seed复核；不晋级 |
| R26 | 固定预测器 → 流式获取 → mean/item分解 | CI跨零；不开发正式labeling.py |
| R27 | 核对理论目标 → 解析风险检查 → 预算比较 | minimax性质不迁移为平均Brier优势 |
| R28 | 找回E3负证据 → 先验对称化 → 同矩控制 | 仍输固定均值；停止泛用shape拟合 |
| R29 | 分离方向与子空间 → beta二阶统计 → 内层分组选择 | 真实/打乱均全关；不扫rank/rho |
| R30/R31 | 审计覆盖 → 区分外推/实测 → 冻结模型扩大实际pair | 全5242pair改善0.00356仍未达门槛；不称新盲测 |
<!-- end-current-v2-r29 -->

| Round | Operation / object | Input -> output and dependency | Falsifier / boundary | Status |
|---|---|---|---|---|
| R10/V2-01 | Audit score-to-mechanism claim | R7 positive ALC -> rival mean/discrimination explanations | R7/R8 alone cannot identify learning-source cause | Completed |
| R10/V2-01 | Confound-isolating diagnostic | Fixed predictions -> exact mean/centered components | Identity residual3.1e-16;93.6% net gain centered item component | Completed |
| R10/V2-01 | Correspondence intervention | Same labels/counts/kernel -> shuffled associations | Shuffling removes gains, supporting local information use within this setting | Completed |
| R11/V2-02 | Separate population and subject rates -> conditioning | R9 heterogeneity concern -> fixed/adaptive hierarchy | Three-seed adaptive-vs-fixed CI crosses0 | Completed; no complexity justification |
| R12/V2-03 | Audit attribution -> match coefficient | R11 improvement -> strength-matched plug-in control | Hierarchical integration does not outperform matched simple estimate | Completed; simplify |
| R13/V2-04 | Decompose -> differentiated treatment -> factorial | R10 item effect plus R12 simple prevalence estimator -> four cells | Components nearly additive; conditional gate not met against peer | Completed |
| R14/V2-05 | Audit uncertainty unit -> composition stress | Different itemseeds do not vary peers ->3cohort assignments | Conditional gains persist, strongpeer increment small | Completed |
| R15/V2-06 | Audit identity coverage -> relative capability fit | Source overlap -> benchmark-centered ability correction | Only188supported targets; rank/correction fails | Completed; reject scalar axis |
| R16/V2-07 | Audit measurement -> controlled source formation | Event-label semantics -> higher-direction source-only filter | Filter fails; semantics cannot explain all prior error | Completed |
| R17/V2-08 | Oracle bound -> eliminate repair family | FrozenR7 -> best possible convex attenuation using evaluation labels | Maximum0.002131gain<0.005 | Completed; stop attenuation route |
| R18/V2-09 | Audit omitted input -> matched window restoration | 1783/3600items exceed126tokens; freeze encoder/solver | 256semantic and lexical both fail to improve128 | Completed; no window expansion |
| R19/V2-10 | Audit unused permitted item keys -> correspondence control | Exact peer matches cover12.9%at31; source-only strength selection | Three-seed improvement0.000556; shuffling loses it | Completed; small conditional signal |
| R20/V2-11 | Factor subject mean -> expand current slope evidence | Fixed semanticvectors, centered within groups; pooled vsown | Three-seed pooled0.202835 vsown-selected0.203918; permutation removesaddedgain | Completed; gate not met againstpeer |
| R21/V2-12 | Exact feature-space lift -> equivalence/control | Same vectors/data, tensor-power4kernel vslinear | Increment−0.000384butonefoldworse,CIcrosses0 | Completed; no degree sweep |

Rejected: larger MLP, arbitrary fixed mixing sweeps, or expanding hierarchy grids. R12 coefficient control was an attribution test, not claimed innovation. The generic `Delegate` operator in the reference is a solver-design concept; no external expert or sub-agent review is claimed.
