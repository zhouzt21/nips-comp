# R69–R72 最小研究合同（2026-10-23，执行前）

原始、完整用户计划保留于 `doc/plan_6.md`，该文件不改写。这里记录可执行选择与证据合同，非替代原文。

## Question / claim / non-goals

探索性实证：R64 已有负 residual alignment 能否通过来源可迁移的预算规律或合法可靠性变量抵消平方位移成本？仅当前已揭示 peer 标签能否在强 R25 上增加独立信息？贡献类型为机制诊断与可复核小算子，不声称新算法或无偏新盲测。

H70：来源 OOF 估计的预算收缩可改善外层 R25，且优于统一幅度收缩。H71：语义/词法最近支持项一致性可在控制预算后增加效益。H72：独立 peer 当前响应的同输入残差或共享局部方向，可超过 own-only 与仅增加标签的均值控制。足以推翻：外层无增益、CI跨零、置换不损失、随机等能量控制相当、仅特定覆盖情景有增益。ADeLe、大模型、网格搜索、正式提交均非目标。

## 数据、计分和隔离

- 固定 measurement-db `54074453da7b9911be0f48e4c560be547bc84d21`；MiniLM `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；CPU 4线程，无新增 embedding/MLP。
- 沿用 `reproduction_v4/source_fold*.json` 的来源/外层 Benchmark 划分。source seed73、每 BM 至多25pairs、64 query items，重复 response 全计数；完整外层 5242pairs×seeds0/1/2，固定R62/R64 query variant identity。
- 主指标 pair-equal 六预算 Brier ALC，权重[.1,.2,.2,.2,.2,.1]；BM-equal、fold、seed、10000次BM-cluster bootstrap（20261022）、accuracy≥.5、alignment_faking贡献、最差回归齐报。
- 对每个外层fold，不读取其query标签来训练、选择、标准化或校准。来源池彼此重叠，只作开发数据。公开外层已反复使用，当前结果为一次冻结探索，不能称全新确认。
- R64 source 是来源开发预测，原semantic ridge由整个来源池选择，不能直接当严格OOF。重建每个固定inner-half的预测，ridge仅用另一half，从原R25 grid[None,16,4,1,.25]选择（pair等权）。OOF行只来自未参与该ridge选择的BM；最终校准拟合全部OOF行。
- 来源交叉验证另行严格执行：每half作heldout，ridge和α只由另一half拟合；允许fit内选ridge和拟合α，但不把其fit成绩当验证。不会使用另一half参与ridge选择的OOF行去验证该half。

## R70：只一次预先指定的算子族

p=R25，d=(template_lexical−p)/2。A=pair平均 Σ(c p−s)d/Σc，C=pair平均 Σc d²/Σc。
- independent：α_n=clip(−A_n/C_n,0,1)。B0/B1=0；C=0回退0。
- monotone：在B3/7/15/31对−A/C作权重ALC_weight×C的非增PAVA，再裁剪[0,1]。B0/1不加入单调约束，因d恒0。
- global：跨B3–31按ALC权重合并A/C估计单一系数，作为仅缩小幅度控制。
- 保留R25、原R64、frozen smoothed2、training-selected mean和R56_segment对照。外层oracle仅诊断。

## R71：单一预注册可观察条件

z=词法与语义（含metadata，按该ridge缩放）的余弦最近support index是否一致，确定性argmax tie。每个预算两组，不加新的阈值、维度、MLP。与R43仅语义杠杆/LOO特征及R45/R46固定平均不同。
- 分预算/组拟合α=clip(−A/C,0,1)；一组至少20个具有非零d的source pair且至少3BM，否则α=0，推理缺失回退R25。
- R71已包含预算条件，直接与R70独立预算、非增预算比较，不另扫描组合。
- 4个support-label permutation seed7101–7104：在每预算的允许前缀内置换，重新计算R25、词法以及所有依赖标签的量，冻结α，不重训。保留正确/置换预测配对。
- 4个随机排序seed7111–7114的选择控制：在每episode/预算按随机query排序分配修正能量，最后一个用分数系数精确匹配Σc(αd)²；所有系数[0,1]。不使用query outcome。记录能量匹配误差和alignment变化，不将其当可部署模型。
- 至少运行一次R71完整冻结探索，来源不一致也不反复调整。若不稳定优于R70则停止此分组设计。

## R69 / R72

先按官方文档和repo冻结输入合同。官方runner不可访问时R72标记conditional_simulation，不声称hostedscore；不因未知而停止覆盖率与最小预测实验。独立预算、无跨调用状态、仅当前输入attributes/revealed labels。B0空、无peer回退，same visible subject不能借助内部subject_id区分。内部item_id仅用于离线审计，不是推理匹配键。

R72覆盖率审计后、任何预测计分前，另冻结具体peer算子和完整条件场景；审计仅看subject/input/支持分配，不看query outcome。必须包含0/1/3peer覆盖、同题/异题、peer置换、强own-only、R20/R36共享方向和均值控制。

## 资源、停止与晋级

预估整个新轮CPU批处理≤4小时；每单阶段上限90分钟，超限先保存进度并缩减冗余诊断，不能悄悄缩小主评价。保留中断/异常原记录。旧mlp/、R25/R56/R64/fallback不改。训练来源参数和代码hash先冻结再完整外层。每≤60分钟写回todo。

Research Gate仍为相对两个强基线至少.005绝对改善、3fold/BM改善、clusterCI上界<0、无未解释严重回归；小收益可标记探索候选，不自动晋级或提交。所有报告区分信息不足、估计失败、迁移失败、接口未知及主指标收益不足。
