# plan_8 R73–R76 单一 structural review

审查日期：2026-10-23。本版证据截点：2026-10-22 19:02:50 UTC。审查者：独立只读 structural reviewer；仅本文件由 reviewer 写入。没有训练、重跑整批实验、修改实验资产或启用第二 reviewer。

## 结论与范围

**已读取范围内未发现必须修复的阻断问题。** R74 可支持“教师来源门槛失败、停止外层”的决定；R75 的来源选择及 R76 已冻结算子满足本次隔离合同。R76 全量评分在审查时尚未结束，因此本结论不等于最终数值或 Hosted 验收。

依据为 `doc/plan_8.md`、`research_loop/DUAL_TRACK_SPEC.md`、`demand_pilot_SPEC.md`、`SHARED_SLOPE_CONTRACT.md`；审查覆盖共享路线 data/model/source/outer/adapter/smoke/analyze/equivalence，需求路线 annotation preparation/audit、student、pilot runner/report，以及已落盘的 R73–R76 合同与结果。

## 问题与解释边界

### 必须修复

无已确认的实现阻断。没有以尚未完成的全量回放替代检查，也没有将后续可能变化的部署代码纳入本版通过范围。

### 非阻断，但报告必须保留的边界

1. **Medium / 部署证据范围：raw adapter 的 peer 选择与 R76 随机 cohort 不相同。** `shared_slope_adapter.py:66` 取可见配置排序后的前三个 peer；`shared_slope_model.py:9` 按预定随机种子建立最多四人的 cohort，`shared_slope_outer.py:19` 使用种子 7501/7502。两者可以共用预测算子，但现有 synthetic smoke 不能证明 adapter 在端到端输入流上重现 R76 的分数。若报告直接作此主张则需修正；截至本版未发现该主张。已通知主线程与 shared_slope worker，worker确认会明确区分。
2. **Low / 外部有效性：异质性门槛是固定的模型化诊断。** `shared_slope_model.py:68` 从当前 support 拟合共同斜率，再生成 32 个 Bernoulli null 样本；`SHARED_SLOPE_CONTRACT.md:15` 已承认 fitted-null 错配。通过门槛不证明已辨识真实潜在能力结构；未通过也不证明各 subject 斜率相同。source 三池有重叠，不能称独立重复。
3. **Low / 输入身份范围：可见字典无法揭示所有隐藏配置碰撞。** `shared_slope_adapter.py:57` 无真实可见 roster 时关闭共享；`:60` 排除重复/不完整 roster，`:65` 排除重复可见输入。该策略保守，但不能检测不可见身份差异。它忽略隐藏 ID，匿名 Benchmark token 仅用于相等分组，未送入 encoder（`:39`、`:47`）。合成案例不是 Hosted 证据。
4. **Low / Track A 负结果范围：只否定本轮小样本四维与固定 ridge 适配的进入条件。** `demand_pilot_run.py:317`–`:338` 同时要求教师与学生来源条件；实际 `R74/decision.json` 的教师条件失败，故未外层。不能将该结果扩大成 ADeLe 总体无效，也不能把学生的微小来源增益称作认知需求已被识别。现有 `demand_pilot_report.py:191` 后的解释保留了这些限制。

## Track B 已核对证据

- **隔离和覆盖。** 从实际 `source_fold*/complete.json` 与原 R64 outer 目录读取：R75 完成来源 Benchmark 为 25/23/22，各自与对应外层 10/13/13 个 Benchmark 的交叉均为零；异质性诊断为 19/16/15 个 Benchmark。没有把原始 train list 中无可用记录或无可共享配置的来源算作已完成评分。选取代码为 `shared_slope_source.py:16`、`:23`，来源池只由 `source(f)` 给定。
- **选择因果顺序。** `shared_slope_source.py:15`–`:34` 先用 support 完成 heterogeneity entry，`:74`–`:77` 再按来源 query 的 partial-vs-pool ALC 决定是否进入外层；没有外层 query outcome 进入决定。三池 entry 均通过，来源差值分别为 −0.0000566789674、−0.0000399388366、−0.0001229231500。其微小来源收益属于选择证据，不是独立确认。
- **partial8 公式。** `shared_slope_model.py:39`–`:45` 的 kernel 为 `XX' * (1 + same_subject/8)`；query cross-kernel 仅对 own subject 的 delta 加 `1/8`。与 `||beta_shared||² + 8Σ||delta_s||²` 的目标一致。既有 `R76/smoke/results.json` 记录 primal/dual 最大误差 7.77e−16；`real_source_equivalence.json` 的 R36 缩放和 R72 pooled 算子有限真实几何核对最大误差 2.22e−16。reviewer读取了检查代码和产物，没有自行重跑。
- **合法预算、随机性与回退。** `shared_slope_data.py:25`–`:32` 只返回当前 acquisition 前缀；`shared_slope_outer.py:31`–`:49` 的 missing、uneven、permutation 随机性只依赖标识和固定种子，未使用带 query outcomes 的 audit identity。own B0、peer0 和强制 ambiguity 保留原 R25，代码有逐元素断言（`:51`–`:52`）。预算不均时只截短前缀，没有未来标签。
- **计分与强对照。** `shared_slope_outer.py:24`–`:60` 按原 R64 index/chunk 边界复用 R72 的 probabilities/counts/successes，保留旧 R72 cohort 的明确名称。`historical_positive_common.py:69`、`:81` 使用 response-counted Brier/accuracy、pair/seed 等权及原六预算 ALC。R56 对照从相同 identity 的旧评分读取，未把其 variant 概率平均后重算。
- **source 汇总不挑好结果。** `shared_slope_analyze.py:72`–`:86` 采用每个 Benchmark 第一个可用 source fold 的固定去重规则，没有按分数选 source context；每 fold 的实际选择与最终外层对应关系仍保留。
- **冻结完整性。** 读取并逐文件计算 SHA256，R75/contract 的 6 个文件、R76/full/contract 的 5 个文件及 R76/smoke/contract 的 3 个文件均与冻结值一致。

## Track A 已核对证据

- **教师隔离。** `prepare_demand_annotation.py:33`–`:46` 重建原请求，输出 item 文本与 rubric 请求；`audit_demand_annotations.py:45`–`:46` 保留主标注，不按 response outcome 仲裁。512 主标注及 64 复标是同模型标注/一致性材料，不是人类校准；本审查没有逐条重新判级。
- **学生与 response fit 隔离。** 六个 `student_isolation.json` 中 fit/validation Benchmark 和 item 交叉均为零，fit 与对应 outer 交叉也为零。`demand_pilot_run.py:97`–`:122` 每 head 仅使用 fit 半组的有效 grade；`demand_student_cpu.py:9`–`:25` 的中心、标准差和 ridge 都从 fit 样本估计。response 模型只用 fit episodes（runner `:217`–`:239`）。
- **权重及信息对照。** `rows_for`（runner `:44`–`:62`）每 pair 内按 response count 归一，所有 pair 的总权重相同；teacher 只用四维完整 subset，full-source 学生模型另用完整 fit query 数据。新增模型的同数据 base、真实 demand 和置换 demand 保留相同槽位；R56/R66 仅作历史参照，不声称训练数据/模型容量完全相同。
- **置换无 outcome RNG。** runner `:20`–`:40` 的种子只含固定 seed、Benchmark、rubric；teacher 在 Benchmark 内联合置换完整四维，student 在 Benchmark 内对每 head 的有效 grades 置换。该控制保留 Benchmark 级分布，不是抹除所有任务类型信息。
- **query variants 与原计数。** runner `:65`–`:94` 通过原 R59 `query_groups` 映射 item 模型到 variants，核对 R66 counts/successes 和 query metadata 契约；`:248`–`:289` 保留 R56/R66 原 variant 概率。原失败断言在任何拟合前修复，旧 contract 保留，正式 contract_v2 记录该 amendment。该 contract 的 29 个文件 hash 当前均一致。
- **低预算因果性。** `demand_student_cpu.py:48`–`:56` 的 B1 仅使用首个 own support 标签与其预测，B0 不读标签；local 路线只改变 B3 及以上（runner `:261`–`:268`）。未使用 query outcomes 或未来 support 标签进入预测。
- **独立 grade 质量。** grade_summary 中 872 是跨来源池重复暴露；有效 unique item 为 110。teacher 176 exposure/88 unique pairs，deployable 824/412；本次读取确认每个 pair 恰好出现两次，未把重复池计作独立样本。整体 grade MAE：student 0.991676、fit mean 0.999972，不能称高保真提取器。
- **保存结果回放。** 已读更新后的 `demand_pilot_report.py:14`–`:45` 与 `R74/probability_replay.json`：1,000 stage-trajectory records、4,394,670 个保存概率值；重算 Brier/accuracy 最大误差 0，原 R25 重建最大误差 2.22e−16。R25/R56/R66 原 source paired scores 的新增对照最大误差也为 0。原身份对照另由 runner 的显式映射检查提供。
- **停止规则实际生效。** teacher 对 base 的来源 B0 差值 +0.0884234515，BM-cluster CI [0.0377889151, 0.1284794342]；teacher entry 为 false。尽管 prior_student、local_student 的形式来源条件通过，`decision.json` 仍为 `stop_at_pilot_keep_R25`，候选为 null，未启动新外层。

## 尚未覆盖 / 后续验收条件

- 本版读取时 R76 已有 27/36 个 Benchmark 的 complete 文件，最终 summary/replay_validation 尚未生成。最终 5,242 pairs、15,726 trajectories、149 chunks、所有 identity/count/success/probability 回放和严重回归判定，需要依据完整产物补核；不能从本版审查推定这些已完成。
- 新增 fresh-item CPU encoder 或部署 wrapper 若在本版后出现，需要只针对新增输入规范化、状态与精度维度做增量复核；不必重复整个科研审查。
- 没有现场 Hosted、真实 coordinator 或线上资源验证；16 个 synthetic smoke cases 仅证明其列出的合成输入和代数边界。
- 没有再次进行文献搜索或逐句原文复核。本版审查的是冻结科研合同、实现和本地产物，不替代论文复现范围说明、人工 grade 校准或最终科研价值判断。

## 增量：新 item CPU 编码器与 wrapper

主线程提交新增实现后，已只读复核 `dual_track_cpu_features.py`、`verify_dual_track_cpu_features.py`、`shared_slope_adapter_cpu.py` 和 `shared_slope_cpu_smoke.py`；未改已冻结的实验 runner。没有新增阻断。

- `dual_track_cpu_features.py:25`–`:45` 直接复用原 canonical 与 metadata encoder，并保持原 126 内容 token 截断、mask pooling 和归一化；没有创建历史 SQLite/cache 实例，不接收或缓存 response 标签。模型仅从本地加载并固定到 CPU。
- `shared_slope_adapter_cpu.py:10` 用该 encoder 包装原 adapter，解决旧 cache-only factory 无法处理新 item 的能力缺口；其 docstring 明确 peer 排序策略不等于 R76 随机 cohort。
- 已读取 `dual_track_v8/cpu_feature_validation.json`：4 个参考文本的特征最大差 7.4505805969e−8，新文本 640 维有限，重复及 scaling 差为 0。属于有限样本的容差核对，不是逐位等价或全语料验证。该检查的四个文本均配 `item_features='{}'`（验证脚本 `:12`），因此这份产物本身未覆盖真实非空 metadata；静态实现仍调用同一原 metadata encoder。
- 本增量读取时 `R76/cpu_adapter_smoke/results.json` 尚未生成，故只审了新 smoke 的代码，不声称实际执行通过。R76 outer 当时为 28/36 complete；最终全量回放仍待补核。

## 最终限定增量复核：全量产物与 CPU wrapper

证据截点：2026-10-22 19:16:09 UTC（北京时间 2026-10-23）。本节更新前文“R76 未完成”和“CPU smoke 待生成”的历史状态；仅追加本报告，没有重跑实验或改动其他文件。

**结论：新增范围未发现阻断实现问题。最终全量覆盖和保存概率回放已核对；“No-Peer 停止、Shared 条件候选不晋级”的决定有现有证据支持。** 这里的“不晋级”并非来自数值回放失败，而是 conditional evidence、实际严重回归和部署证据边界仍在。

### 全量覆盖和可回放性

- `R76/full/replay_validation.json` 的 36 个独立 Benchmark 检查项汇总为 **5,242 pairs、15,726 trajectories、149 chunks、1,005,741 query-variant rows**；本次读取逐项求和与顶层字段及 `summary.json` 一致，seed 为 0/1/2。计分脚本沿用六预算。
- 全量保存概率重算的最大 Brier 误差 **8.8817841970e−16**，accuracy 误差 **0**。已审过的 `shared_slope_analyze.py:16`–`:37` 对各 index 重建 identity hash，核对 counts/successes，并逐 method 重计分；本次核对其最终产物，没有把 reviewer 的静态检查称作重新运行全实验。
- R75 来源重放记录 **1,146 trajectory exposures、573 个按既定规则去重的 reporting trajectories**，最大 Brier 误差 6.1062266354e−16，accuracy 误差 0。`source_coverage.json` 正确列出 25/23/22 个实际评分 BM、19/16/15 个诊断 BM，source/对应 outer 交叉均为零。
- 当前 R75 科研合同 6 个文件、R76 full 合同 5 个文件及新增 CPU adapter smoke 合同 5 个文件 hash 均与冻结值一致。

### partial 相对同 cohort 的结果和 Gate

- fully pooled 主 cohort ALC **0.19072876104595127**；partial8 ALC **0.19054785628746854**。配对差值 **−0.00018090475848277**，95% Benchmark-cluster CI **[−0.00020224678502622, −0.00003805619611857]**。本次从 summary 相减与 mechanism_comparisons 的差异仅 3.66e−17。
- Benchmark-equal 差值为 −0.00010577052630876，三个外层 fold 差值均为负。因此可以报告“在预定主 cohort 与公共代理评价下，partial 相对同 cohort fully pooled 有小幅增益”；不能把旧 R72 cohort、主 cohort 和备选 cohort 混作完全同条件机制比较，也没有按外层最好 cohort 反选规则。
- `gate.json` 对 partial 标记 `four_numerical_conditions=true`，同时 `regression_review_required=true`、`scope=conditional_simulation`、`hosted_deployment=false`。这不是 full Gate 通过。相对 smoothed2 的 dave 回归 +0.0077622653；相对 training-selected mean 还有 cellverse、cruxeval、dave、egoschema 的既定阈值回归。
- 相对强 R25，**dave partial ALC 恶化 +0.0066328664009453**。相对同 cohort fully pooled 的小幅修复没有消除其相对 R25 的严重负迁移。该问题必须在主结论保留，不能被总体均值或通过四个数值条件掩盖。

### 严重回归分解：数值正确，字段名称有解释边界

`shared_slope_regression_audit.py:18` 使用精确恒等式：

`ΔBrier = 2 E_response[(p_R25 − y) × (p_new − p_R25)] + E_response[(p_new − p_R25)²]`。

`:22`–`:28` 先做 pair/seed 等权，再按原 ALC 权重汇总，并断言与 Benchmark ALC 差一致。dave 的 partial 项为 **+0.00162805036150567** 的 residual alignment 项与 **+0.00500481603943964** 的平方位移成本，相加等于上述 +0.0066328664。

**Low / 非阻断术语注意：** JSON/CSV 中 `variance_cost` 在代码上是平方位移成本，未中心化，不能直接称“方差增量”；聚合 JSON 的 `delta_brier` 实际已经按六预算加权成为 ALC 差。该恒等分解解释当前概率变化如何计入损失，不能单独证明由随机方差、错误标签或真实潜在斜率差异造成。已提醒主线程与 worker 保留此边界，不要求改写冻结数值资产。

### 新 CPU wrapper 与实际 smoke

- 新 `shared_slope_adapter_cpu.py:12`–`:17` 在调用原 adapter 前对白名单 subject/item 字段规范化，连 incomplete-configuration fallback 和注入的 own_predictor 都只收到可见字段；隐藏 subject/item ID 不参与 fallback 字典相等性。没有引入标签跨请求状态。
- `shared_slope_cpu_smoke.py:22`–`:30` 补测 incomplete 配置下修改隐藏 ID 不改变结果；实际 `R76/cpu_adapter_smoke/results.json` 为 `passed_new_text_synthetic_only`。新文本 pooled 使用 own 2/peer 2 个标签，缺少 roster 时 own-only 概率与仅 own 输入严格相同。该产物仍明确不是 Hosted 或 R76 随机 cohort 的端到端分数验证。
- `R74/cpu_feature_benchmark.json` 补充 **8 个真实 item、各自原始 metadata** 的共享 CPU encoder 对照。代码 `demand_pilot_report.py:57`–`:74` 读取原内容/metadata，比较历史缓存特征及四个 student heads；最大特征差 **7.4505805969e−8**、最大 demand 预测差 **9.3173606164e−8**。这补足了此前四个文本均为空 metadata 的有限覆盖，但仍不是全语料逐位保证。
- 资源实际记录为来源 91.27 秒、R76 prediction 794.73 秒；Windows 生命周期 peak working set 3550.81 MB，最大 visible labels/dual dimension 为 124。原 ctypes 的零内存值明确标为无效，未当作真实内存测量。微基准的进程启动、数据读取、模型初始化等包含/排除项单独报告，不能推定 Hosted 吞吐。

最终仍保留的边界：没有真实 Hosted/coordinator 验证；adapter 排序取 peer 与离线随机 cohort 不同；同模型复标不等于人工校准；公共数据被自适应反复使用；本次审查不批准发布或改变任何 frozen fallback。
