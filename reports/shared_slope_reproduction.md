# R75/R76 复现入口与原始证据

## 版本与环境

- 仓库 `D:/nips-comp/nips-comp`；原始 HEAD `fd5b06df51233392ea9d482732af8063f4222ec1`，工作区含已有与本轮未提交改动。各 contract 保存执行时 git status 和逐文件源码 SHA256，不将 HEAD 冒充全部新代码版本。
- 数据 measurement-db revision `54074453da7b9911be0f48e4c560be547bc84d21`，只读 `D:/paiec_payload/data`。
- MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，本地 `D:/paiec_pretrained/all-MiniLM-L6-v2`。
- Python3.12.7 / NumPy1.26.4 / SciPy1.13.1 / torch2.9.1+cpu；迁移 wrapper `D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1`。AMD Ryzen7 4800H、16GB；预测最多4 CPU线程，新item encoder2线程。
- 数据/评测 seed0/1/2；cohort7501/7502；32次 source fitted-null；10000次 Benchmark-cluster bootstrap seed20261022。其余 RNG 字符串及 hash 定义见冻结源码。

## 最短阅读顺序

1. [研究解释与决策](shared_slope_findings.md)
2. [ALC、准确率、预算、fold、Benchmark表](shared_slope_tables.md)
3. [冻结合同](../research_loop/SHARED_SLOPE_CONTRACT.md)
4. [原文与作者代码映射](../research_loop/SHARED_SLOPE_SOURCES.md)

## 代码与运行命令

所有命令从 PowerShell 执行，代码内部将 cwd 设为 paiec_baseline。原执行命令也保存在 contract.json。现有 complete.json 会复用已完成 chunks；重新运行训练/预测入口会重写汇总及本次启动计时，不能把复用耗时当原始预测时长。保留原始资产后在独立副本复跑，或只执行重计分入口。

```powershell
$py = 'D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1'
$src = 'D:\nips-comp\nips-comp\research_loop'
& $py "$src\shared_slope_fetch_sources.py"
& $py "$src\shared_slope_source.py"
& $py "$src\shared_slope_outer.py"
& $py "$src\shared_slope_analyze.py"
& $py "$src\shared_slope_source_analysis.py"
& $py "$src\shared_slope_regression_audit.py"
& $py "$src\shared_slope_report.py"
& $py "$src\shared_slope_smoke.py"
& $py "$src\shared_slope_equivalence.py"
& $py "$src\shared_slope_cpu_smoke.py"
& $py "$src\shared_slope_cpu_cost.py"
```

`shared_slope_monitor.ps1` 在 full预测进程存活时记录 Windows lifetime peak working set；无法在已结束后恢复原进程。AnySearch 来源下载为公开原始资料，未运行所下载 MATLAB/MEX 二进制。

核心文件：`shared_slope_model.py`（纯算子），`shared_slope_data.py`（只读离线桥接），`shared_slope_source.py`，`shared_slope_outer.py`。新输入入口为 `shared_slope_adapter_cpu.py`，其 `make_r25_encoder(semantic_ridge)` 接 `dual_track_cpu_features.py`；peer缺失/身份歧义自动回退。partial 模式使用固定罚项8，不需要隐藏 fold/Benchmark名称查表。

## 原始输出

根目录：`D:/nips-comp/nips-comp/paiec_baseline/runs/research_loop/`。

|路径|内容|
|---|---|
|R75/contract/|实际源代码快照、哈希、命令、环境、预注册metadata|
|R75/literature/|AnySearch发现记录、作者PDF/文本、MALSAR固定commit代码与manifest|
|R75/source_fold*/BENCHMARK/|选择名单、cohort、heterogeneity、原始predictions.npz/index.json、paired_scores|
|R75/selection.json|在外层预测前冻结的三个source入口与partial选择|
|R75/source_coverage.json|计划BM、实际BM、未评分BM、outer交集|
|R75/deduplicated_source/|按BM固定首个source context去重的表与比较|
|R75/factor_comparisons_by_selection_fold.json|完整三个source环境的因素差异，特别是仅fold2有实际ridge对照|
|R75/source_replay_validation.json|1146暴露轨迹、573去重轨迹的概率重计分|
|R76/full/outer/fold*/BENCHMARK/|所有新概率、counts/successes、原始index、cohort mapping、逐请求标签成本|
|R76/full/summary.json|5242×3×6主指标、每BM/seed/fold、CI、alignment贡献|
|R76/full/additional_comparisons.json|R25/R56/R72与同cohort fully pooled配对比较|
|R76/full/mechanism_comparisons.json|peer数量、组成、partial与置换比较|
|R76/full/label_cost*.csv|各预算实际标签数量、dual矩阵规模|
|R76/full/severe_regression*.csv/json|回归切片与精确平方位移成本分解；不是纯方差增量|
|R76/full/gate.json|四项数值条件与严重回归待审分开；没有Hosted部署标记|
|R76/full/replay_validation.json|149chunks、1,005,741 variant行、Brier最大误差8.88e-16、accuracy误差0|
|R76/full/resources.json + R76/process_memory.csv|原实际耗时、外部OS内存测量；内置0内存值无效|
|R76/smoke/|16项synthetic与原R36/R72公式等价性|
|R76/cpu_adapter_smoke/|真实新文本CPU编码、fallback、隐藏ID白名单验证|
|R76/cpu_adapter_cost/|124标签单请求CPU时间与真实process memory；单次synthetic观察|
|R76/delivery_manifest.json|本Track新代码/报告/结果逐文件哈希及历史freeze检查|

R56_segment 的概率仍保存在 R62 原始输出；R76 的汇总引用其相同 identity 的冻结 paired scores，**R76新npz没有假装另存一份R56概率**。R72 same3/pooled3/diff3 的原始概率被显式复制作为旧cohort对照。

两张标准 matplotlib 图：[预算曲线](shared_slope_figures/budget_brier.png)、[随机组成与CI](shared_slope_figures/cohort_delta_ci.png)。图的生成入口为 `shared_slope_analyze.py`。

## 可复核边界

原 R62/isolation 的183资产及 R72/full、R72/coverage frozen sources 均核对通过；R1–R72历史源码、权重、fallback未改。源码数学测试和CPU synthetic不证明Hosted生命周期/标签可见性。当前 adapter的排序取peer策略也不同于离线随机cohort；结果报告对该差异已作显式限制。

修复过的本轮辅助脚本问题：文献UCL HTTPS连接失败后原作者HTTP成功；最初内存 ctypes计数为0，改用外部Windows OS观测；新CPU smoke夹具 deepcopy共享字典造成KeyError后修正独立夹具；回归报告的 pandas tuple索引改为list。它们未改变冻结概率生成模型、预注册超参、cohort或评测结果。
