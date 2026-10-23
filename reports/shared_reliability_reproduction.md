# R77–R79 复现与文件入口

原始授权：[plan_9.md](../doc/plan_9.md)。实验合同：[SHARED_RELIABILITY_SPEC](../research_loop/SHARED_RELIABILITY_SPEC.md)；来源选择隔离：[OOF补充](../research_loop/SHARED_RELIABILITY_OOF_ADDENDUM.md)。H3另有[辅助诊断范围](../research_loop/SHARED_RELIABILITY_HETEROGENEITY_NOTE.md)，不用于选模型。

## 环境和冻结资产

- `D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1`：Python3.12.7，NumPy1.26.4、SciPy1.13.1、torch2.9.1+cpu；CPU最多4线程，本机Ryzen7 4800H/16GB。
- data `D:/paiec_payload/data`，revision `54074453da7b9911be0f48e4c560be547bc84d21`。
- MiniLM `D:/paiec_pretrained/all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；沿用既有cache和metadata，不做新encoder训练。
- 源码HEAD `fd5b06df51233392ea9d482732af8063f4222ec1`，工作区有未提交代码；各轮contract中的source SHA256和快照才是实际新实现版本。
- 直接调用冻结 `shared_slope_model.predict`。没有重新定义Partial8的中心化、平滑均值、核和clip。无peer/身份歧义显式退回R25。
- AnySearch日志和原文/代码定位：[source_evidence](../research_loop/literature/shared_reliability/source_evidence.md)。MALSAR作者code和KDD2004原文沿用R75归档，DomainBed原文§3–3.1与固定commit `model_selection.py`重新读取；当前支持可靠性实验为自有机制适配。

## 命令

下面是实际入口。重跑应先保留原产物；已有complete会被复用，且execution会写入本次启动时长，不能把续跑耗时当原始实验总时长。冻结源码漂移会拒绝继续。

```powershell
$py = 'D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1'
$src = 'D:\nips-comp\nips-comp\research_loop'
& $py "$src\shared_reliability_sources.py"
& $py "$src\verify_support_diagnostic.py"
& $py "$src\run_shared_cohort_robustness.py" --stage alternative
& $py "$src\run_shared_cohort_robustness.py" --stage full
& $py "$src\analyze_shared_cohort_robustness.py"
& $py "$src\diagnose_cohort_heterogeneity.py"
& $py "$src\run_support_sharing_reliability.py"
& $py "$src\analyze_support_sharing_reliability.py"
& $py "$src\run_shared_reliability_policy.py" # 本轮R78入口通过，实际执行了来源阶段
& $py "$src\verify_shared_reliability_policy.py"
& $py "$src\summarize_shared_reliability_costs.py"
& $py "$src\summarize_shared_reliability_diagnostics.py"
& $py "$src\report_shared_reliability.py"
& $py "$src\validate_shared_reliability_delivery.py"
```

R79不可无条件启动；必须先读取R78决定。本轮实际运行来源验证，`source_policy_no_independent_gain`，完整外层未触发；没有R79/full或R79/outer。最终决定在[SHARED_RELIABILITY_DECISION](../research_loop/SHARED_RELIABILITY_DECISION.md)。单独重画R77图可执行`analyze_shared_cohort_robustness.py --plots-only`，不重复计算模型或改写原执行耗时。

## 原始数据位置

根目录 `D:/nips-comp/nips-comp/paiec_baseline/runs/research_loop/`。

|路径|内容|
|---|---|
|R77/alternative/|第二套既有cohort补评，原同信息pooled与新partial；独立全5242×3|
|R77/full/|五cohort、主cohort压力场景、source拟合幅度对照的完整评价|
|R77/*/contract/|命令、环境、git状态、源码hash及snapshot|
|R77/*/outer/fold*/BM/|cohort映射、可见配置审计、query index、counts/successes、predictions_*.npz、paired_scores、实际标签成本|
|R77/source_shrinkage.json|仅来源预测几何拟合的每fold/预算幅度；没有outer调参|
|R77/comparisons.json|同cohort配对区间、cohort平均风险、fold/预算、alignment/rest、严重回归|
|R77/benchmark_cohort_delta.csv|Benchmark×cohort增量及方向翻转|
|R77/prediction_geometry.csv|逐pair/seed/cohort的位移成本、方向交叉项与误差对齐分解|
|R77/cohort_heterogeneity.json|每BM/cohort固定四人组、B31/seed0支持异质性及null；不作模型选择|
|R78/source_fold*/BM/|来源可见cohort、query_predictions/index、support_predictions/index、diagnostics、complete|
|R78/diagnostic_metrics.csv|所有来源暴露逐pair/seed/budget/scenario的z、query Brier/accuracy、支持Brier、标签数|
|R78/deduplicated_diagnostic_metrics.csv|按BM首个合法来源池去重的报告样本，不挑最好结果|
|R78/associations.json|BM分组rho/CI、可靠组实际风险、mask重复稳定性、预算/组成/置换切片|
|R78/decision.json|事前测试逐项结果；是否进入R79来源策略验证|
|R78/replay_validation.json|保存的支持/查询概率独立重计分与mask覆盖|
|R78/budget_metrics.csv、benchmark_budget_metrics.csv、mechanism_slices.csv|逐场景/预算风险与response准确率、实际标签数、alignment/rest/dave切片|
|R79/source_contract/|R79来源阶段源码、输入概率/index、R78变量表的运行前hash及快照|
|R79/nested_source_policies.json|每pool内两半的合法fit/validation BM、fit-only变量筛选和matched常数|
|R79/source/fold*/BM/|仅来源阶段的own/pooled/partial/gate/matched概率、原identity、counts/successes及逐预算计分|
|R79/source_comparisons.json、decision.json|来源配对ALC差、停止条件；无新完整外层成绩|
|R79/source_error_decomposition.json|错误关闭损失与避免有害共享的离线分解，非推理特征|
|R79/source_replay_validation.json、source_isolation_validation.json|保存概率重计分及独立来源/目标BM隔离核对|
|shared_reliability_v9/|合成隔离检查、最终交付核对和文件hash|

R77的每个NPZ与相邻原index配套；Brier必须保留response counts/successes，不能先跨response平均概率后冒充原分数。R78的query NPZ仅两个预算列 `[15,31]`；support NPZ为逐mask留出概率，index记录fit/holdout、rep/half、当前预算peer前缀。每条own支持标签各被留出两次，非四个独立样本。

## 统计和资源边界

主要新机制比较使用10000次Benchmark-cluster bootstrap，seed20261023，保留BM内部pair、item seed、cohort依赖。R77已有通用summary中的历史均值对照自动汇总沿用旧工具seed20261022；最终机制结论以comparisons.json的新配对计算为准，不能混用随机种子的CI。

三item seeds不是三次训练初始化，五cohorts不是五倍独立Benchmark。R78三个来源池有重叠，综合统计去重；变量/门控选择若触发必须按OOF补充嵌套于对应来源，不能从来源池union回填outer。

R79只改变B15/B31，报告`.2*ΔB15+.1*ΔB31`的六预算ALC精确差值；没有把两预算风险之和伪装成绝对完整ALC。来源pool去重按BM首个合法pool，保留重叠信息，不将1146轨迹暴露当1146独立pair。

Windows内存计数设置正确HANDLE/函数签名后取进程lifetime peak working set，拒绝0值。各execution记录CPU运行的墙钟耗时和峰值MiB，不是CPU core-seconds，不记为人类投入。结果仍为`conditional_simulation`，没有Hosted输入契约或部署验收。
