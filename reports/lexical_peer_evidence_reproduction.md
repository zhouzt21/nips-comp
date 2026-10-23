# R69–R72 复现与存储说明

原始用户计划：[plan_6.md](../doc/plan_6.md)。执行参数：[LEXICAL_PEER_EVIDENCE_SPEC.md](../research_loop/LEXICAL_PEER_EVIDENCE_SPEC.md)，R72补充：[R72_OPERATOR_CONTRACT.md](../research_loop/R72_OPERATOR_CONTRACT.md)。原prompt未改写。

## 环境与输入

- Windows，Ryzen7 4800H，8核/16线程，16GB RAM；每个预测批处理4线程。主要计算按阶段推进，R71最后若干小Benchmark与R72启动短时重叠；实际wall time不能简单相加。
- 项目 `D:/nips-comp/nips-comp`；基线 `paiec_baseline`；解释器入口 `D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1`。
- Python3.12.7、NumPy1.26.4、SciPy1.13.1；其他实际版本及commit/dirty tree在各阶段contract.json。新增神经网络训练0、embedding重算0、付费API0。
- measurement-db `54074453da7b9911be0f48e4c560be547bc84d21`，本地 `D:/paiec_payload/data`。
- MiniLM `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，本地 `D:/paiec_pretrained/all-MiniLM-L6-v2`；缓存 `D:/paiec_research_cache`。
- source seed73、每Benchmark至多25pairs；外层seed0/1/2、5242pairs、64 query items（不足64则全部）、所有query response variants与重复计数。不是全部可能评价item无上限扫描，也不是私有Hosted测试。

## 执行命令

在PowerShell中按顺序运行，脚本内部会切换到基线目录；不要使用系统默认Python替代迁移runtime。

```powershell
$py = 'D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1'
$src = 'D:\nips-comp\nips-comp\research_loop'
& $py "$src\run_lexical_source_bank.py"
& $py "$src\fit_lexical_calibration.py"
& $py "$src\run_lexical_budget_outer.py"
& $py "$src\run_lexical_reliability_outer.py"
& $py "$src\audit_peer_coverage.py"
& $py "$src\verify_peer_operator.py"
& $py "$src\run_peer_current_outer.py"
& $py "$src\analyze_lexical_peer.py" R70
& $py "$src\analyze_lexical_peer.py" R71
& $py "$src\analyze_lexical_peer.py" R72
& $py "$src\build_lexical_peer_tables.py"
& $py "$src\audit_peer_postrun.py"
& $py "$src\finalize_lexical_peer_delivery.py"
```

完成标记存在时生成脚本复用已完成Benchmark。遇到部分npz但没有complete.json会停止，避免静默把半批当完成；应先保留异常目录再重建，不能覆盖历史输出。原始依赖指纹不一致时应先审计，不绕过freeze。

## 输出目录

|目录|内容与读取方法|
|---|---|
|`paiec_baseline/runs/research_loop/R69`|官方页面、repo/client源码、PAVA全文与作者代码、URL/hash、失败记录、机器runtime合同与结构复核记录|
|`R70/source_bank`|412个不同来源pair的多ridge预测、合法support置换、近邻组、计数；同一pair可进入多个来源池，不能按824个独立pair计|
|`R70/calibration/fold*`|来源OOF系数、inner-half独立验证、源码冻结，α数组对应[0,1,3,7,15,31]|
|`R70/full`|完整冻结预算候选、原R64/R25/均值对照、各budget/BM/fold/seed/clusterCI、原始概率|
|`R71/full`|近邻一致性分组、4次完整support-label置换、4次等能量随机选择、近邻组与residual alignment/cost|
|`R71/invalid_control_seed_20261023`|已中止且排除的错误诊断批次。审计ID含query successes，误用作RNG；修复后独立重跑。不要读入最终统计|
|`R72/contexts`|固定可见subject分组、support/query完整输入键、peer列表、歧义与覆盖率；内部ID仅离线复核与内容查表|
|`R72/full`|0/1/3peer条件模拟、same-input/diff-input共享、R20/R36匹配场景对照、仅增加标签均值、peer-label置换|

`predictions_####.npz`中的每个方法数组为`[query_variant_rows,6]`，预算顺序如上；counts/successes保留真实重复response。相应index JSON的start/size切出一个subject–Benchmark–seed trajectory，identity包含完整审计元数据；该identity只能用于审计，不能用于任何预测随机种子。

每pair的Brier=`Σ(c*p²−2*s*p+s)/Σc`；accuracy=`Σ(where(p≥.5,s,c−s))/Σc`；先pair内计数、再等权pair与seed。ALC=.1B0+.2B1+.2B3+.2B7+.2B15+.1B31。R56_segment原概率保存在R62，对比行从同identity的R62冻结分数引入，没有重新训练或改写。

## 验证与证据边界

各full/replay_validation.json逐一重计保存概率、身份和计数；full/additional_comparisons.json提供R25/R56/R70/R36等配对CI；full/gate.json是数值条件，不代替对严重回归与真实runtime的判断。full/execution.json记录实际批处理时间，均不计为用户投入时间。

R69机器合同分为初始PAIEC_RUNTIME_EVIDENCE_CONTRACT.json与官方client补证JSON，前者被coverage冻结，后者被R72预测冻结。没有实际Hosted runner执行记录。R72的peer组成/跨subject重叠明确是条件模拟。

主报告、指标表、图和最终manifest以本轮交付文件为准；旧R25、R56、R64和smoothed_mean_strength2.zip不修改，没有自动提交。

当前入口：[完整发现与决策](lexical_peer_evidence_findings.md)、[数值/正确率/逐BM表](lexical_peer_evidence_tables.md)、[CSV](lexical_peer_evidence_metrics.csv)、[汇总JSON](lexical_peer_evidence_summary.json)。图由 `build_lexical_peer_tables.py` 生成PNG和SVG，已直接查看图像；字体、图例、坐标与数据趋势可读。

实际外层wall time：R70 73.59秒、R71正确重跑1389.70秒、R72 1410.69秒；重计分29.58/101.60/75.19秒。source bank162.92秒。单阶段均未达到90分钟上限。运行目录execution.json保留精确值；这些不是单模型延迟或用户投入时间。

`R72/full/matched_slice_alc.csv`按完整pair分母汇总切片贡献，不是切片内条件均值；matched使用保守的peer唯一输入键，diff3移除所有同键记录，详见主报告。`R72/postrun_own_geometry_audit.json`是36BM×3seed首条轨迹的去peer算子等价补充审计。
