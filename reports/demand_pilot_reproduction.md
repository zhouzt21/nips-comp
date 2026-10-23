# R74 复现与文件入口

工作目录：`D:\nips-comp\nips-comp\paiec_baseline`。CPU 2线程；无新增依赖。

```powershell
& D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1 D:\nips-comp\nips-comp\research_loop\demand_pilot_run.py
& D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1 D:\nips-comp\nips-comp\research_loop\demand_pilot_report.py
```

主runner重跑会复用固定输入但重建本轮R74产物；冻结hash不一致会拒绝。原失败contract仅作历史，正式执行依据contract_v2。不要修改R73原标注后覆盖本轮。

- 方案：`research_loop/demand_pilot_SPEC.md`；上位原计划`doc/plan_8.md`。
- 主结论：`reports/demand_pilot_findings.md`。
- 全表：`reports/demand_pilot_tables.md`、`demand_pilot_metrics.csv`、`demand_pilot_benchmark_metrics.csv`。
- 图：`reports/demand_pilot_figures/source_comparisons.png`、`grade_mae.png`，由report脚本生成。
- 所有R74证据根：`paiec_baseline/runs/research_loop/R74/`。
- `contract_v2/contract.json`：代码/hash、commit、环境、命令、冻结参数；`sources/`完整snapshot。
- `environment.json`：实际包版本、CPU逻辑核数与2线程设置；训练峰值内存未实测，不用后处理进程内存冒充。
- `pool{0,1,2}_inner{0,1}/`：4个student、4个permuted student、teacher/full响应prior，拟合BM/item和响应行身份。
- `pool{f}/{teacher,deployable}/predictions.npz` + `index.json`：概率、response counts/successes、item和variant身份；不依赖把平均概率误当完整Brier。
- `teacher_summary.json` / `deployable_summary.json`：所有方法/配对CI/pool/BM和回归。
- `grade_predictions.csv` / `grade_group_metrics.csv` / `grade_summary.json`：逐item grade真值/预测和分组误差。
- `decision.json`：停止决定；`probability_replay.json`：保存概率重计分；`cpu_feature_benchmark.json`：fresh新item实际特征耗时。report脚本默认保留已测时间；传`--refresh-cpu-cost`才重新测量。
- `delivery_manifest.json`：报告、脚本和本轮产物hash；排除它自身。

输入数据 revision：54074453da7b9911be0f48e4c560be547bc84d21；MiniLM revision：1110a243fdf4706b3f48f1d95db1a4f5529b4d41。所有source OOF结果仅作探索性证据，不与full outer数字直接横比。无新的outer结果。
