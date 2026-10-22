# Loop v3 复现入口

Authority: 根目录 `loop_2.md` 与四份参考材料。v2报告、权重和ZIP均保留。复用公开数据及旧分组；以下不是独立盲测。

```powershell
Set-Location -LiteralPath 'E:\VScode_data\project\nips-comp\paiec_baseline'
$env:PYTHONPATH='D:/paiec_research_deps'
$env:HF_HUB_OFFLINE='1'
$env:HF_HOME='D:/paiec_pretrained/hf_cache'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
```

## R32 评价测量

```powershell
python -m mlp.audit_evaluation_items
python -u -m mlp.audit_full_queries
python -u -m mlp.full_query_dominant
python -m mlp.validate_full_query_replay
python -m mlp.summarize_measurement_audit
python -m mlp.checkpoint_v3
```

`audit_full_queries`按已写出的benchmark JSON恢复；仍运行时不要启动第二份。参数、125个分层选中pair、日志在`runs/research_loop/R32/`及`R32_full_queries.log`。R32c为全部4278个alignment_faking pair的全评价；不要误称R32分层汇总为全部5242个pair的完整评价分数。若需要从头重算，使用新实验目录，保留已有产物与源代码快照。

数据revision `54074453da7b9911be0f48e4c560be547bc84d21`；原MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。新增编码缓存放D盘，不重新下载数据/模型，不更改全局依赖。R32c代码的执行后排序优化说明保存在`R32c/runtime_note.md`；实际执行版本在`R32c/code/`。

## 条件诊断与检查

```powershell
python -m pytest tools/test_full_query_measurement.py tools/test_oracle_scale.py -q
python -u -m mlp.audit_scale_headroom
python -m pytest tools/test_local_scale.py -q
python -u -m mlp.run_local_scale_round
python -u -m mlp.diagnose_scale_identifiability
python -m pytest tools/test_conditional_components.py -q
python -u -m mlp.run_conditional_factorial
python -m pytest tools/test_partition_mean.py -q
python -u -m mlp.run_partition_round
python -u -m mlp.audit_subject_affinity
python -m pytest tools/test_local_bernoulli.py -q
python -u -m mlp.run_bernoulli_round
python -m pytest tools/test_local_strata.py -q
python -u -m mlp.run_strata_round
python -m pytest tools/test_input_equivalence.py -q
python -u -m mlp.audit_input_equivalence
python -m pytest tools/test_context_tokens.py -q
python -u -m mlp.run_context_pilot
python -m pytest tools/test_evidence_reliability.py -q
python -u -m mlp.run_reliability_round
python -m pytest tools/test_moment_covariance.py tools/test_complementarity.py -q
python -u -m mlp.run_moment_pilot
python -u -m mlp.run_complementarity_pilot
python -u -m mlp.run_complementarity_outer
python -m pytest tools/test_oracle_scale.py tools/test_oracle_crossfit.py -q
python -u -m mlp.audit_oracle_optimism
python -m pytest tools/test_split_calibration.py -q
python -u -m mlp.run_split_calibration_pilot
python -u -m mlp.audit_interactor_coverage
python -m mlp.plot_v3_evidence
python -m mlp.plot_v3_oracle_transfer
```

R33命令只在R32汇总结论存在且满足预先登记条件时执行，否则断言停止。R33使用evaluation labels计算固定预测方向的乐观上界，只用于诊断；不能用于提交、实际调参或性能晋级。其真实完成状态以`runs/research_loop/R33/summary.json`及日志为准；文件或命令已存在不代表实验完成。

R34来源选择与外层评价分离，保存三折tau和固定幅度、各benchmark及逐pair预算分数。原始预测与统计baseline会重新回放校验。运行中不要修改该轮源码；恢复时先查看进程、日志和源文件哈希，避免重复运行。

不要运行旧`sync_v2_state`覆盖当前控制状态。fallback仍为`dist/smoothed_mean_strength2.zip`；没有正式提交或公开发布。

R47是公开资料与运行时接口审计，保留资产目录和真实TLS失败记录；不是像素模型实验。R48使用评价池两半之间的额外标签，仅诊断R33样本内oracle乐观性；不能当作预算内方法或应用晋级Gate。它同时重放R33的capped结果和主导benchmark的full结果；两种范围不能混成全benchmark全量成绩。
