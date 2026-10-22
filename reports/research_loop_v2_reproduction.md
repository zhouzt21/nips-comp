# PAIEC v2 复现入口

工作目录 `E:\VScode_data\project\nips-comp\paiec_baseline`。保留原数据revision54074453da7b9911be0f48e4c560be547bc84d21及所有benchmark splits。原737pair代理与5242pair覆盖扩展分开报告；均非盲测。

## 不需要新下载的末轮实验

```powershell
Set-Location -LiteralPath 'E:\VScode_data\project\nips-comp\paiec_baseline'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='4'
python -m mlp.run.run_budget_mean_round
python -m mlp.run.run_symmetric_prior_round
python -m mlp.run.run_covariance_round
python -m mlp.audit.audit_evaluation_coverage
```

R27/R28复用已下载数据的记录缓存；R28依赖runs/mlp_failure_2026_10_22/stat_prior/pair_rates.json。R29需要D盘原语义缓存，不重新训练编码器。R29真实与打乱协方差均被内层选择关闭，保存的npz不是更优模型的证明。

## 全合格pair覆盖扩展

```powershell
$env:PYTHONPATH='D:/paiec_research_deps'
$env:HF_HUB_OFFLINE='1'
$env:HF_HOME='D:/paiec_pretrained/hf_cache'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
python -u -m mlp.run.run_full_pair_round
```

模型固定在D:/paiec_pretrained/all-MiniLM-L6-v2，revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41。只补离线缓存。R31每benchmark结果已完成时复用；若需从头重评，应改用新输出目录并保存新协议/源码，不覆盖历史结果。原source脚本哈希与快照见每轮provenance.json和code/。

## 实验索引

| 内容 | 真实产物 |
|---|---|
| 扩大池监督训练R22 | runs/research_loop/R22/fold0、fold1、fold2；权重、元数据、train.log |
| 三种子最佳独立pair候选R25 | runs/research_loop/R25_verify/summary.json；参数在R25/foldN_selection.json |
| 条件共享R20 | runs/research_loop/R20_verify/summary.json；不能与独立pair场景直接排名 |
| R27–R29负结果 | 各轮summary.json；R28/recovery.md保留schema恢复记录 |
| 覆盖审计R30 | summary.json，重加权明确标为外推 |
| R31实际扩大评估 | summary.json、各benchmark JSON、*_pair_brier.csv、R31_run.log |
| 主报告及认识状态 | reports/research_loop_v2_2026-10-22.md；research_loop/HIGH_LEVEL_STATE.md |

## 检查与报告

```powershell
python -m pytest ../tests -q
python -m mlp.state.sync_v2_state
python -m mlp.state.checkpoint_v2
```

本轮完整tools回归实际63passed；R31另外以实际数据回放旧子集。没有新ZIP晋级；fallback仍为dist/smoothed_mean_strength2.zip，SHA256 E55EE0B267F3B345685EF072CFA511C3C017D9082256573793961EA20FEADCB9。未正式提交，也不在本会话结束后自动继续运行。
