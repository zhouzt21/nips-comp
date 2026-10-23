# R62–R64 实际复现入口与产物

原需求：`doc/plan_4.md`。实验合同：`research_loop/HISTORICAL_POSITIVE_FOLLOWUP_SPEC.md`。使用迁移的Python3.12.7 CPU运行时，Torch2.9.1+cpu、NumPy1.26.4；Ryzen7 4800H（8核16线程），Torch/BLAS限制4线程。没有新安装依赖。

## 依赖

- 数据revision `54074453da7b9911be0f48e4c560be547bc84d21`，入口 `D:/paiec_payload/data` 指向迁移的数据目录。
- MiniLM `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；原语义、source episode、R55重采样banks缓存。
- R25/R31/R42/R43/R45/R46和R54/R55/R56历史选择/参考JSON；R43三份full-source权重在两份迁移归档中。
- R54/R56训练seed2026；R43/R55训练seed4307；R55 episode seeds5501至选定epoch。外层0/1/2是item-split seeds，不是三次模型训练。
- 权重、原始预测NPZ、逐pair CSV和大型cache通常被Git忽略。只克隆代码无法恢复本轮所有证据，需保留本机run目录与迁移归档。

## 命令

以下均为已存在且实际使用的入口；在PowerShell运行：

```powershell
Set-Location D:\nips-comp\nips-comp\paiec_baseline
$py = 'D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1'
$scripts = 'D:\nips-comp\nips-comp\research_loop'

& $py "$scripts\run_historical_positive_refits.py" audit
& $py "$scripts\run_historical_positive_refits.py" r62

# 数值不变的编码缓存；initialize只冻结原评分器合同，不开始慢速评分。
& $py "$scripts\check_historical_encoding_equivalence.py"
& $py "$scripts\initialize_historical_outer_contract.py"
& $py "$scripts\run_historical_outer_cached.py"
& $py "$scripts\analyze_historical_positive.py" R62

& $py "$scripts\run_historical_positive_refits.py" r63
& $py "$scripts\run_historical_positive_outer.py" R63
& $py "$scripts\analyze_historical_positive.py" R63

& $py "$scripts\run_historical_template_pilot.py"
# 仅当R64/summary.json的enter_outer为true：
& $py "$scripts\run_historical_template_outer.py"
& $py "$scripts\analyze_historical_positive.py" R64

& $py "$scripts\summarize_historical_positive_reports.py"
& $py "$scripts\render_historical_positive_figures.py"
```

冻结输入、源码或权重若变化，入口会报错，不能复用旧结果并标为新实验。本轮首次曾运行未缓存的`run_historical_positive_outer.py R62`，在alignment_faking计算成本过高时停止并保存部分产物，再启用数值等价缓存。已有慢速chunks的本机审计命令：

```powershell
& $py "$scripts\audit_historical_cache_overlap.py"
```

全新运行若从一开始使用缓存，就没有`incomplete_original`旧chunks，不能虚构这项对比；仍执行真实输入编码对照及每个Benchmark首pair原编码检查。

## 原始结果结构

| 路径（相对paiec_baseline/runs/research_loop） | 内容 |
|---|---|
| `R62/isolation/` | 外层来源隔离、冻结源码/数据manifest/原选择/归档fallback哈希与快照 |
| `R62/models.json`、`R62/refits/` | 六份全来源MLP权重；原配置、epoch、更新步数、训练监控、导出误差 |
| `R62/evaluation_contract/` | 评分前冻结的完整候选与模型权重 |
| `R62/outer/fold*/<benchmark>/` | 逐pair×seed六预算Brier/正确率、概率NPZ、support/query IDs及response计数 |
| `R62/low_budget_decomposition.csv`、`gate_decisions.csv` | 精确均值/变化项分解，首标签门控决策与后验收益对照 |
| `R62/incomplete_original/`、`encoding_equivalence.json`、`cache_overlap_audit.json` | 原慢速执行留痕与加速等价性证据 |
| `R63/models.json`、`R63/reused/`、`R63/refits/` | 三份原R43权重复用，四个非零epoch的R55必要refit，来源epoch0策略 |
| `R63/outer/` | 与R62逐轨迹身份一致的完整配对评价 |
| `R62/analysis_audit.json`、`R63/analysis_audit.json` | 已保存概率重新计分、预算不变式、数值Gate、机制诊断 |
| `R64/pilot_contract/`、`R64/pool*/` | 单一模板增量pilot、原R42数值回放、来源预测与分解 |
| `R64/full/` | 只有来源进入条件满足时才生成；同样5242pair×3种子完整评价 |
| `R64/status.json` | 最终执行状态；顶层summary的required是冻结的进入决定，不能误读成尚未执行 |

每轮的`summary.json`、`metrics.csv`、`benchmark_metrics.csv`、`paired_scores.csv`为机器可读入口。报告侧集中副本为`historical_positive_metrics.json/csv`和`historical_positive_benchmark_metrics.csv`；所有数字仍能回到各轮原始产物。

## 证据边界与异常保留

来源内选择的epoch已先冻结；新全来源监控曲线不用于选择epoch。R54/R56/R55标记为历史候选的foldwise refit，不冒称原内层权重。原R43权重保留原排列顺序，新R55使用half0+half1顺序，因此两者差值不能单独归因为重采样。

初始数值审计反复解压NPZ导致无效耗时，停止后改为每chunk一次载入；只优化分析读取，没有改变概率或评分公式。相关原日志保留。未运行额外全套pytest；本轮验证是实际实验中的隔离、回放、概率重计分与限定独立结构复核。

原始提示未改，旧R1–R61产物未覆盖；`dist/smoothed_mean_strength2.zip`的两份归档fallback SHA256始终应为`E55EE0B267F3B345685EF072CFA511C3C017D9082256573793961EA20FEADCB9`。未自动生成正式提交包或正式提交。
