# R65–R68 实际执行与复核入口

## 环境与数据

- 代码仓库`D:/nips-comp/nips-comp`，基线工作目录`paiec_baseline`；Git HEAD `fd5b06df51233392ea9d482732af8063f4222ec1`，工作树含此前及本轮改动，实际代码hash/快照在各contract内。没有清理或覆盖旧工作。
- Python3.12.7、Torch2.9.1+cpu、NumPy1.26.4，Ryzen7 4800H；R66/R67用threadpoolctl及Torch4线程、串行fit。依赖来自迁移runtime及外部research deps，没有新安装。
- 数据revision `54074453da7b9911be0f48e4c560be547bc84d21`，`D:/paiec_payload/data`；MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`只读既有cache，没有新embedding。
- R66新特征缓存`D:/paiec_research_cache/transferable_prior_v1`；`R66/cache_inventory.json`记录hash。原full编码/样本/episode回放后才保存消融特征。
- 模型seed2026；来源item-split seed73；外层item-split seeds0/1/2，不是三个模型随机初始化。

## 真实命令

下列入口均已实际执行，原脚本和首次冻结/修复快照均保存。已完成的summary属于下游冻结输入，重复覆盖会触发hash检查；本机只读核验优先执行最后的validation命令。完整恢复需要代码、run产物及外部cache，不能只克隆Git而期待忽略的权重/概率自动存在。

```powershell
Set-Location D:\nips-comp\nips-comp\paiec_baseline
$py = 'D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1'
$scripts = 'D:\nips-comp\nips-comp\research_loop'

& $py "$scripts\run_transferable_prior_diagnostics.py"
& $py "$scripts\run_transferable_prior_ablations.py"
# R66 item数值条件满足，冻结R67 amendment后才实际执行：
& $py "$scripts\run_transferable_item_outer.py"
& $py "$scripts\audit_transferable_prior_resources.py"
& $py "$scripts\validate_transferable_prior_results.py"
& $py "$scripts\summarize_transferable_prior.py"
```

AnySearch使用`C:/Users/Administrator/.codex/skills/anysearch/scripts/anysearch_cli.py`。academic检索前实际调用`get_sub_domains --domain academic`，随后search及extract原文/官方页面；未用摘要替代全文。完整依据见[literature ledger](../research_loop/literature/transferable_prior_mechanism.md)。

## 产物索引

以下均相对`paiec_baseline/runs/research_loop/`。

| 路径 | 内容 |
|---|---|
| `R65/descriptor_v2/summary.json`、`metrics.csv` | 最终修正版本，B0/B1全部分解及首标签贡献 |
| `R65/descriptor_v2/decomposition.csv` | 15726轨迹×5方法×2预算的精确response加权分解 |
| `R65/descriptor_v2/source_statistics.json`、`observable_diagnostics.csv` | fit-only统计、逐pair合法输入描述；query汇总仅离线诊断 |
| `R65/descriptor_v2/benchmark_*.csv` | 逐BM预测/真实均值、离散/协方差、描述统计和相关性 |
| `R65/contract*`、root CSV | 历史失败/修复留痕；root描述统计被descriptor_v2替代，原Brier不变 |
| `R66/fold{f}_{subject,item}_inner{i}/` | 12fits×31checkpoint，配置、isolation、history、风险/梯度审计、每epoch指标 |
| `R66/fold*_selection.json`、`fold*_predictions.npz`、`fold*_index.json` | 来源选点、完整6预算概率与支持/查询身份、计数 |
| `R66/summary.json`、`metrics.csv` | 三池结果及预冻结R67探索条件 |
| `R66/identity_and_rescore_audit.json` | query_groups独立重建及1648消融轨迹重计分，误差0 |
| `R67/contract/`、`evaluation_contract/`、`models.json`、`refits/` | 候选/epoch先冻结，3次全来源短fit与权重hash |
| `R67/outer/fold*/<benchmark>/` | 原R62全部身份、六预算概率、首支持prior、逐pair分数与耗时 |
| `R67/summary.json`、`versus_R56.json`、`decision.json` | 全体/逐fold/seed/BM、clusterCI、双均值Gate及最终不晋级 |
| `R67/validation.json` | 15726轨迹概率重计分、B1/B3–31不变式和183份旧冻结输入检查 |
| `R68/adele_audit.json` | 128items×4维×512请求的作者prompt精确重建及资源状态 |
| `R68/shared_runtime_audit.json`、`official_interface.md`、`official_baseline/` | 官方接口快照、固定commit代码、已确认与未知字段 |

报告表格/图由`research_loop/summarize_transferable_prior.py`生成，PNG/SVG/PDF在`reports/transferable_prior_figures/`。主报告8节为机制与决策层，不把相关性写成已识别因果机制。

## 运行量与异常

新增15fits：R66 12×30epochs，372checkpoint，训练函数累计178.827秒；R67选epoch1/2/2三短fit，8checkpoint，训练函数累计7.406秒。以上包含训练函数内监控/存盘，不含全部特征准备和错误恢复。R67单次完整入口144.241秒、BM回放合计125.252秒；不能把这些离线缓存测量当线上延迟保证。R66的summary秒数只记录续跑调用，不代表从首次准备到结束的总墙钟窗口；完整AI进展见D:/todo/reports/work-traces/nips-comp/2026-10-22-transferable-prior-mechanism.md。

保留的异常和修正：

1. R65初次枚举宽train_benchmarks时包含无joined样本来源，未生成结果；修正为实际inner_halves fit来源。
2. 单一只读structural reviewer识别R65 descriptor缓存遗漏metadata。相同content不同metadata真实存在；修复key，输出到独立`descriptor_v2`。新旧Brier分解完全一致，R66冻结引用未被覆盖。
3. R66在两次fit完成后读R59不存在的query_ids字段中断；改按真实query_groups合同核对，复用已完成fit。原contract及新的contract_v2保留。最终从raw response重建每个item–variant顺序逐项一致，未只用相同分数代替身份核验。
4. 官方baseline没有本项目smoothed_mean路径，一次404后以真实README/empirical_mean源码为依据；starter extract失败不解释为整个AnySearch不可用。

## 限定独立复核

按Global AGENTS复杂科研复现合同触发一个只读structural reviewer，检查R65/R66的response权重/消融/选择隔离及R67的variant映射/首标签/单输入合法性。metadata缓存P2已修复；query_groups待办经独立重建关闭。R67增量复核未发现阻断。最终机制文字由主agent根据原始指标核对，未表述为双审、人类质量校准或盲测通过。

fallback仍为两份归档中SHA256 `e55ee0b267f3b345685ef072cfa511c3c017d9082256573793961ea20feadcb9`；无新提交包、正式提交或付费调用。没有运行无关全套测试，本轮核验为实验隔离、数值/身份回放及冻结资产核对。
