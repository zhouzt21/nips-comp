# R80–R83 复现入口与证据边界

原始授权：[plan_10](../doc/plan_10.md)；事前合同：[PARTIAL8_NEXT_SPEC](../research_loop/PARTIAL8_NEXT_SPEC.md)；结论：[Q1–Q5](../research_loop/PARTIAL8_NEXT_DECISION.md)。完整数字、response正确率和资源见[指标表](partial8_decision_tables.md)。R80/R81实际执行；R82/R83未满足来源条件，没有其模型、目录或新外层成绩。

## 环境与版本

- Python入口 `D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1`，Python3.12.7、NumPy1.26.4、SciPy1.13.1、torch2.9.1+cpu。
- Ryzen7 4800H、16GB，BLAS最多4线程；不新增依赖、encoder训练或外部收费模型。
- data `D:/paiec_payload/data`，revision `54074453da7b9911be0f48e4c560be547bc84d21`；MiniLM `D:/paiec_pretrained/all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。
- 项目HEAD `fd5b06df51233392ea9d482732af8063f4222ec1`，有既有未提交文件。各contract的SHA256、源码快照和git状态记录实际实现，不能只凭HEAD恢复新脚本。
- 所有模型仍调用原冻结算子；raw包装只补充clip前及共享/偏移分解。原R1–R79不改写，只有全局状态和ledger追加本轮结论。
- `doc/plan_10.md`原为空文件，已从用户本次附件逐字节恢复；SHA256 `6685dadac0c93971673bdcb6873691199aa4dcca8993a36fc136c487af87df79`。

## 原文与作者代码

检索及提取采用AnySearch，[证据映射](../research_loop/literature/partial8_next/source_evidence.md)列出正式论文、版本、精确位置及适配边界。JMLR2004全文、scikit-learn1.5.2源码、MALSAR固定commit源码已实际读取；KDD2004与DomainBed沿用前轮原文及固定版本记录。CV方差理论不保证本项目LOIO有效；分组5fold随机轮转不同于sklearn该版本的按组大小贪心分配。机制对照是项目自有实验设计，不是论文数值复现。

PDF最初提取因迁移环境缺少pypdf/fitz失败，改用本机已有TeX Live `pdftotext -layout`。保留AnySearch JSON、PDF magic、SHA和文本，没有用摘要代替关键原文。

## 实际入口

```powershell
$py = 'D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1'
$src = 'D:\nips-comp\nips-comp\research_loop'
& $py "$src\partial8_next_sources.py"
& $py "$src\partial8_next_originals.py"
& $py "$src\run_partial8_identifiability.py"
& $py "$src\analyze_partial8_identifiability.py"
& $py "$src\run_partial8_mechanism.py"
& $py "$src\analyze_partial8_mechanism.py" --stage full
& $py "$src\run_partial8_heterogeneity.py"
& $py "$src\analyze_partial8_mechanism.py" --stage heterogeneity
& $py "$src\report_partial8_next.py"
& $py "$src\validate_partial8_next_delivery.py"
```

模型runner有complete复用与合同漂移检查；重跑前保留原execution，避免续跑耗时覆盖原实验墙钟。仅重画图表可运行report脚本，不重复预测。参数、seed、原始argv、硬件与源码hash在contract/execution中；这里不提供绕过合同的覆盖命令。

## 资产地图

以下目录均在`paiec_baseline/runs/research_loop/`。

|路径|作用|
|---|---|
|R80/contract/|事前源码、输入版本、命令及快照|
|R80/source_fold*/BM/|cohort、完整/留出支持预测、masked query预测、mask/peer前缀、OOB重拟合概率和index|
|R80/metrics.csv、deduplicated_metrics.csv|完整来源暴露与按BM首个合法来源池去重结果|
|R80/summary.json、decision.json|BA/rho、风险误差、配对CI、有害组检验及R82/R83停止决定|
|R80/calibration_bins.csv、benchmark_metrics.csv|来源校准分箱、逐BM/预算分数；不是新部署校准器|
|R80/refit_bootstrap_metrics.csv、refit_bootstrap_summary.csv|固定经验支持分布的32次OOB重拟合稳定性|
|R81/contract/、outer/fold*/BM/|冻结完整5242×3×6五cohort机制评价；原概率/新控制、raw值、query index/counts/successes、paired_scores|
|R81/comparisons.json|同cohort及cohort平均配对ALC、三fold、各预算、BM CI、alignment/rest、最差回归与强基线|
|R81/geometry.csv、geometry_by_*.csv|逐请求与汇总的方向、平方成本、clip、peer标签成本、风险差分解|
|R81/heterogeneity/contract/|真实偏移和已知机制实验的事前版本|
|R81/heterogeneity/BM/|25BM固定四人组、真实支持偏移、permutation、两世界生成参数、32次模拟标签、真概率与预测|
|R81/heterogeneity/summary.json|真实偏移重复性/风险、共同及异质世界的expected Brier；不是PAIEC成绩|
|R80/replay_validation.json、R81/replay_validation.json、R81/heterogeneity/replay_validation.json|独立保存概率重计分、response身份、group隔离与恒等分解|
|partial8_next_v10/|最终交付核对、历史保留检查、全部新资产hash|

R80的full风险/原half估计与R78核对。LOIO/group5fold将相同item全部放入同一侧；heldout标签不能进入fit均值、中心化或solver。R80来源三池有重叠，实际1146轨迹暴露，去重573轨迹=191pairs×3seeds/35BM。

R81所有控制共享query/支持/peer输入；exact-cost使用整组query预测和response counts，虽不看successes仍只作离线归因。逐点方向控制保留Partial绝对幅度，但clip/零pooled方向可能使幅度不匹配，必须同时看成本与边界率。

原runner几何CSV的方向mask只对pool施加数值零容差；独立分析按合同对pool和partial两向量均施加1e−12，保留原CSV并生成canonical根目录geometry。概率、Brier和正确率未因此改变。报告表头的绝对值竖线已转义，生成脚本同步修改。

## 统计与解释

主指标pair-equal Brier ALC，预算权重.1/.2/.2/.2/.2/.1。response正确率按p>=.5、保留重复响应counts；符号BA是判断共享收益正负的balanced accuracy，不能互换。

新机制CI用10000次BM cluster bootstrap，seed20261023；保留同BM内pair/seed/cohort依赖，五cohort先平均风险而非概率ensemble。旧通用summary的历史均值对照CI采用旧工具seed20261022；新决策以comparisons.json为准。三个item-split seeds不是三个初始化；35/36BM不是五cohort乘五倍。

OOF item bootstrap只表示固定预测下的条件评分敏感性，不能宣称普适95%风险置信保证。OOB重拟合不提供新标签；LOIO没有mask随机性不能视为零估计方差。共同latent斜率合成世界存在概率clip，不能当严格共同线性概率null。

所有公开数据已被反复探索，完整外层仍不是独立盲测；Shared-Peer保持conditional_simulation，Hosted工作继续按用户要求暂停。实验秒数为实际CPU执行墙钟，内存为Windows进程峰值working set；不混入用户投入时间。图表位于`reports/partial8_next_figures/`。
