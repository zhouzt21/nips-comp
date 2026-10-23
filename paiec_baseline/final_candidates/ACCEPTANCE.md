# 两套最终候选：验收与结果

生成于 2026-10-23T11:43:18.301474+00:00。两套 ZIP 均固定 semantic_ridge=4；metadata ridge=16，dual ridge=1，Partial8 参数=8。

**最终状态：A、B 均为 NOT_READY；已完成的 Windows 本地验收通过。** 实际 Starting Kit/Hosted 安装与运行尚未验收，Linux WSL 探测挂起。B 另外缺少官方可传入的可信 Subject 名单；当前官方入口严格回退 R25，不能称作已部署的共享方案。

## 三层证据

|配置/层次|R25 ALC|Partial8 ALC|含义|
|---|---:|---:|---|
|historical_foldwise_score (4/4/1)|0.195123926|0.190547856|历史全量回放，不是新 ZIP 分数|
|release_fixed4_local_score (4/4/4)|0.197285385|0.194974829|相同冻结输入与主 cohort 的算子分数|
|固定4 + 可信名单适配器|0.197285385|0.194977444|包含重复证据/身份不完整等回退的条件模拟|
|固定4 + 当前官方 predict 接口|0.197285385|0.197285385|B 无可信名单，回退 A；本地核算|
|hosted_score|null|null|未发生真实 Hosted 运行|

历史逐概率最大误差 3.33e-16；R25/Partial8 的历史 ALC、逐预算 Brier、accuracy 与 R81 保存汇总完全一致。每个 query 均与 R76、R77、R81 主 cohort 保存概率核对。共 5,242 pairs × 3 seeds = 15,726 trajectories，36 Benchmarks，六预算。

完整评分使用冻结的公开数据 embedding cache；ZIP 对原始文本的独立重编码验收覆盖每个 Benchmark 的4个 query（共144个）、六预算。没有把这个抽样 ZIP 验收写成全量文本重编码。R25_verify 早期子集报告不是本次全量参考分母。

## 固定4的变化与风险

|方法|pair-equal ΔALC|Benchmark-equal 历史→固定4|第三折 ΔALC|B31 accuracy 历史→固定4|
|---|---:|---:|---:|---:|
|R25|+0.002161459|0.212920227 → 0.212778251|+0.002561692|76.3170% → 73.9944%|
|Partial8|+0.004426973|0.212085411 → 0.211598287|+0.005246708|79.1862% → 76.0340%|

前两折完全不变。固定4相当于对原第三折增加语义收缩；主要退化出现在高预算，尤其 alignment_faking。它占本次 pair 权重约81.6%，因此 pair-equal 退化和 Benchmark-equal 小幅改善可以同时发生。将退化归因于更强语义收缩是对同输入单参数对照的解释，不能据此推断隐藏 Benchmark 上的效果。

严重回归按预注册 ΔALC≥0.005 定义：R25 无；Partial8 的 alignment_faking 为 +0.005493410（第三折总体 +0.005246708）。两者均无 Benchmark 达到 +0.03。固定4 Partial8 相对固定4 R25 的最差 Benchmark 是 dave，+0.002699387。逐预算 Brier 的较大损失也保留在下表，不能仅靠整体 ALC 掩盖。

固定4 Partial8 相对固定4 R25 的主 cohort ΔALC=-0.002310556，Benchmark-cluster 95% CI=[-0.0025650812693783615, -0.0004211096601199864]；该结果仍是反复使用公开外层后的条件模拟。它不解决合法 Peer 名单和实际 cohort 组成问题，也不构成盲测或 Hosted 成绩。没有据此改参数或决定采用哪套。

### 三折 ALC

|折（从1计数）|pairs|R25 历史|R25 固定4|Partial8 历史|Partial8 固定4|
|---|---:|---:|---:|---:|---:|
|1|420|0.186244028|0.186244028|0.184963871|0.184963871|
|2|399|0.244494660|0.244494660|0.243848889|0.243848889|
|3|4423|0.191513398|0.194075090|0.186269801|0.191516509|

### 固定4逐预算指标（accuracy 采用 p≥0.5）

|预算|R25 Brier|R25 accuracy|Partial8 Brier|Partial8 accuracy|
|---|---:|---:|---:|---:|
|0|0.250000000|43.0466%|0.250000000|43.0466%|
|1|0.222119481|66.6975%|0.222119481|66.6975%|
|3|0.199412903|68.9978%|0.198584253|69.0118%|
|7|0.183529680|70.7752%|0.181315284|70.9071%|
|15|0.173352532|72.1031%|0.168784193|73.4857%|
|31|0.166024653|73.9944%|0.158141868|76.0340%|

### 第三折逐预算变化（固定4减历史1）

|预算|R25 ΔBrier|R25 Δaccuracy百分点|Partial8 ΔBrier|Partial8 Δaccuracy百分点|
|---|---:|---:|---:|---:|
|0|+0.000000000|+0.0000|+0.000000000|+0.0000|
|1|+0.000000000|+0.0000|+0.000000000|+0.0000|
|3|+0.000675652|+0.0057|+0.002095267|-0.1587|
|7|+0.002259252|-0.5472|+0.005874930|-2.0328|
|15|+0.005093315|-1.9540|+0.010802577|-2.9850|
|31|+0.009560486|-2.7527|+0.014921535|-3.7359|

### 三个种子的固定4 ALC

|方法|seed0|seed1|seed2|seed间样本标准差|
|---|---:|---:|---:|---:|
|R25|0.197396830|0.197330644|0.197128680|0.000139687|
|Partial8|0.195082047|0.194989110|0.194853330|0.000115025|

逐 seed、三折、各 Benchmark、预算差值和 cluster CI 全部保存在 release_scores.json、release_metrics.csv、release_benchmark_metrics.csv；未挑选最好 seed。

## ZIP 实测

|候选|ZIP字节数|原始输入最大概率误差|冷加载秒|最大请求秒|峰值RSS MiB|本地状态|
|---|---:|---:|---:|---:|---:|---|
|r25_nopeer|83506303|5.96e-08|0.832|7.524|557.7|PASS / overall NOT_READY|
|partial8_shared|83506305|8.51e-08|1.247|9.979|622.8|PASS / overall NOT_READY|

实测主机为 AMD Ryzen 7 4800H（8核16线程），CPU-only，Python3.12.7、NumPy1.26.4、Torch2.9.1+cpu、Transformers4.44.2；推理线程2。完整依赖及模块实际加载位置在 dependencies.lock.txt 与 artifacts/*_full.json。CPU耗时是AI实验运行时间，不是人类投入时间。

通过项：两份官方 ZIP checker；接口返回有限[0,1]浮点数；六预算与0/1/2/3 Peer；无自身支持时0.5；身份歧义和缺失配置回退；重复/冲突 support 不去重且关闭共享；重复 query 概率相同；匿名分组隔离；忽略隐藏ID；B31后B0无标签残留；并发调用；两个全新进程逐项一致；空HF cache、阻断网络、禁止旧数据/缓存目录读取；实际依赖来自新venv；冷加载、warm请求、峰值内存。资源阈值是本地验收阈值，不是已确认的 Hosted 限额。

数值算子和适配器函数 AST 与冻结源码一致；structured_text/metadata 文件逐字节一致。仅更换相对导入、内置权重位置，并添加输入特征LRU、Windows mmap释放；无训练、R79门控或新预测算法。LRU最多2048个输入向量，不含 Subject、标签或预测。

官方公开 streaming client 对两套实际 ZIP 的六预算及重复 query 测试均通过；交换与分组为本地注入，不是真实 coordinator/worker。上游 tools/smoke_test.py 是 BLE 基线工具，干净推理venv运行缺pandas；换已有研究工具环境后通过其合成预测/错误处理段，在代理TLS重试mock断言失败（期望TLS错误，实际包装为network错误）。未修改官方源码，也未称此工具整体通过。该失败与本候选不调用API的路径分开记录。官方 Starting Kit ZIP 仍未获得；此前用户因Codabench故障暂停下载，未恢复登录尝试。公开 GitHub 最新commit为 cc88a959c1e5e7462f31a9215baba066ce6d6e2b。

官方公开说明指出额外包安装默认关闭、由组织方政策控制。requirements.txt 是本地依赖声明，不证明 Hosted 会安装固定版本；需确认基础镜像已经满足依赖或组织方明确允许安装。ZIP包含MiniLM与预测源码，不包含完整Python解释器/全部wheel。来源：[官方打包说明](https://aims.stanford.edu/competition/#submission-package)、[固定版本ZIP检查器](https://github.com/aims-foundations/paiec_baseline/blob/cc88a959c1e5e7462f31a9215baba066ce6d6e2b/check_submission_zip.py)、[固定版本streaming客户端](https://github.com/aims-foundations/paiec_baseline/blob/cc88a959c1e5e7462f31a9215baba066ce6d6e2b/tools/streaming_ingestion.py)。

初次 Windows ZIP检查因 safetensors mmap 文件占用导致清理失败，已在构造时克隆等值 tensor 存储后重跑通过。首轮A/B测试在各自全部36 Benchmark通过后，因测试夹具共享字典重复pop字段报错；修复测试脚本后分别完整重跑，不修改模型。日志均保留，不将失败运行当通过。

## SHA256

- `paiec_r25_nopeer_final_candidate.zip`：`88d5e026015f219118b4e0c5bf91995397267d9983c757b56d4ec6e2756d69e9`
- `paiec_partial8_shared_final_candidate.zip`：`7a294166bf22892cb8ee3e73f268263bdfaf28895a5f6d2a5452d16866feac73`

## 保留项与复现

历史保护核验：12843 个文件内容哈希未变，含 R1–R81 和保护的 smoothed_mean_strength2.zip。未正式提交、未重训、未更新历史分数。

复现命令与目录入口见 README.md；实际运行命令及返回码见 command_ledger.json（保留初次失败及最终recheck）、acceptance_commands.json、streaming_commands.json、artifacts/commands/。Git提交仅含源码、合同、汇总、清单和适合Git的小型研究记录；两份ZIP及模型权重为本地产物，不上传Codabench。未纳入Git的大型记录见 publication_scope.json。
