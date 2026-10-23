"""Generate concise acceptance and score reports from actual completed evidence."""
import shutil,csv
from common import *
def fmt(v):return f'{v:.9f}'
def main():
    summary=read(ART/'evaluation/summary.json');execution=read(ART/'evaluation/execution.json');deltas=read(ART/'evaluation/release_comparisons.json');adapter=read(ART/'adapter_evaluation/summary.json')
    builds=read(HERE/'build_manifest.json');tests={k:read(ART/(k+'_full.json')) for k in builds};commands=read(HERE/'acceptance_commands.json')
    for k,v in tests.items():
        assert v['status']=='PASS' and len(v['fixtures'])==36 and len(v['synthetic_cases'])>=30
        assert read(ART/(k+'_fresh1.json'))['predictions']==read(ART/(k+'_fresh2.json'))['predictions']
        assert read(ART/(k+'_streaming.json'))['status']=='PASS';assert sha(builds[k]['zip'])==builds[k]['sha256']
    scores={
      'historical_foldwise_score':{k:summary['results']['historical_'+k] for k in ['R25','Partial8']},
      'release_fixed4_local_score':{k:summary['results']['release_'+k] for k in ['R25','Partial8']},
      'release_fixed4_adapter_score':adapter['results'],
      'hosted_score':None,'protocol':{'pairs':5242,'seeds':[0,1,2],'budgets':[0,1,3,7,15,31],'main_cohort':7501,'historical_ridges':[4,4,1],'release_ridges':[4,4,4]},
      'benchmark_equal_alc':summary['benchmark_equal_alc'],'folds':summary['folds'],'by_seed':summary['by_seed'],'comparisons':deltas,
      'boundary':'Full scores use frozen cached embeddings. Actual ZIP raw-input encoding validated on 36 Benchmarks x 4 queries x 6 budgets. Partial8 operator/roster-adapter are conditional local simulations; public entry has no trusted roster and falls back.'}
    write(HERE/'release_scores.json',scores)
    for name in ['metrics.csv','benchmark_metrics.csv']:shutil.copyfile(ART/'evaluation'/name,HERE/('release_'+name))
    status={'finished':now(),'candidates':{},'historical_replay':execution,'preservation':read(HERE/'historical_preservation.json'),'official_commit':read(HERE/'official_manifest.json')['commit'],'real_hosted_calls':0,'official_submissions':0,'hosted_score':None}
    for kind,v in tests.items():
        status['candidates'][kind]={'status':'NOT_READY','local_acceptance':'PASS','sha256':builds[kind]['sha256'],'bytes':builds[kind]['bytes'],'benchmarks':36,'real_query_inputs':144,'six_budgets':True,'synthetic_cases':len(v['synthetic_cases']),'max_probability_error':max(x['max_probability_error'] for x in v['fixtures']),'max_scaled_feature_error':max(x['max_scaled_feature_error'] for x in v['fixtures']),'resources':v['resources'],'fresh_process_repeat':'exact','public_streaming':'PASS_LOCAL_INJECTED_EXCHANGE','blockers':['Actual Starting Kit and Hosted dependency installation/runtime not verified','Linux clean-environment acceptance unavailable: WSL echo and Python calls hung; owned client probes stopped']+(['Trusted Subject roster absent from official predict signature; shared mode cannot activate there'] if kind=='partial8_shared' else []),'zip':builds[kind]['zip']}
    write(HERE/'acceptance_summary.json',status)
    (HERE/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+Path(v['zip']).name+'\n' for v in builds.values()),encoding='utf-8')
    ledger=[read(p) for p in sorted((ART/'commands').glob('*.json'))]
    write(HERE/'command_ledger.json',{'commands':ledger,'final_full_acceptance':['r25_nopeer_full_recheck','partial8_shared_full_recheck'],'initial_failures_retained':True})
    lines=['# 两套最终候选：验收与结果','',f'生成于 {now()}。两套 ZIP 均固定 semantic_ridge=4；metadata ridge=16，dual ridge=1，Partial8 参数=8。','',
      '**最终状态：A、B 均为 NOT_READY；已完成的 Windows 本地验收通过。** 实际 Starting Kit/Hosted 安装与运行尚未验收，Linux WSL 探测挂起。B 另外缺少官方可传入的可信 Subject 名单；当前官方入口严格回退 R25，不能称作已部署的共享方案。','',
      '## 三层证据','',
      '|配置/层次|R25 ALC|Partial8 ALC|含义|','|---|---:|---:|---|',
      f"|historical_foldwise_score (4/4/1)|{fmt(scores['historical_foldwise_score']['R25']['alc_mean'])}|{fmt(scores['historical_foldwise_score']['Partial8']['alc_mean'])}|历史全量回放，不是新 ZIP 分数|",
      f"|release_fixed4_local_score (4/4/4)|{fmt(scores['release_fixed4_local_score']['R25']['alc_mean'])}|{fmt(scores['release_fixed4_local_score']['Partial8']['alc_mean'])}|相同冻结输入与主 cohort 的算子分数|",
      f"|固定4 + 可信名单适配器|{fmt(adapter['results']['R25']['alc_mean'])}|{fmt(adapter['results']['Partial8_trusted_roster_adapter']['alc_mean'])}|包含重复证据/身份不完整等回退的条件模拟|",
      f"|固定4 + 当前官方 predict 接口|{fmt(adapter['results']['R25']['alc_mean'])}|{fmt(adapter['results']['Partial8_public_interface']['alc_mean'])}|B 无可信名单，回退 A；本地核算|",
      '|hosted_score|null|null|未发生真实 Hosted 运行|','',
      f"历史逐概率最大误差 {execution['max_historical_probability_error']:.3g}；R25/Partial8 的历史 ALC、逐预算 Brier、accuracy 与 R81 保存汇总完全一致。每个 query 均与 R76、R77、R81 主 cohort 保存概率核对。共 5,242 pairs × 3 seeds = 15,726 trajectories，36 Benchmarks，六预算。",'',
      '完整评分使用冻结的公开数据 embedding cache；ZIP 对原始文本的独立重编码验收覆盖每个 Benchmark 的4个 query（共144个）、六预算。没有把这个抽样 ZIP 验收写成全量文本重编码。R25_verify 早期子集报告不是本次全量参考分母。','',
      '## 固定4的变化与风险','',
      '|方法|pair-equal ΔALC|Benchmark-equal 历史→固定4|第三折 ΔALC|B31 accuracy 历史→固定4|','|---|---:|---:|---:|---:|']
    for k in ['R25','Partial8']:
        h=summary['results']['historical_'+k];r=summary['results']['release_'+k];d=deltas[k]
        lines.append(f"|{k}|{d['delta']:+.9f}|{summary['benchmark_equal_alc']['historical_'+k]:.9f} → {summary['benchmark_equal_alc']['release_'+k]:.9f}|{d['fold_delta'][2]:+.9f}|{100*h['accuracy_at_0_5'][5]:.4f}% → {100*r['accuracy_at_0_5'][5]:.4f}%|")
    lines+=['','前两折完全不变。固定4相当于对原第三折增加语义收缩；主要退化出现在高预算，尤其 alignment_faking。它占本次 pair 权重约81.6%，因此 pair-equal 退化和 Benchmark-equal 小幅改善可以同时发生。将退化归因于更强语义收缩是对同输入单参数对照的解释，不能据此推断隐藏 Benchmark 上的效果。','',
      '严重回归按预注册 ΔALC≥0.005 定义：R25 无；Partial8 的 alignment_faking 为 +0.005493410（第三折总体 +0.005246708）。两者均无 Benchmark 达到 +0.03。固定4 Partial8 相对固定4 R25 的最差 Benchmark 是 dave，+0.002699387。逐预算 Brier 的较大损失也保留在下表，不能仅靠整体 ALC 掩盖。','',
      f"固定4 Partial8 相对固定4 R25 的主 cohort ΔALC={deltas['release_partial_vs_r25']['delta']:+.9f}，Benchmark-cluster 95% CI={deltas['release_partial_vs_r25']['cluster_ci95']}；该结果仍是反复使用公开外层后的条件模拟。它不解决合法 Peer 名单和实际 cohort 组成问题，也不构成盲测或 Hosted 成绩。没有据此改参数或决定采用哪套。",'',
      '### 三折 ALC','', '|折（从1计数）|pairs|R25 历史|R25 固定4|Partial8 历史|Partial8 固定4|','|---|---:|---:|---:|---:|---:|']
    for f,v in summary['folds'].items():lines.append('|'+str(int(f)+1)+'|'+str(v['pairs']//3)+'|'+'|'.join(fmt(v['results'][k]['alc_mean']) for k in ['historical_R25','release_R25','historical_Partial8','release_Partial8'])+'|')
    lines+=['','### 固定4逐预算指标（accuracy 采用 p≥0.5）','', '|预算|R25 Brier|R25 accuracy|Partial8 Brier|Partial8 accuracy|','|---|---:|---:|---:|---:|']
    for j,n in enumerate([0,1,3,7,15,31]):
        a=summary['results']['release_R25'];b=summary['results']['release_Partial8']
        lines.append(f"|{n}|{a['brier_mean'][j]:.9f}|{a['accuracy_at_0_5'][j]*100:.4f}%|{b['brier_mean'][j]:.9f}|{b['accuracy_at_0_5'][j]*100:.4f}%|")
    lines+=['','### 第三折逐预算变化（固定4减历史1）','', '|预算|R25 ΔBrier|R25 Δaccuracy百分点|Partial8 ΔBrier|Partial8 Δaccuracy百分点|','|---|---:|---:|---:|---:|']
    fold=summary['folds']['2']['results']
    for j,n in enumerate([0,1,3,7,15,31]):
        values=[]
        for k in ['R25','Partial8']:
            a=fold['historical_'+k];b=fold['release_'+k];values.extend([f"{b['brier_mean'][j]-a['brier_mean'][j]:+.9f}",f"{100*(b['accuracy_at_0_5'][j]-a['accuracy_at_0_5'][j]):+.4f}"])
        lines.append('|'+str(n)+'|'+'|'.join(values)+'|')
    lines+=['','### 三个种子的固定4 ALC','', '|方法|seed0|seed1|seed2|seed间样本标准差|','|---|---:|---:|---:|---:|']
    import statistics
    for k in ['R25','Partial8']:
        vals=[summary['by_seed'][str(s)]['results']['release_'+k]['alc_mean'] for s in [0,1,2]]
        lines.append('|'+k+'|'+'|'.join(fmt(x) for x in vals)+'|'+fmt(statistics.stdev(vals))+'|')
    lines+=['','逐 seed、三折、各 Benchmark、预算差值和 cluster CI 全部保存在 release_scores.json、release_metrics.csv、release_benchmark_metrics.csv；未挑选最好 seed。','', '## ZIP 实测','', '|候选|ZIP字节数|原始输入最大概率误差|冷加载秒|最大请求秒|峰值RSS MiB|本地状态|','|---|---:|---:|---:|---:|---:|---|']
    for k,v in status['candidates'].items():
        r=v['resources'];lines.append(f"|{k}|{v['bytes']}|{v['max_probability_error']:.3g}|{r['cold_load_seconds']:.3f}|{r['max_request_seconds']:.3f}|{r['peak_rss_mib']:.1f}|PASS / overall NOT_READY|")
    lines+=['','实测主机为 AMD Ryzen 7 4800H（8核16线程），CPU-only，Python3.12.7、NumPy1.26.4、Torch2.9.1+cpu、Transformers4.44.2；推理线程2。完整依赖及模块实际加载位置在 dependencies.lock.txt 与 artifacts/*_full.json。CPU耗时是AI实验运行时间，不是人类投入时间。','',
      '通过项：两份官方 ZIP checker；接口返回有限[0,1]浮点数；六预算与0/1/2/3 Peer；无自身支持时0.5；身份歧义和缺失配置回退；重复/冲突 support 不去重且关闭共享；重复 query 概率相同；匿名分组隔离；忽略隐藏ID；B31后B0无标签残留；并发调用；两个全新进程逐项一致；空HF cache、阻断网络、禁止旧数据/缓存目录读取；实际依赖来自新venv；冷加载、warm请求、峰值内存。资源阈值是本地验收阈值，不是已确认的 Hosted 限额。','',
      '数值算子和适配器函数 AST 与冻结源码一致；structured_text/metadata 文件逐字节一致。仅更换相对导入、内置权重位置，并添加输入特征LRU、Windows mmap释放；无训练、R79门控或新预测算法。LRU最多2048个输入向量，不含 Subject、标签或预测。','',
      '官方公开 streaming client 对两套实际 ZIP 的六预算及重复 query 测试均通过；交换与分组为本地注入，不是真实 coordinator/worker。上游 tools/smoke_test.py 是 BLE 基线工具，干净推理venv运行缺pandas；换已有研究工具环境后通过其合成预测/错误处理段，在代理TLS重试mock断言失败（期望TLS错误，实际包装为network错误）。未修改官方源码，也未称此工具整体通过。该失败与本候选不调用API的路径分开记录。官方 Starting Kit ZIP 仍未获得；此前用户因Codabench故障暂停下载，未恢复登录尝试。公开 GitHub 最新commit为 cc88a959c1e5e7462f31a9215baba066ce6d6e2b。','',
      '官方公开说明指出额外包安装默认关闭、由组织方政策控制。requirements.txt 是本地依赖声明，不证明 Hosted 会安装固定版本；需确认基础镜像已经满足依赖或组织方明确允许安装。ZIP包含MiniLM与预测源码，不包含完整Python解释器/全部wheel。来源：[官方打包说明](https://aims.stanford.edu/competition/#submission-package)、[固定版本ZIP检查器](https://github.com/aims-foundations/paiec_baseline/blob/cc88a959c1e5e7462f31a9215baba066ce6d6e2b/check_submission_zip.py)、[固定版本streaming客户端](https://github.com/aims-foundations/paiec_baseline/blob/cc88a959c1e5e7462f31a9215baba066ce6d6e2b/tools/streaming_ingestion.py)。','',
      '初次 Windows ZIP检查因 safetensors mmap 文件占用导致清理失败，已在构造时克隆等值 tensor 存储后重跑通过。首轮A/B测试在各自全部36 Benchmark通过后，因测试夹具共享字典重复pop字段报错；修复测试脚本后分别完整重跑，不修改模型。日志均保留，不将失败运行当通过。','',
      '## SHA256','']
    for k,v in builds.items():lines+=['- `'+Path(v['zip']).name+'`：`'+v['sha256']+'`']
    lines+=['','## 保留项与复现','',f"历史保护核验：{status['preservation']['files_checked']} 个文件内容哈希未变，含 R1–R81 和保护的 smoothed_mean_strength2.zip。未正式提交、未重训、未更新历史分数。",'',
      '复现命令与目录入口见 README.md；实际运行命令及返回码见 command_ledger.json（保留初次失败及最终recheck）、acceptance_commands.json、streaming_commands.json、artifacts/commands/。Git提交仅含源码、合同、汇总、清单和适合Git的小型研究记录；两份ZIP及模型权重为本地产物，不上传Codabench。未纳入Git的大型记录见 publication_scope.json。','']
    (HERE/'ACCEPTANCE.md').write_text('\n'.join(lines),encoding='utf-8')
    target=REPO/'reports/final_submission_candidates.md';target.write_text('# 最终候选打包与三层验收\n\n完整报告：[ACCEPTANCE.md](../paiec_baseline/final_candidates/ACCEPTANCE.md)\n\n两套固定 semantic_ridge=4 的 ZIP 均为 **NOT_READY**，Windows 本地验收通过；Hosted尚未验证，B缺可信Subject名单时回退R25。\n\n历史ALC：R25 0.195123926 / Partial8 0.190547856。固定4算子ALC：R25 0.197285385 / Partial8 0.194974829；后者为条件共享模拟，不能标为当前官方入口ZIP的实测总分。详见完整报告的适配器计分边界、第三折退化和SHA256。\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v['status'] for k,v in status['candidates'].items()}))
if __name__=='__main__':main()
