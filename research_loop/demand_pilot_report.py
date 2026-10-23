"""Replay R74 probabilities; write complete tables, figures and interpretation."""
import time
import platform
import argparse
import importlib.metadata
import pandas as pd
from threadpoolctl import threadpool_limits
from historical_positive_common import *
from demand_student_cpu import load_demand_models, predict_demands

OUT=ROOT/'R74'; REPORT=REPO/'reports'


def replay():
    errors=[];historical_errors=[];records=0;predictions=0
    for stage in ['teacher','deployable']:
        rows=read(OUT/f'{stage}_paired_scores.json')
        lookup={(r['fold'],r['benchmark'],r['subject_id']):r for r in rows}
        for f in range(3):
            bank=load_npz(OUT/f'pool{f}/{stage}/predictions.npz')
            index=read(OUT/f'pool{f}/{stage}/index.json')
            old_r59={(r['benchmark'],r['subject_id']):r for r in read(ROOT/f'R59/fold{f}_paired_scores.json')}
            old_r66={(r['benchmark'],r['subject_id']):r for r in read(ROOT/f'R66/fold{f}_paired_scores.json')}
            for ix in index:
                a,b=ix['start'],ix['start']+ix['size'];c,s=bank['counts'][a:b],bank['successes'][a:b]
                r=lookup[(f,ix['benchmark'],ix['subject_id'])]
                assert len(ix['query_groups'])==len(c) and np.all(c>0) and np.all((s>=0)&(s<=c))
                for m in r['brier']:
                    p=bank[m][a:b];assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                    errors += [abs(brier(p[:,j],c,s)-v) for j,v in enumerate(r['brier'][m])]
                    errors += [abs(accuracy(p[:,j],c,s)-v) for j,v in enumerate(r['accuracy'][m])]
                    predictions+=p.size
                if stage=='deployable':
                    old=old_r59[ix['benchmark'],ix['subject_id']]
                    original={'R25':old['brier']['R25'],
                        'R56_segment':old['brier']['R56'][:2]+old['brier']['R25'][2:],
                        'R66_item_segment':old_r66[ix['benchmark'],ix['subject_id']]['brier']['item_segment']}
                    for m,v in original.items():historical_errors.extend(abs(np.asarray(r['brier'][m])-v))
                records+=1
    result={'status':'pass','trajectory_stage_records':records,'saved_probability_values':predictions,
            'max_brier_or_accuracy_replay_error':max(errors),'checked_counts_successes_and_shapes':True,
            'R25_R56_R66_original_source_score_replay_max_error':float(max(historical_errors)),
            'historical_r25_max_probability_replay_error':max(r['r25_replay_max_probability_error'] for r in read(OUT/'isolation_audit.json'))}
    assert max(errors)<1e-12 and max(historical_errors)<1e-12
    write_json(OUT/'probability_replay.json',result);return result


def cpu_benchmark():
    """Fresh CPU item encoding without modifying any historical SQLite cache."""
    import pyarrow.parquet as pq
    from dual_track_cpu_features import CPUItemFeatures
    from mlp.cohort_semantic import CachedSemantic
    from mlp.metadata_features import encode as meta_encode
    start=time.perf_counter();encoder=CPUItemFeatures(threads=2)
    heads=load_demand_models(OUT/'pool0_inner0')
    init_seconds=time.perf_counter()-start
    chosen=read(ROOT/'R73/item_mapping.json')[:8];items=[]
    for r in chosen:
        table=pq.read_table(Path('D:/paiec_payload/data')/r['benchmark']/'items.parquet',columns=['item_id','content','item_features'])
        found=[v for v in table.to_pylist() if str(v['item_id'])==r['item_id']]
        assert found;items.append({'item_content':found[0]['content'],'item_features':found[0]['item_features']})
    def features(raw):
        return encoder.encode_items(raw)
    # Warmup is explicit and excluded from repeats, which always recompute embeddings.
    warm_start=time.perf_counter();features(items[:1]);warm_seconds=time.perf_counter()-warm_start
    batch_times=[];single_times=[];head_times=[]
    for _ in range(3):
        t=time.perf_counter();x=features(items);predict_demands(heads,x);batch_times.append(time.perf_counter()-t)
        t=time.perf_counter();one=features(items[:1]);predict_demands(heads,one);single_times.append(time.perf_counter()-t)
    cache=CachedSemantic()
    cached=np.stack([np.r_[cache.vector(v['item_content']),meta_encode(v['item_features'])] for v in items])
    feature_error=float(np.max(np.abs(x-cached)))
    demand_error=float(np.max(np.abs(predict_demands(heads,x)-predict_demands(heads,cached))))
    assert feature_error<2e-6 and demand_error<2e-6
    x=np.tile(x,(125,1))
    for _ in range(20):
        t=time.perf_counter();predict_demands(heads,x);head_times.append(time.perf_counter()-t)
    output={'cpu_threads':2,'items':8,'repeats':3,'initialization_seconds':init_seconds,'warmup_seconds':warm_seconds,
        'batch8_seconds':batch_times,'median_batch8_seconds':float(np.median(batch_times)),
        'single_seconds':single_times,'median_single_seconds':float(np.median(single_times)),
        'heads_1000_median_seconds':float(np.median(head_times)),
        'item_identities':[r['custom_item_id'] for r in chosen],
        'grade_heads_bytes':sum((OUT/'pool0_inner0'/f'student_{d}.npz').stat().st_size for d in ['AS','MCr','QLl','KNf']),
        'includes':'canonicalization, original 126-content-token+2-special MiniLM CPU inference, legal metadata, four grade heads',
        'excludes':'item parquet disk loading, process startup/import and response prior/local solve; initialization separately reported',
        'cache_write':False,'cuda_used':False,'platform':platform.platform(),'model_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
        'implementation':'dual_track_cpu_features.CPUItemFeatures',
        'implementation_sha256':sha(REPO/'research_loop/dual_track_cpu_features.py'),
        'historical_cached_feature_max_error':feature_error,'historical_cached_demand_max_error':demand_error}
    write_json(OUT/'cpu_feature_benchmark.json',output);return output


def table_lines(summary, teacher=False):
    lines=['| 方法 | '+('B0 Brier' if teacher else '六预算 ALC')+' ↓ | Benchmark-equal ↓ | '+('B0准确率' if teacher else 'B0 / B1 / B31准确率')+' |',
           '|---|---:|---:|---:|']
    for m,r in summary['results'].items():
        acc=r['accuracy_at_0_5'];astr=f'{100*acc[0]:.2f}%' if teacher else ' / '.join(f'{100*acc[i]:.2f}%' for i in [0,1,5])
        lines.append(f"| {m} | {r['alc']:.9f} | {r['benchmark_equal']:.9f} | {astr} |")
    return lines


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-cpu-cost',action='store_true');args=parser.parse_args()
    verified=replay()
    cost=cpu_benchmark() if args.refresh_cpu_cost or not (OUT/'cpu_feature_benchmark.json').exists() else read(OUT/'cpu_feature_benchmark.json')
    teacher=read(OUT/'teacher_summary.json');deploy=read(OUT/'deployable_summary.json')
    grades=read(OUT/'grade_summary.json');decision=read(OUT/'decision.json');scope=read(OUT/'annotation_scope.json')
    audits=read(OUT/'isolation_audit.json');run=read(OUT/'run_summary.json')
    environment={'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
        'logical_cpus':os.cpu_count(),'training_thread_limit':2,'peak_training_memory':'not_measured',
        'packages':{name:importlib.metadata.version(name) for name in
            ['numpy','pandas','scipy','scikit-learn','torch','transformers','threadpoolctl','pyarrow','matplotlib']}}
    write_json(OUT/'environment.json',environment)
    grade_rows=read(OUT/'grade_predictions.json');gdf=pd.DataFrame(grade_rows)
    gmetrics=[]
    for groupby in ['benchmark','fold','rubric']:
        for key,part in gdf.groupby(groupby):
            for m in ['student','fit_mean','permuted_student']:
                d=part[m].to_numpy()-part['grade'].to_numpy()
                gmetrics.append({'scope':groupby,'group':key,'method':m,'exposures':len(part),
                    'mae':float(np.mean(abs(d))),'rmse':float(np.sqrt(np.mean(d*d)))})
    pd.DataFrame(gmetrics).to_csv(OUT/'grade_group_metrics.csv',index=False)
    records=[];bm_records=[]
    for stage,summary in [('teacher',teacher),('deployable',deploy)]:
        for m,r in summary['results'].items():
            for j,n in enumerate([0] if stage=='teacher' else BUDGETS):
                records.append({'stage':stage,'method':m,'budget':n,'brier':r['brier'][j],'accuracy':r['accuracy_at_0_5'][j],
                                'alc':r['alc'],'benchmark_equal':r['benchmark_equal']})
        for bm,v in summary['by_benchmark'].items():
            for m,value in v['scores'].items():bm_records.append({'stage':stage,'benchmark':bm,'method':m,'exposures':v['exposures'],'score':value})
    pd.DataFrame(records).to_csv(REPORT/'demand_pilot_metrics.csv',index=False)
    pd.DataFrame(bm_records).to_csv(REPORT/'demand_pilot_benchmark_metrics.csv',index=False)
    lines=['# R74 来源 pilot 完整数值表','',
        '所有数字均为 source pool 的 Benchmark-grouped OOF；三个 pool 重叠，不能当作三份独立样本。准确率是 response outcome 在 p≥0.5 下的预测正确率，不是 subject 原始答题正确率。','',
        '## 教师信息诊断：B0、四维完整注释 query 子集','']+table_lines(teacher,True)+['',
        '## CPU 学生部署：原完整来源 episode 六预算','']+table_lines(deploy)+['','## 六预算 Brier 与准确率','',
        '| 方法 | B0 | B1 | B3 | B7 | B15 | B31 |','|---|---:|---:|---:|---:|---:|---:|']
    for m,r in deploy['results'].items():lines.append('| '+m+' Brier | '+' | '.join(f'{v:.9f}' for v in r['brier'])+' |')
    for m,r in deploy['results'].items():lines.append('| '+m+' accuracy | '+' | '.join(f'{100*v:.3f}%' for v in r['accuracy_at_0_5'])+' |')
    lines+=['','## 关键配对差值（负数为改善）','',
        '| 范围 / 方法 − 对照 | Δ | BM-equal Δ | 三个pool Δ | 95% BM-cluster CI | 非alignment Δ |','|---|---:|---:|---|---|---:|']
    comparisons=[('teacher','teacher','base'),('teacher','teacher','teacher_permuted'),('teacher','student_subset','base'),
                 ('deployable','prior_student','R25'),('deployable','prior_student','prior_base'),
                 ('deployable','prior_student','prior_permuted_student'),('deployable','prior_student','R56_segment'),
                 ('deployable','prior_student','R66_item_segment'),('deployable','local_student','R25'),
                 ('deployable','local_student','local_permuted_student')]
    for stage,m,b in comparisons:
        c=({'teacher':teacher,'deployable':deploy}[stage])['comparisons'][m][b]
        lines.append(f"| {stage}: {m} − {b} | {c['delta']:+.9f} | {c['benchmark_equal_delta']:+.9f} | "+
            ', '.join(f'{v:+.8f}' for v in c['pool_deltas'])+' | ['+', '.join(f'{v:+.8f}' for v in c['cluster_ci95'])+
            f"] | {c['without_alignment_delta']:+.9f} |")
    lines+=['','## Grade 质量','', '| Rubric | 模型 | MAE ↓ | RMSE ↓ | rounded exact | rounded error>1 |','|---|---|---:|---:|---:|---:|']
    for rub,v in grades.items():
        for m,r in v['models'].items():lines.append(f"| {rub} | {m} | {r['mae']:.6f} | {r['rmse']:.6f} | {r['rounded_exact_agreement']:.4f} | {r['rounded_gt_one_error']:.4f} |")
    lines+=['','## 全部 Benchmark 分解','', '| 范围 | Benchmark | exposures | 方法 | score |','|---|---|---:|---|---:|']
    for r in bm_records:lines.append(f"| {r['stage']} | {r['benchmark']} | {r['exposures']} | {r['method']} | {r['score']:.9f} |")
    (REPORT/'demand_pilot_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=REPORT/'demand_pilot_figures';figs.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    names=['base','teacher','teacher_permuted','student_subset','R66_item_B0']
    ax[0].barh(names,[teacher['results'][n]['alc'] for n in names]);ax[0].axvline(.25,color='grey',ls='--');ax[0].set_title('Teacher subset: source OOF B0 Brier')
    names=['R25','R56_segment','R66_item_segment','prior_base','prior_student','local_student']
    ax[1].scatter([deploy['results'][n]['alc'] for n in names],names,s=65);ax[1].set_xlim(.208,.213);ax[1].set_title('Complete source episodes: ALC');ax[1].grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(figs/'source_comparisons.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for m in ['student','fit_mean','permuted_student']:
        ax.plot(['AS','MCr','QLl','KNf'],[grades[r]['models'][m]['mae'] for r in ['AS','MCr','QLl','KNf']],marker='o',label=m)
    ax.set_ylabel('Held-out grade MAE');ax.legend();fig.tight_layout();fig.savefig(figs/'grade_mae.png',dpi=160);plt.close(fig)
    tb=teacher['comparisons']['teacher']['base'];tp=teacher['comparisons']['teacher']['teacher_permuted']
    pb=deploy['comparisons']['prior_student']['prior_base'];pp=deploy['comparisons']['prior_student']['prior_permuted_student']
    pr=deploy['comparisons']['prior_student']['R25'];lr=deploy['comparisons']['local_student']['R25']
    findings=f'''# R74 No-Peer：四维任务需求与 CPU 学生来源 pilot

日期：2026-10-23。状态：完成来源 pilot；按冻结条件停止，未运行新的 5,242-pair 外层。原始 R25 / R56 / R72 与 fallback 未修改。

## 结论

本轮**没有证明 Luna 四维需求标注在现有输入之外提供可迁移的 response 预测价值**。四维完整 query 子集的教师增强 B0 Brier 为 **{teacher['results']['teacher']['alc']:.9f}**，同数据 base 为 **{teacher['results']['base']['alc']:.9f}**，置换教师为 **{teacher['results']['teacher_permuted']['alc']:.9f}**。教师对 base 的 Δ={tb['delta']:+.9f}，95% BM-cluster CI={tb['cluster_ci95']}；对置换 Δ={tp['delta']:+.9f}。预注册教师进入条件未通过。

CPU ridge 学生完成训练和实际六预算评价。其 grade MAE **{grades['ALL']['models']['student']['mae']:.6f}**，fit-mean **{grades['ALL']['models']['fit_mean']['mae']:.6f}**；提升很小。学生低预算先验比同容量 base 的来源 ALC 改善 **{-pb['delta']:.9f}**（CI={pb['cluster_ci95']}），比置换学生改善 **{-pp['delta']:.9f}**（CI={pp['cluster_ci95']}）。这些是小幅来源信号，不能抵消教师信息检验失败，也不能冒充完整外层收益。

## 原文、作者代码与适配边界

[Zhou et al., Nature 2026](https://www.nature.com/articles/s41586-026-10303-2) 原文 Methods 的 LLM annotators、Assessors and metrics 和补充 Table 23，及[固定作者仓库](https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation/tree/816321f327dfe66910b72b514156aaa296b15f00)是依据。`src/adele/analysis/prediction.py` 保留 subject-specific RF（100 trees、min_samples_split=50、entropy）、Benchmark grouped OOF、complete-case需求处理。原论文使用18维+UG和GPT-4o，而本轮是原4rubrics + Luna、无UG、跨subject共享的线性ridge先验及4个蒸馏head。**这是有原文依据的机制适配，不是原论文准确率的逐参数复现。** 不使用作者 battery 的 response 数据。

## 数据与合法性

- 原 R53 的128 distinct query items；436/512维度有效、108个四维完整。grade评价872条是各有效item-rubric在两个重叠pool中的暴露，只有110个至少一维有效的unique items；并不是872独立标注。源标注由root在看response结果前定稿；null不当成0。教师用完整case，学生按维度忽略null。同模型Luna的重复QC是重复性信息，不是人工校准的正确性。
- 三个原来源pool分别有316/260/248个pair，合计824 exposure、412 distinct pairs；25上限/BM、seed73、<=64query。三个pool重叠，不能称独立三次重复。教师子集有{teacher['exposures']} exposure、{teacher['unique_pairs']} distinct pairs，{len(teacher['by_benchmark'])}个BM；其分数不能和完整来源或5,242外层直接比较。
- 每个pool内原 `inner_halves` 双向BM OOF；student和response ridge仅fit半组。outer BM排除，fit IDs/BM清单在 `isolation_audit.json` 和各inner目录。响应训练pair-equal、pair内保留重复response权重。
- B0教师只评价已注释query。大多数support没有teachergrade，故**没有纯teacher B1或teacher局部六预算结果**。实际学生可给任何已有640表示的item计算四维，完整六预算单独评价。
- R56是原R59的OOF；R66 item是R67的来源原型，不是R67 outer-refit预测。两者训练样本/模型不同，只作强历史参照；需求增量由同数据644槽base/teacher/permutation和student比较支持。

## 实际分数与正确率

完整来源 R25 ALC={deploy['results']['R25']['alc']:.9f}；prior_student={deploy['results']['prior_student']['alc']:.9f}；R56_segment={deploy['results']['R56_segment']['alc']:.9f}；R66_item_segment={deploy['results']['R66_item_segment']['alc']:.9f}。学生先验仍弱于历史两种先验。其 B0/B1准确率为 {100*deploy['results']['prior_student']['accuracy_at_0_5'][0]:.3f}% / {100*deploy['results']['prior_student']['accuracy_at_0_5'][1]:.3f}%，R25对应为 {100*deploy['results']['R25']['accuracy_at_0_5'][0]:.3f}% / {100*deploy['results']['R25']['accuracy_at_0_5'][1]:.3f}%。准确率指p≥0.5对response outcome的分类正确率；Brier才是主要概率质量指标。

学生局部追加四维 ALC={deploy['results']['local_student']['alc']:.9f}，相对R25 Δ={lr['delta']:+.9f}，95% CI={lr['cluster_ci95']}；B0/B1逐元素保持R25。仅需求局部模型 ALC={deploy['results']['local_demand_only']['alc']:.9f}，更弱。数值变动极小，不作有实质高预算收益的主张。

完整六预算、BM-equal、三个pool、BM切片、accuracy、区间详见 [demand_pilot_tables.md](demand_pilot_tables.md) 和CSV。源数据被多轮使用，这些都是探索性结果。

## 机制理解与失败边界

1. 合法的需求标签不自动产生稳定的成功率关系。本轮教师分数显著恶化，说明这个小样本、四维、线性先验的映射在新BM上不可靠。可能原因包括跨BM成功率方向变化、小样本不稳定、遗漏视觉/上下文、rubric质量和表示尺度；没有额外实验来唯一识别原因，不能宣称ADeLe总体无效。
2. 当前学生几乎只能达到常数grade均值。平均MAE下降不到0.01级，置换学生也接近；不能说已经获得高保真需求提取器。每rubric/每BM误差见grade_group_metrics.csv。
3. 学生是已有MiniLM/metadata的确定性映射。运行时没有独立新增raw information；小幅先验改善可以来自教师监督诱导的正则化/特征重加权。教师本身未通过独立信息对照，不能把学生轻微收益解释为已证实“认知需求结构”。
4. 需求teacher可看完整规范化文本，而MiniLM输入实际只留最多126内容tokens（前96+后30）加2特殊tokens；canonical输入还受20,000字符上限。原始长文本中丢失的信息无法由这个学生恢复。长度统计由R73最终质量审计提供，不能用模型默认256token替代真实代码窗口。
5. 教师的BM内联合置换只有{scope['teacher_permutation_changed_fraction']:.1%}完整items改变；学生逐维置换{scope['student_permutation_changed_fraction']:.1%}有效grades改变。同BM同等级项不改变，控制保留BM级分布；它检验的是实例对应关系，不是抹除全部任务类型信息。

## 回归与停止决定

prior_student 相对R25最差BM：{pr['worst_benchmark']} Δ={pr['worst_delta']:+.9f}；>=.005回归为 {pr['regression_ge_005']}。teacher对base最差BM：{tb['worst_benchmark']} Δ={tb['worst_delta']:+.9f}。

冻结合同要求教师与部署候选共同通过来源进入条件。两个部署候选形式上通过小幅来源比较，但教师条件全部失败，故 `decision.json` 明确 **stop_at_pilot_keep_R25**。不运行外层、不扩大注释、不改ridge或维度寻优。No-Peer实际保留R25；本轮最强新增来源候选是prior_student，但没有晋级为新方案。

## 资源与工程产物

来源全部训练、预测、区间计算耗时 **{run['seconds']:.2f}秒**，CPU线程2。四个学生head压缩文件共 **{cost['grade_heads_bytes']} bytes**。独立fresh-encoding测量（不写旧cache）：MiniLM+metadata+4head单item中位 **{cost['median_single_seconds']*1000:.2f}ms**，8item batch中位 **{cost['median_batch8_seconds']*1000:.2f}ms**；模型初始化 **{cost['initialization_seconds']:.3f}秒**，首次单item warmup额外 **{cost['warmup_seconds']:.3f}秒**，不包含在warm中位延迟中。已计算640向量后，1000item四head中位 **{cost['heads_1000_median_seconds']*1000:.3f}ms**。包含/排除项、三次观测、固定8items见 `cpu_feature_benchmark.json`。这是本机微基准，不是Hosted吞吐承诺。

所有inner head与response prior以NumPy保存。`demand_student_cpu.py::load_demand_models` / `predict_demands` 可对新640向量推理；实际新item端到端入口为 `dual_track_cpu_features.CPUItemFeatures(threads=2).predict_demands(item, heads)`，只接收item_content/item_features、不查dataset和标签cache。8item实算特征与原cache最大误差{cost['historical_cached_feature_max_error']:.3g}，grade预测最大误差{cost['historical_cached_demand_max_error']:.3g}。本轮没有晋级，实际方案仍使用原R25。没有要求调用Luna才能完成学生推断。没有新模型下载、GPU或收费API。

## 重放与异常记录

保存全teacher/full-source probabilities、counts、successes、subject/item/variant identity、fit清单和代码snapshot。重放 **{verified['saved_probability_values']:,}** 个概率值，Brier/accuracy误差最大 **{verified['max_brier_or_accuracy_replay_error']:.3g}**；历史R25重建误差≤{verified['historical_r25_max_probability_replay_error']:.3g}。R25/R56/R66在原source全query上的分数复用核对误差≤{verified['R25_R56_R66_original_source_score_replay_max_error']:.3g}。

首次运行在任何fit前因R66 query metadata与variant数组顺序的错误断言中止；原 `contract/` 保留，解释见 `demand_pilot_IDENTITY_AMENDMENT.md`，修复后 `contract_v2/` 冻结。不是重调参数，也没有把失败运行当效果结果。历史资产未修改。
'''
    (REPORT/'demand_pilot_findings.md').write_text(findings,encoding='utf8')
    commands='''# R74 复现与文件入口

工作目录：`D:\\nips-comp\\nips-comp\\paiec_baseline`。CPU 2线程；无新增依赖。

```powershell
& D:\\nips-comp\\paiec-runtime-win64-py312-cpu\\python.ps1 D:\\nips-comp\\nips-comp\\research_loop\\demand_pilot_run.py
& D:\\nips-comp\\paiec-runtime-win64-py312-cpu\\python.ps1 D:\\nips-comp\\nips-comp\\research_loop\\demand_pilot_report.py
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
'''
    (REPORT/'demand_pilot_reproduction.md').write_text(commands,encoding='utf8')
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
    files += [p for p in REPORT.glob('demand_pilot_*') if p.is_file()]+list(figs.glob('*.png'))
    files += [REPO/'research_loop/demand_pilot_report.py',REPO/'research_loop/demand_pilot_run.py',REPO/'research_loop/demand_student_cpu.py',REPO/'research_loop/demand_pilot_SPEC.md']
    files += [REPO/'research_loop/dual_track_cpu_features.py']
    write_json(OUT/'delivery_manifest.json',{'files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files],
        'status':'complete_source_pilot_stopped_before_outer','files_count':len(files)})
    print(json.dumps({'replay':verified,'cost':cost,'findings':str(REPORT/'demand_pilot_findings.md')}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
