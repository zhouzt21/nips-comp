"""Generate numeric tables and standalone research figures, without selecting candidates."""
import pandas as pd
from historical_positive_common import *

def fmt(x):return f'{float(x):.9f}'
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',*['|'+'|'.join(map(str,r))+'|' for r in rows]])

def main():
    reports=REPO/'reports';figs=reports/'lexical_peer_figures';figs.mkdir(exist_ok=True)
    summaries={s:read(ROOT/s/'full/summary.json') for s in ['R70','R71','R72']}
    chosen=[('R70',m) for m in ['smoothed_mean_2','training_selected_smoothed','R25','R56_segment','R64','R70_independent','R70_monotone','R70_global']]
    chosen += [('R71','R71_grouped')]
    chosen += [('R72',m) for m in ['R72_nopeer','R72_same1','R72_same3','R72_pooled1','R72_pooled3','R72_diff3','R72_morelabels_mean3','R36_own_pooled3','R20_peer_pooled3']]
    lines=['# R69–R72 数值表','', '全部主分数为公开探索评价5242pairs×3seeds×6budgets，pair等权、pair内response计数加权；同一64 query items/原query variants。R72全部为条件共享模拟，不能当Hosted分数。Accuracy按p≥.5，仅辅助。','', '## 总览','']
    overview=[];jsonrows=[]
    for stage,m in chosen:
        s=summaries[stage];v=s['results'][m];base=s['results']['R25']['alc_mean'];cmp=s['comparisons'].get(m,{}).get('R25',{})
        seed=[s['by_seed'][str(i)]['results'][m]['alc_mean'] for i in [0,1,2]]
        overview.append([stage,m,fmt(v['alc_mean']),fmt(v['alc_mean']-base),fmt(s['benchmark_equal_alc'][m]),fmt(np.std(seed,ddof=1)),str([round(x,9) for x in cmp.get('cluster_ci95',[])])])
        jsonrows.append({'stage':stage,'method':m,'alc':v['alc_mean'],'delta_R25':v['alc_mean']-base,'benchmark_equal_alc':s['benchmark_equal_alc'][m],
            'seed_std':float(np.std(seed,ddof=1)),'comparison_R25':cmp,'brier':v['brier_mean'],'accuracy':v['accuracy_at_0_5']})
    lines.append(table(['Stage','Method','ALC','ΔR25','BM-equal ALC','3seed SD','BM-cluster95%CI ΔR25'],overview))
    lines+=['','## 六预算 Brier','',table(['Method',*map(str,BUDGETS)],[[m,*map(fmt,summaries[s]['results'][m]['brier_mean'])] for s,m in chosen]),'',
        '## 六预算预测正确率（%）','',table(['Method',*map(str,BUDGETS)],[[m,*[f'{x*100:.4f}' for x in summaries[s]['results'][m]['accuracy_at_0_5']]] for s,m in chosen]),'',
        '## 三fold、alignment_faking 与最大回归','']
    foldrows=[]
    for s,m in chosen:
        if m=='R25':continue
        c=summaries[s]['comparisons'][m]['R25'];foldrows.append([m,*map(fmt,c['fold_deltas']),fmt(c['alignment_faking_contribution']),fmt(c['others_contribution']),c['worst_benchmark'].split('/')[0],fmt(c['worst_delta'])])
    lines.append(table(['Method','Fold0 Δ','Fold1 Δ','Fold2 Δ','alignment贡献','其他贡献','最差BM','最大回归'],foldrows))
    lines+=['','## 来源OOF系数与严格inner-half验证','']
    for f in range(3):
        coef=read(ROOT/f'R70/calibration/fold{f}/coefficients.json');s=read(ROOT/f'R70/calibration/fold{f}/summary.json')
        lines+=['',f'### Source pool {f}','',table(['Coefficient','B0','B1','B3','B7','B15','B31'],[[k,*map(fmt,coef[k])] for k in ['independent','monotone','global']]+[[f'group z={g}',*[fmt(v[g]) for v in coef['grouped']]] for g in [0,1]]),'',
            table(['CV method','Δ inner R25 ALC'],[[m,fmt(s['results'][m]['alc_mean']-s['results']['R25']['alc_mean'])] for m in ['R64','R70_independent','R70_monotone','R71_grouped']])]
    raw=pd.read_csv(ROOT/'R70/full/decomposition.csv');raw=raw[raw.method=='R64']
    fold_oracle=raw.groupby(['fold','budget'])[['A','cost']].mean().reset_index()
    fold_oracle['alpha']=np.clip(-fold_oracle.A/np.maximum(fold_oracle.cost,1e-30),0,1)
    fold_oracle['delta_brier']=2*fold_oracle.alpha*fold_oracle.A+fold_oracle.alpha**2*fold_oracle.cost
    fold_oracle.to_csv(ROOT/'R70/full/fold_budget_oracle_diagnostic.csv',index=False)
    total=0.
    for f in range(3):
        weight=float((raw.fold==f).mean());loss=float(fold_oracle[fold_oracle.fold==f].sort_values('budget').delta_brier.to_numpy()@ALC_WEIGHTS);total+=weight*loss
    global_raw=raw.groupby('budget')[['A','cost']].mean().reindex(BUDGETS);alpha=np.clip(-global_raw.A/np.maximum(global_raw.cost,1e-30),0,1)
    global_delta=float((2*alpha*global_raw.A+alpha**2*global_raw.cost).to_numpy()@ALC_WEIGHTS)
    write_json(ROOT/'R70/full/oracle_family_bounds.json',{'scope':'POSTHOC diagnostic only; not model selection','global_budget_alpha_delta':global_delta,'fold_budget_alpha_delta':total,'global_budget_alc':summaries['R70']['results']['R25']['alc_mean']+global_delta,'fold_budget_alc':summaries['R70']['results']['R25']['alc_mean']+total})
    by=raw.groupby(['benchmark','budget'])[['A','cost']].mean().reset_index();by['posthoc_alpha']=np.clip(-by.A/np.maximum(by.cost,1e-30),0,1)
    by.to_csv(reports/'lexical_peer_budget_by_benchmark.csv',index=False)
    trends=[]
    for b,df in by[by.budget>=3].groupby('benchmark'):
        aa=df.sort_values('budget').posthoc_alpha.to_numpy();active=bool((df.cost>1e-14).any())
        trends.append({'benchmark':b,'active':active,'nonincreasing_oracle':bool(np.all(np.diff(aa)<=1e-10)),'alpha_B3':aa[0],'alpha_B31':aa[-1]})
    write_json(ROOT/'R70/full/budget_trend_diagnostic.json',{'scope':'posthoc descriptive, not calibration','benchmarks':trends,'active':sum(v['active'] for v in trends),'nonincreasing_active':sum(v['active'] and v['nonincreasing_oracle'] for v in trends)})
    oracle=pd.read_csv(ROOT/'R70/full/outer_oracle_diagnostic.csv')
    lines+=['','## 原R64残差成本（外层事后诊断，不作系数拟合）','',table(['Budget','A','C','oracle α'],[[int(v.budget),fmt(v.A),fmt(v.cost),fmt(v.posthoc_oracle_alpha)] for v in oracle.itertuples(index=False)])]
    cg=read(ROOT/'R71/full/control_analysis.json')
    lines+=['','## 支持对应关系破坏与等能量控制','',table(['Method','正确增量 vs R25','置换后增量 vs 对应置换R25均值','置换减正确'],[[m,fmt(cg[m]['true_increment']['delta']),fmt(np.mean([v['delta'] for v in cg[m]['permuted_increment']])),fmt(cg[m]['permutation_minus_true_mean'])] for m in ['R64','R70_independent','R70_monotone','R71_grouped']]),'',
       table(['Control','R71−随机选择 ΔALC','95%CI'],[[k,fmt(v['delta']),str(v['cluster_ci95'])] for k,v in cg['random_energy'].items()])]
    groups=pd.read_csv(ROOT/'R71/full/group_alignment_summary.csv')
    lines+=['','## R71 可见分组的原始词法信息（事后）','',table(['Budget','z','A','C','response share','oracle α'],[[int(v.budget),int(v.group),fmt(v.A),fmt(v.C),fmt(v.response_share),fmt(v.posthoc_alpha)] for v in groups.itertuples(index=False)])]
    parts=[]
    for p in sorted((ROOT/'R71/full/outer').glob('fold*/*/decomposition.csv')):
        df=pd.read_csv(p);part=df.groupby(['method','budget']).agg(alignment=('alignment','sum'),cost=('cost','sum'),delta_brier=('delta_brier','sum'),records=('delta_brier','size')).reset_index();parts.append(part)
    dec=pd.concat(parts).groupby(['method','budget']).sum().reset_index()
    for k in ['alignment','cost','delta_brier']:dec[k]/=dec.records
    dec.to_csv(ROOT/'R71/full/decomposition_summary.csv',index=False)
    alc=dec.copy();alc['weight']=alc.budget.map(dict(zip(BUDGETS,ALC_WEIGHTS)))
    for k in ['alignment','cost','delta_brier']:alc[k]*=alc.weight
    alc=alc.groupby('method')[['alignment','cost','delta_brier']].sum().reset_index();alc.to_csv(ROOT/'R71/full/decomposition_alc.csv',index=False)
    lines+=['','## R71 修正收益与位移成本的ALC分解','',table(['Method','2A alignment','C cost','净ΔR25'],[[v.method,fmt(v.alignment),fmt(v.cost),fmt(v.delta_brier)] for v in alc.itertuples(index=False)])]
    coverage=pd.read_csv(ROOT/'R72/coverage_summary.csv')
    lines+=['','## R72 条件输入覆盖率','',table(['情景peer上限','Budget','实际平均peer','peer标签数','同输入response覆盖%'],[[int(v.scenario_peers),int(v.budget),f'{v.actual_peers:.4f}',f'{v.peer_labels:.4f}',f'{100*v.same_input_response_fraction:.4f}'] for v in coverage.itertuples(index=False)])]
    bmc=[]
    for audit in read(ROOT/'R72/coverage_summary.json')['benchmarks']:
        folder=ROOT/'R72/contexts'/audit['benchmark'].split('/')[0];cov=pd.DataFrame(read(folder/'coverage.json'))
        vals=cov[(cov.scenario_peers==3)&(cov.budget==31)]
        bmc.append([audit['benchmark'].split('/')[0],audit['subjects'],audit['ambiguous_subjects_excluded_from_sharing'],audit['no_peer_subjects'],audit['same_visible_key_multiple_internal_items'],f'{100*vals.same_input_response_fraction.mean():.3f}'])
    lines+=['','### 每Benchmark共享机会与歧义','',table(['BM','subjects','重复配置排除','无peer subjects','可见key对应多个内部item数','B31覆盖%'],bmc)]
    cc=read(ROOT/'R72/full/control_analysis.json');lines+=['','## R72 正确peer对应关系 − 置换关系','',table(['Method','Permutation','ΔALC 正确−置换','95%CI'],[[m,k,fmt(v['delta']),str(v['cluster_ci95'])] for m,d in cc.items() for k,v in d.items()])]
    ac=read(ROOT/'R72/full/additional_comparisons.json')
    pairs=[('R72_pooled3',b) for b in ['R25','R56_segment','R36_own_pooled3']]+[('R72_diff3','R72_pooled3')]
    lines+=['','## R72 关键配对差值','',table(['Method','Baseline','Delta ALC','BM-equal delta','Cluster95%CI'],[[m,b,fmt(ac[m][b]['delta']),fmt(ac[m][b]['benchmark_equal_delta']),str(ac[m][b]['cluster_ci95'])] for m,b in pairs])]
    slices=pd.read_csv(ROOT/'R72/full/matched_slice_summary.csv');slices['weight']=slices.budget.map(dict(zip(BUDGETS,ALC_WEIGHTS)))
    for k in ['alignment','cost','delta_contribution']:slices[k]*=slices.weight
    slice_alc=slices.groupby(['method','matched'])[['alignment','cost','delta_contribution']].sum().reset_index()
    slice_alc.to_csv(ROOT/'R72/full/matched_slice_alc.csv',index=False)
    lines+=['','## R72 覆盖切片对总ALC的贡献','',
        'matched指至少一个peer前缀中唯一出现、可用于same-input residual的完整输入键；False可能包含被保守拒绝的重复键。diff3则移除所有同键记录，故False切片与pooled3可能有极小差异。各切片按完整pair分母计权，表中是总体贡献，不是切片内条件均值。','',
        table(['Method','matched','2A','C','Delta contribution'],[[v.method,v.matched,fmt(v.alignment),fmt(v.cost),fmt(v.delta_contribution)] for v in slice_alc.itertuples(index=False)])]
    gates=read(ROOT/'R72/full/gate.json')
    lines+=['','## R72 数值门槛与部署状态','',
        '数值门槛使用两冻结基线、总体改善至少.005、三fold改善、BM-equal改善、clusterCI上界<0和既有.03严重回归标记；不代替真实runtime验证、逐BM回归解释或无偏确认。','',
        table(['Method','Numerical gate','Hosted','Submission'],[[m,v['numerical_gate'],v['hosted_deployment'],v['official_submission']] for m,v in gates.items()])]
    lines+=['','## 逐 Benchmark ALC','']
    bm_methods=[('R70','R25'),('R70','R56_segment'),('R70','R64'),('R70','R70_independent'),('R71','R71_grouped'),('R72','R72_same3'),('R72','R72_pooled3'),('R72','R72_diff3')]
    names=sorted(summaries['R70']['by_benchmark'])
    lines.append(table(['Benchmark','pairs',*[m for s,m in bm_methods]],[[b.split('/')[0],summaries['R70']['by_benchmark'][b]['pairs']//3,*[fmt(summaries[s]['by_benchmark'][b]['alc_mean'][m]) for s,m in bm_methods]] for b in names]))
    lines+=['','## 数值来源','', '各阶段 `R70–R72/full/metrics.csv`、`benchmark_metrics.csv`、`paired_scores.csv` 为全方法机器表；additional_comparisons.json含R56/R70/R36配对CI，replay_validation.json为保存概率重计分。原结果和错误控制批次分别保留，错误批次未混入表。']
    (reports/'lexical_peer_evidence_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    pd.concat([pd.read_csv(ROOT/s/'full/metrics.csv').assign(stage=s) for s in summaries]).to_csv(reports/'lexical_peer_evidence_metrics.csv',index=False)
    write_json(reports/'lexical_peer_evidence_summary.json',jsonrows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':150})
    fig,ax=plt.subplots(figsize=(7,4))
    for f in range(3):
        co=read(ROOT/f'R70/calibration/fold{f}/coefficients.json');ax.plot([3,7,15,31],co['independent'][2:],'o-',label=f'Source OOF fold {f}')
    ax.plot([3,7,15,31],oracle.posthoc_oracle_alpha.to_numpy()[2:],'k--',label='Outer oracle (diagnostic only)')
    ax.set(xlabel='Support label budget',ylabel='Lexical coefficient alpha',ylim=(-.03,1.08),title='Source calibration does not recover the outer budget trend');ax.legend();fig.tight_layout()
    fig.savefig(figs/'budget_coefficients.png');fig.savefig(figs/'budget_coefficients.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for k in [1,3]:
        df=coverage[coverage.scenario_peers==k];axes[0].plot(df.budget,100*df.same_input_response_fraction,'o-',label=f'Up to {k} peers')
    axes[0].set(xlabel='Budget',ylabel='Response coverage (%)',title='Conditional same-input coverage');axes[0].legend()
    for m in ['R72_same3','R72_pooled3','R72_diff3','R72_morelabels_mean3']:
        v=summaries['R72']['comparisons'][m]['R25']['per_budget_delta'];axes[1].plot(BUDGETS,v,'o-',label=m.replace('R72_',''))
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].set(xlabel='Budget',ylabel='Brier delta vs R25',title='Conditional peer evidence');axes[1].legend(fontsize=8);fig.tight_layout()
    fig.savefig(figs/'peer_coverage_and_effect.png');fig.savefig(figs/'peer_coverage_and_effect.svg');plt.close(fig)
    print(json.dumps({'tables':str(reports/'lexical_peer_evidence_tables.md'),'rows':len(jsonrows),'trend_active':sum(v['active'] for v in trends),'trend_monotone':sum(v['active'] and v['nonincreasing_oracle'] for v in trends)}),flush=True)

if __name__=='__main__':main()
