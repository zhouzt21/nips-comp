"""Generate R65-R68 score tables, plots, and explicit research decisions."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from historical_positive_common import *

REPORT=REPO/'reports';FIG=REPORT/'transferable_prior_figures'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    r65=read(ROOT/'R65/descriptor_v2/summary.json');r66=read(ROOT/'R66/summary.json');r67=read(ROOT/'R67/summary.json')
    frames=[]
    d=pd.read_csv(ROOT/'R65/descriptor_v2/metrics.csv');d['stage']='R65';frames.append(d)
    frames.append(pd.read_csv(ROOT/'R66/metrics.csv'))
    d=pd.read_csv(ROOT/'R67/metrics.csv').rename(columns={'accuracy_at_0_5':'accuracy'});d['scope']='outer_all';d['stage']='R67';frames.append(d)
    df=pd.concat(frames,ignore_index=True);reference={(r.stage,r.scope,r.budget):r.brier for r in df[df.method=='R25'].itertuples()}
    df['delta_brier_R25']=[r.brier-reference[(r.stage,r.scope,r.budget)] for r in df.itertuples()]
    df.to_csv(REPORT/'transferable_prior_metrics.csv',index=False)
    metrics=[]
    for scope in ['all','alignment_faking','others']:
        for m in ['R25','R54_full','R56_full','source_constant_pair','source_constant_mixture']:
            v=[r for r in r65['metrics'] if r['scope']==scope and r['method']==m]
            metrics.append([scope,m,*[f'{x["brier"]:.9f}' for x in v],*[f'{x["accuracy"]*100:.4f}%' for x in v]])
    lines=['# R65–R68 实际指标与诊断表','','## R65：B0/B1（同一15726条轨迹）','',
        table(['范围','方法','B0 Brier','B1 Brier','B0预测正确率','B1预测正确率'],metrics),'',
        '预测正确率按p≥.5判为1，目标是response二元结果；R25 B0全部为.5，tie约定影响accuracy而不影响Brier。',
        '', '## R65：首标签的相对贡献','','ΔB1=ΔB0+[update_model−update_R25]；以下未乘ALC权重，负值改善。','']
    rows=[]
    for scope in ['all','alignment_faking','others']:
        cells={v['method']:v for v in r65['support_update'] if v['scope']==scope};base=cells['R25']
        for m in ['R54_full','R56_full']:
            v=cells[m];rows.append([scope,m,f'{v["B0"]-base["B0"]:+.9f}',f'{v["update_net"]-base["update_net"]:+.9f}',f'{v["B1"]-base["B1"]:+.9f}'])
    lines += [table(['范围','方法','原B0先验差','首标签更新相对差','B1净差'],rows),'','## R66：来源匹配消融','',
        '池0/1/2为316/260/248 pair，共824池暴露、412不同pair；pool间重叠，不是三次独立验证。只在来源B0选择epoch，以下为selection evidence。','']
    rows=[]
    for f in range(3):
        s=r66['folds'][str(f)]
        for m in ['R25','R56_segment','subject_segment','item_segment','source_constant_segment']:
            v=s['results'][m];rows.append([f,m,f'{v["brier_mean"][0]:.9f}',f'{v["brier_mean"][1]:.9f}',f'{v["alc_mean"]:.9f}',
                f'{v["alc_mean"]-s["results"]["R25"]["alc_mean"]:+.9f}',f'{s["benchmark_equal_alc"][m]:.9f}',
                f'{100*v["accuracy_at_0_5"][0]:.4f}%',f'{100*v["accuracy_at_0_5"][1]:.4f}%'])
    lines += [table(['池','方法','B0','B1','ALC','ΔR25','BM等权ALC','B0正确率','B1正确率'],rows),'',
        '## R67：全外层5242pairs ×3item seeds','']
    rows=[]
    for m in ['smoothed_mean_2','training_selected_smoothed','R25','R56_segment','R67_item_segment']:
        v=r67['results'][m];rows.append([m,f'{v["alc_mean"]:.9f}',f'{r67["benchmark_equal_alc"][m]:.9f}',*[f'{x:.9f}' for x in v['brier_mean']],*[f'{100*x:.4f}%' for x in v['accuracy_at_0_5']]])
    lines += [table(['方法','ALC','BM等权ALC']+[f'B{n}' for n in BUDGETS]+[f'Acc{n}' for n in BUDGETS],rows),'',
        '### R67配对差异与不确定性','']
    rows=[]
    for b,c in r67['comparisons']['R67_item_segment'].items():
        rows.append([b,f'{c["delta_alc"]:+.9f}',f'{c["benchmark_equal_delta"]:+.9f}',
            '/'.join(f'{x:+.9f}' for x in c['fold_deltas']),str(c['cluster_ci95']),f'{c["worst_delta"]:+.9f} ({c["worst_benchmark"].split("/")[0]})'])
    c=read(ROOT/'R67/versus_R56.json');rows.append(['R56_segment',f'{c["delta"]:+.9f}',f'{c["benchmark_equal_delta"]:+.9f}', '/'.join(f'{x:+.9f}' for x in c['fold_delta']),str(c['cluster_ci95']),'见逐BM表'])
    lines += [table(['对照','ΔALC','ΔBM ALC','三fold Δ','cluster95%CI','最大BM退化'],rows),'',
        'cluster bootstrap按Benchmark重采样、保留该BM全部pairs与seeds，10000次，seed20261022。公开数据已反复用于开发，CI不能消除选择偏差。','',
        '### R67逐item-split seed（不是模型初始化seed）','']
    lines += [table(['seed','R25','R56_segment','R67_item_segment'],[[s,*[f'{r67["by_seed"][s]["results"][m]["alc_mean"]:.9f}' for m in ['R25','R56_segment','R67_item_segment']]] for s in r67['by_seed']]),'']
    (REPORT/'transferable_prior_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    bm=pd.read_csv(ROOT/'R67/benchmark_metrics.csv');bm.to_csv(REPORT/'transferable_prior_benchmark_metrics.csv',index=False)
    decisions={}
    for baseline in ['smoothed_mean_2','training_selected_smoothed']:
        c=r67['comparisons']['R67_item_segment'][baseline]
        decisions[baseline]={'gain_ge_005':c['delta_alc']<=-.005,'three_folds_improve':all(d<0 for d in c['fold_deltas']),
            'benchmark_equal_improves':c['benchmark_equal_delta']<0,'cluster_CI_upper_lt_zero':c['cluster_ci95'][1]<0,
            'no_ge_03_regression':not c['regression_ge_03']}
    assert not all(all(v.values()) for v in decisions.values())
    write_json(ROOT/'R67/decision.json',{'status':'not_promoted','research_success':False,'baseline_gates':decisions,
        'R25_comparison':r67['comparisons']['R67_item_segment']['R25'],'R56_comparison':read(ROOT/'R67/versus_R56.json'),
        'stop':'No further item-only thresholds, mixtures, architecture or LR search on these outer results.',
        'submission':'unchanged; no official submission'})
    write_json(ROOT/'R67/status.json',{'status':'complete_not_promoted','finished_utc':now(),'pairs':5242,'seeds':[0,1,2],'new_fits':3})
    FIG.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for m,label in [('R25','R25'),('R56_segment','R56 segment'),('R67_item_segment','R67 item segment')]:
        axes[0].plot(range(6),r67['results'][m]['brier_mean'],marker='o',label=label)
    axes[0].set_xticks(range(6),BUDGETS);axes[0].set_xlabel('Support labels');axes[0].set_ylabel('Brier (lower is better)');axes[0].legend(fontsize=8);axes[0].set_title('Full outer: 5,242 pairs x 3 seeds')
    for i,m in enumerate(['R56_segment','subject_segment','item_segment']):
        ds=[r66['folds'][str(f)]['results'][m]['alc_mean']-r66['folds'][str(f)]['results']['R25']['alc_mean'] for f in range(3)]
        axes[1].bar(np.arange(3)+(i-1)*.24,ds,width=.24,label=m.replace('_segment',''))
    axes[1].axhline(0,color='black',lw=.7);axes[1].set_xticks(range(3),['Pool 0','Pool 1','Pool 2']);axes[1].set_ylabel('Delta ALC vs R25');axes[1].set_title('Source-selected ablations');axes[1].legend(fontsize=8)
    for i,m in enumerate(['R56_segment','R67_item_segment']):
        c=r67['comparisons'][m]['R25'];axes[2].bar(np.arange(2)+(i-.5)*.32,[c['alignment_faking_contribution'],c['others_contribution']],width=.32,label=m)
    axes[2].set_xticks(range(2),['alignment_faking','Other 35 benchmarks'],rotation=12);axes[2].axhline(0,color='black',lw=.7);axes[2].set_title('Outer weighted contributions');axes[2].set_ylabel('Contribution to delta ALC');axes[2].legend(fontsize=8)
    for ext in ['png','svg','pdf']:fig.savefig(FIG/f'mechanism_summary.{ext}',dpi=180)
    print(json.dumps({'report_tables':str(REPORT/'transferable_prior_tables.md'),'metrics_rows':len(df),'R67_decision':'not_promoted'}))

if __name__=='__main__':main()
