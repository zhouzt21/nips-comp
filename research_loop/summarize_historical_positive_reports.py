"""Export readable metric tables and one combined machine-readable report bundle."""
from historical_positive_common import *
import pandas as pd

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    folder=REPO/'reports';r62=read(ROOT/'R62/summary.json');r63=read(ROOT/'R63/summary.json');r64=read(ROOT/'R64/summary.json')
    stages={'R62':r62,'R63':r63}
    if (ROOT/'R64/full/summary.json').exists():stages['R64']=read(ROOT/'R64/full/summary.json')
    bundles={'full_outer':stages,'R64_source_pilot':r64,
             'R62_analysis':read(ROOT/'R62/analysis_audit.json'),'R63_analysis':read(ROOT/'R63/analysis_audit.json')}
    if (ROOT/'R64/full/analysis_audit.json').exists():bundles['R64_analysis']=read(ROOT/'R64/full/analysis_audit.json')
    write_json(folder/'historical_positive_metrics.json',bundles)
    frames=[];bms=[]
    for stage in stages:
        root=ROOT/stage if stage!='R64' else ROOT/'R64/full'
        frames.append(pd.read_csv(root/'metrics.csv').assign(stage=stage))
        bms.append(pd.read_csv(root/'benchmark_metrics.csv').assign(stage=stage))
    pd.concat(frames,ignore_index=True).to_csv(folder/'historical_positive_metrics.csv',index=False)
    pd.concat(bms,ignore_index=True).to_csv(folder/'historical_positive_benchmark_metrics.csv',index=False)
    chosen=[('R62',m) for m in ['smoothed_mean_2','training_selected_smoothed','R25','R54_full','R54_segment','R56_full','R56_segment','R61_support1_gate']]
    chosen += [('R63',m) for m in ['R43_query_best20','R55_resampled_query','R55_resampled_episode']]
    if 'R64' in stages:chosen += [('R64',m) for m in ['R46_raw_average','template_average']]
    lines=['# R62–R64 完整指标表','',
        '来源：各轮summary.json，由`research_loop/summarize_historical_positive_reports.py`生成。完整外层为5242 unique pairs×3 item-split seeds、36 Benchmarks；重复responses计数保留。Brier越低越好。正确率是以p≥.5预测二元response的正确率，不是subject模型本身的任务通过率。','',
        '## 外层总体','']
    rows=[]
    for stage,m in chosen:
        s=stages[stage];v=s['results'][m];c=s['comparisons'][m].get('R25')
        rows.append([stage,m,f"{v['alc_mean']:.9f}",f"{s['benchmark_equal_alc'][m]:.9f}",
            f"{v['alc_mean']-s['results']['R25']['alc_mean']:+.9f}",
            '['+', '.join(f'{x:+.9f}' for x in c['cluster_ci95'])+']' if c else '—'])
    lines += [table(['阶段','方法','pair ALC','Benchmark ALC','ΔR25','cluster95% ΔR25'],rows),'',
              '## 三折、主要Benchmark贡献和最大退化','']
    rows=[]
    for stage,m in chosen:
        c=stages[stage]['comparisons'][m].get('R25')
        if not c:continue
        rows.append([m,*[f'{d:+.9f}' for d in c['fold_deltas']],f"{c['alignment_faking_contribution']:+.9f}",
            f"{c['others_contribution']:+.9f}",c['worst_benchmark'].split('/')[0],f"{c['worst_delta']:+.9f}"])
    lines += [table(['方法','fold0','fold1','fold2','alignment加权贡献','其余加权贡献','最大退化BM','ΔALC'],rows),'',
        '贡献使用原5242pair分母；没有删除不利Benchmark后重新排名。Bootstrap以Benchmark为cluster、10000次、seed20261022，不能抵消自适应开发偏差。','',
        '## 六预算Brier','']
    lines += [table(['方法']+[f'B{n}' for n in BUDGETS],[[m]+[f'{v:.9f}' for v in stages[stage]['results'][m]['brier_mean']] for stage,m in chosen]),'',
        '## 六预算预测正确率','',
        table(['方法']+[f'B{n}' for n in BUDGETS],[[m]+[f'{v*100:.4f}%' for v in stages[stage]['results'][m]['accuracy_at_0_5']] for stage,m in chosen]),'',
        '## R64来源pilot（不与外层混排）','']
    rows=[]
    for f in range(3):
        s=read(ROOT/f'R64/pool{f}/summary.json')
        for m in ['R25','raw_lexical','template_lexical','R46_raw_average','template_average']:
            rows.append([f,m,s['unique_pairs'],f"{s['results'][m]['alc_mean']:.9f}",
                f"{s['results'][m]['alc_mean']-s['results']['R25']['alc_mean']:+.9f}",f"{s['benchmark_equal_alc'][m]:.9f}"])
    lines += [table(['池','方法','pairs','ALC','ΔR25','BM ALC'],rows),'',
        f"进入完整外层：`{r64['enter_outer']}`；{r64['source_exposures']}次来源暴露对应{r64['distinct_pairs']}不同pair。各池不是独立复现。",'',
        '全量每个Benchmark、逐pair及各seed结果保存在各轮CSV/JSON中；本表只负责展示，不进行任何模型重选。','']
    (folder/'historical_positive_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote historical_positive_metrics.json/csv, benchmark_metrics.csv, tables.md')

if __name__=='__main__':main()
