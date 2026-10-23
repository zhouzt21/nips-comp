"""Generate Track B tables from replayed artifacts, without handwritten numbers."""
from historical_positive_common import *
import pandas as pd

def fmt(x):return f'{x:.9f}'
def main():
    full=ROOT/'R76/full';s=read(full/'summary.json');comparisons=read(full/'additional_comparisons.json');cost=pd.read_csv(full/'label_cost_summary.csv')
    lines=['# R75/R76 Shared-Peer 数值汇总','','所有 R76 分数都是 `conditional_simulation`；共享方法的标签成本另列。准确率为 `p>=0.5` 对 response outcome 的分类准确率，不是被评 Subject 的原始正确率。','','## 完整评价：5242 pairs × 3 seeds × 六预算','','|方法|Pair-equal ALC↓|Benchmark-equal ALC↓|ΔR25|ΔR25 95% cluster CI|','|---|---:|---:|---:|---|']
    for m,v in s['results'].items():
        if m=='R25':delta=0.;ci=[0.,0.]
        else:delta=comparisons[m]['R25']['delta'];ci=comparisons[m]['R25']['cluster_ci95']
        lines.append(f'|{m}|{fmt(v["alc_mean"])}|{fmt(s["benchmark_equal_alc"][m])}|{fmt(delta)}|[{fmt(ci[0])}, {fmt(ci[1])}]|')
    lines+=['','`R72_*` 沿用旧 hash cohort；`R76_peer*` 是冻结 seed7501 随机 cohort；`R76_alt3` 是第二个冻结随机组成。不得按外层结果挑选随机种子。','', '## 各预算 Brier 与准确率','','|方法|预算|Brier↓|准确率|','|---|---:|---:|---:|']
    for m in ['R25','R72_pooled3','R72_diff3','R76_peer1','R76_peer2','R76_peer3','R76_alt3','R76_partial_selected']:
        for j,n in enumerate(BUDGETS):lines.append(f'|{m}|{n}|{fmt(s["results"][m]["brier_mean"][j])}|{s["results"][m]["accuracy_at_0_5"][j]:.6%}|')
    lines+=['','## 三个外层 fold 与主 cohort 比较','','|方法|fold0 ΔR25|fold1 ΔR25|fold2 ΔR25|Δ主cohort fully pooled|95% cluster CI|','|---|---:|---:|---:|---:|---|']
    for m in ['R76_peer1','R76_peer2','R76_peer3','R76_missing','R76_uneven','R76_alt3','R76_partial_selected']:
        fd=comparisons[m]['R25']['fold_delta']
        c=comparisons[m].get('R76_peer3',{'delta':0.,'cluster_ci95':[0.,0.]})
        lines.append('|'+m+'|'+'|'.join(fmt(v) for v in fd)+f'|{fmt(c["delta"])}|[{fmt(c["cluster_ci95"][0])}, {fmt(c["cluster_ci95"][1])}]|')
    lines+=['','## 实际标签成本（每个预测请求可见标签，跨 pair/seed 均值）','','|场景|own预算|平均peer标签|平均有效peer数|平均总标签|','|---|---:|---:|---:|---:|']
    for row in cost.itertuples(index=False):lines.append(f'|{row.method}|{row.budget}|{row.peer_labels:.4f}|{row.active_peers:.4f}|{row.visible_total:.4f}|')
    lines+=['','## Benchmark 切片','','|Benchmark|pairs|R25 ALC|R72 pooled3 ALC|R76主cohort pooled3 ALC|partial8 ALC|partial−pooled|','|---|---:|---:|---:|---:|---:|---:|']
    for b,v in s['by_benchmark'].items():
        a=v['alc_mean'];lines.append(f'|{b}|{v["pairs"]//3}|{fmt(a["R25"])}|{fmt(a["R72_pooled3"])}|{fmt(a["R76_peer3"])}|{fmt(a["R76_partial_selected"])}|{fmt(a["R76_partial_selected"]-a["R76_peer3"])}|')
    lines+=['','## 来源受控消融（Benchmark去重，固定取首个合法 source context）','','|方法|对照|ΔALC|Benchmark-equal Δ|95% cluster CI|','|---|---|---:|---:|---|']
    for c in read(ROOT/'R75/deduplicated_source/factor_comparisons.json'):
        lines.append(f'|{c["method"]}|{c["baseline"]}|{fmt(c["delta"])}|{fmt(c["benchmark_equal_delta"])}|[{fmt(c["cluster_ci95"][0])}, {fmt(c["cluster_ci95"][1])}]|')
    lines+=['','去重约定优先使用 source folds 0/1；这两处 R25/R36 的有效 ridge 恰好相同，故上述 ridge 行为0并不否定 fold2 的正则效应。完整选择环境对照如下，三个 source 环境有重复 Benchmark，不作独立复制统计。','','|选择fold|正则对照|ΔALC|Benchmark-equal Δ|95% CI|','|---|---|---:|---:|---|']
    for c in read(ROOT/'R75/factor_comparisons_by_selection_fold.json'):
        if 'R36ridge' not in c['method']:continue
        lines.append(f'|{c["selection_fold"]}|{c["method"]}−{c["baseline"]}|{fmt(c["delta"])}|{fmt(c["benchmark_equal_delta"])}|[{fmt(c["cluster_ci95"][0])}, {fmt(c["cluster_ci95"][1])}]|')
    lines+=['','来源评分只用于选择；它与完整公开评价共享数据来源，不能当成独立确认结果。','','## 异质性入口（支持标签，不使用query成功标签）','','|选择fold|实际来源BM|可诊断BM|超过null95且可重复BM|mean H|null mean H 95%|partial source ΔALC|','|---|---:|---:|---:|---:|---:|---:|']
    coverage={v['selection_fold']:v for v in read(ROOT/'R75/source_coverage.json')}
    for f,v in read(ROOT/'R75/selection.json').items():lines.append(f'|{f}|{coverage[int(f)]["actually_scored_benchmarks"]}|{v["diagnosed_benchmarks"]}|{v["positive_heterogeneity_benchmarks"]}|{fmt(v["mean_H"])}|{fmt(v["mean_null95"])}|{fmt(v["partial_source_delta"])}|')
    lines+=['','32 次 fitted-null 只是小型探索诊断；超过其95%分位不等价于经多重比较校正的 p<.05。source folds 重复同一 Benchmark，不构成独立复制。','']
    (REPO/'reports/shared_slope_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    print('wrote reports/shared_slope_tables.md')

if __name__=='__main__':main()
