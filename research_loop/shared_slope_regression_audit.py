"""Describe frozen severe regressions; never use this outer audit for tuning."""
from historical_positive_common import *
import pandas as pd

def main():
    root=ROOT/'R76/full';summary=read(root/'summary.json');records=[]
    for folder in sorted((root/'outer').glob('fold*/*')):
        name=folder.name+'/formatted_tables'
        if name not in ['dave/formatted_tables','cellverse/formatted_tables','cruxeval/formatted_tables','egoschema/formatted_tables']:continue
        for ip in sorted(folder.glob('index_*.json')):
            suffix=ip.stem.split('_')[-1]
            with np.load(folder/f'predictions_{suffix}.npz') as a:
                p=a['R25'];c=a['counts'];s=a['successes'];methods={m:a[m] for m in ['R72_pooled3','R76_peer1','R76_peer2','R76_peer3','R76_partial_selected','R76_partial_perm']}
            for idx in read(ip):
                lo=idx['start'];hi=lo+idx['size'];cc=c[lo:hi,None];ss=s[lo:hi,None];pp=p[lo:hi]
                for method,new in methods.items():
                    d=new[lo:hi]-pp
                    align=2*((cc*pp-ss)*d).sum(0)/cc.sum();cost=(cc*d*d).sum(0)/cc.sum()
                    for j,n in enumerate(BUDGETS):records.append({'benchmark':name,'subject_id':idx['subject_id'],'seed':idx['seed'],'method':method,'budget':n,
                        'alignment_term':float(align[j]),'variance_cost':float(cost[j]),'delta_brier':float(align[j]+cost[j])})
    df=pd.DataFrame(records);df.to_csv(root/'severe_regression_decomposition.csv',index=False)
    group=df.groupby(['benchmark','method','budget'])[['alignment_term','variance_cost','delta_brier']].mean().reset_index()
    group.to_csv(root/'severe_regression_summary.csv',index=False)
    agg=[]
    for (bm,m),g in group.groupby(['benchmark','method']):
        g=g.set_index('budget').loc[list(BUDGETS)]
        vals={k:float(g[k].to_numpy()@ALC_WEIGHTS) for k in ['alignment_term','variance_cost','delta_brier']}
        assert abs(vals['delta_brier']-(summary['by_benchmark'][bm]['alc_mean'][m]-summary['by_benchmark'][bm]['alc_mean']['R25']))<1e-12
        agg.append({'benchmark':bm,'method':m,**vals})
    write_json(root/'severe_regression_analysis.json',agg)
    print(json.dumps([v for v in agg if v['benchmark'].startswith('dave/')]))

if __name__=='__main__':main()
