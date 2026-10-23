"""Post-hoc explanatory statistics, never fed back to source coefficients."""
import pandas as pd
from historical_positive_common import *

raw=pd.read_csv(ROOT/'R70/full/decomposition.csv');raw=raw[raw.method=='R64']
global_budget=raw.groupby('budget')[['A','cost']].mean().reindex(BUDGETS)
alpha=np.clip(-global_budget.A/np.maximum(global_budget.cost,1e-30),0,1)
global_bound=float((2*alpha*global_budget.A+alpha**2*global_budget.cost).to_numpy()@ALC_WEIGHTS)
fold_bound=0.;foldwise=[]
for f in range(3):
    df=raw[raw.fold==f].groupby('budget')[['A','cost']].mean().reindex(BUDGETS);aa=np.clip(-df.A/np.maximum(df.cost,1e-30),0,1)
    delta=float((2*aa*df.A+aa**2*df.cost).to_numpy()@ALC_WEIGHTS);fold_bound+=(raw.fold==f).mean()*delta;foldwise.append({'fold':f,'oracle_alpha':aa.tolist(),'delta':delta})
bm=[]
for name,df in raw.groupby('benchmark'):
    v=df.groupby('budget')[['A','cost']].mean().reindex(BUDGETS);aa=np.clip(-v.A/np.maximum(v.cost,1e-30),0,1).to_numpy()[2:]
    bm.append({'benchmark':name,'active':bool((v.cost>1e-14).any()),'monotone':bool(np.all(np.diff(aa)<=1e-10))})
parts=[]
for path in sorted((ROOT/'R71/full/outer').glob('fold*/*/decomposition.csv')):
    df=pd.read_csv(path);g=df.groupby(['method','budget']).agg(alignment=('alignment','sum'),cost=('cost','sum'),net=('delta_brier','sum'),n=('delta_brier','size')).reset_index();parts.append(g)
dec=pd.concat(parts).groupby(['method','budget']).sum().reset_index()
for k in ['alignment','cost','net']:dec[k]/=dec.n
dec['weight']=dec.budget.map(dict(zip(BUDGETS,ALC_WEIGHTS)))
for k in ['alignment','cost','net']:dec[k]*=dec.weight
costs=dec.groupby('method')[['alignment','cost','net']].sum().to_dict('index')
g=pd.read_csv(ROOT/'R71/full/group_alignment_by_benchmark.csv');direction=[]
for n in [3,7,15,31]:
    data=g[g.budget==n].copy();data['conditional_net']=(2*data.A+data.C)/np.maximum(data.response_share,1e-30)
    pivot=data.pivot(index='benchmark',columns='group',values='conditional_net');share=data.pivot(index='benchmark',columns='group',values='response_share')
    eligible=(share[0]>1e-8)&(share[1]>1e-8);diff=(pivot[1]-pivot[0])[eligible]
    direction.append({'budget':n,'BMs_with_both_groups':int(eligible.sum()),'agreement_group_better':int((diff<-1e-10).sum()),'disagreement_group_better':int((diff>1e-10).sum()),'approximately_equal':int((abs(diff)<=1e-10).sum())})
result={'scope':'posthoc descriptive only; no coefficient update','oracle_global_budget_delta':global_bound,'oracle_fold_budget_delta':float(fold_bound),'oracle_by_fold':foldwise,
    'active_benchmarks':sum(v['active'] for v in bm),'nonincreasing_oracle_active':sum(v['active'] and v['monotone'] for v in bm),'budget_trend_by_benchmark':bm,
    'R71_ALC_decomposition':costs,'neighbor_direction_by_budget':direction,'finished':now()}
write_json(ROOT/'R71/full/mechanism_diagnostics.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['budget_trend_by_benchmark']}),flush=True)
