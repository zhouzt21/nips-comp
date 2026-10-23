"""Descriptive slices only; never selects a policy or changes a frozen gate."""
from shared_reliability_common import *
from analyze_support_sharing_reliability import clean

def main():
    r78=ROOT/'R78';d=pd.read_csv(r78/'deduplicated_diagnostic_metrics.csv')
    columns=[c for c in d if c.startswith(('support_brier_','query_brier_','query_accuracy_','z_'))]+['own_labels','peer_labels','active_peers']
    d.groupby(['scenario','budget'])[columns].mean().to_csv(r78/'budget_metrics.csv')
    d.groupby(['scenario','budget','benchmark'])[columns].mean().to_csv(r78/'benchmark_budget_metrics.csv')
    slices=[]
    for scene,g in d.groupby('scenario'):
        for n,part in g.groupby('budget'):
            groups={'all':part,'alignment':part[part.benchmark.str.startswith('alignment_faking/')],
                'without_alignment':part[~part.benchmark.str.startswith('alignment_faking/')],'dave':part[part.benchmark.str.startswith('dave/')]}
            for label,ss in groups.items():
                row={'scenario':scene,'budget':n,'slice':label,'pairs':ss[['benchmark','subject_id']].drop_duplicates().shape[0],
                    'benchmarks':ss.benchmark.nunique(),'support_partial_delta':ss.z_partial.mean(),'query_partial_delta':ss.query_delta_partial.mean(),
                    'query_pooled_delta':ss.query_delta_pooled.mean(),'unreliable_request_fraction':(ss.z_partial>=0).mean(),
                    'support_mean_delta':ss.z_mean.mean(),'query_partial_accuracy':ss.query_accuracy_partial.mean()}
                slices.append(row)
    pd.DataFrame(slices).to_csv(r78/'mechanism_slices.csv',index=False)
    r79=ROOT/'R79'
    if not (r79/'source_metrics.csv').exists():return
    s=pd.read_csv(r79/'source_metrics.csv');first=s.groupby('benchmark').selection_fold.min().to_dict()
    s=s[s.apply(lambda r:r.selection_fold==first[r.benchmark],axis=1)].copy();s.to_csv(r79/'deduplicated_source_metrics.csv',index=False)
    s['weight']=s.budget.map({15:.2,31:.1});s['delta_shared']=s.brier_partial-s.brier_own;s['closed']=s.gate==0
    s['wrong_close_cost']=np.where(s.closed&(s.delta_shared<0),-s.delta_shared*s.weight,0.)
    s['harm_avoided']=np.where(s.closed&(s.delta_shared>0),s.delta_shared*s.weight,0.)
    s['gate_delta']=(s.brier_gate-s.brier_partial)*s.weight
    count=s[['benchmark','subject_id','seed']].drop_duplicates().shape[0]
    assert abs((s.wrong_close_cost.sum()-s.harm_avoided.sum()-s.gate_delta.sum())/count)<1e-12
    result={'unique_trajectories':count,'budget_decisions':len(s),'eligible_policy_fraction':s.metric.notna().mean(),
        'closure_fraction':s.closed.mean(),'closure_fraction_when_peer_present':s.loc[s.active_peers>0,'closed'].mean(),
        'wrong_closed_requests':int((s.closed&(s.delta_shared<0)).sum()),'harmful_closed_requests':int((s.closed&(s.delta_shared>0)).sum()),
        'neutral_closed_requests':int((s.closed&(s.delta_shared==0)).sum()),
        'lost_benefit_ALC':s.wrong_close_cost.sum()/count,'avoided_harm_ALC':s.harm_avoided.sum()/count,
        'net_gate_minus_partial_ALC':s.gate_delta.sum()/count,
        'scope':'offline source decomposition using query outcomes; never inference features'}
    write_json(r79/'source_error_decomposition.json',clean(result))
    s.groupby(['benchmark','budget']).mean(numeric_only=True).to_csv(r79/'benchmark_budget_metrics.csv')
    s.groupby('budget').mean(numeric_only=True).to_csv(r79/'budget_metrics.csv')
    print(json.dumps(clean({'R78_main_slices':[r for r in slices if r['scenario']=='main'],'R79_decomposition':result})),flush=True)
if __name__=='__main__':main()
