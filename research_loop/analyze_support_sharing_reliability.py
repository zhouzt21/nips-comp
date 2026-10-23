"""Independent saved-probability replay and predeclared grouped R78 diagnostics."""
import time
from collections import defaultdict
from scipy.stats import spearmanr
from shared_reliability_common import *
from support_sharing_diagnostic import METRICS

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.bool_):return bool(v)
    return v

def replay_source(folder):
    rows=read(folder/'diagnostics.json');lookup={(r['subject_id'],r['seed'],r['scenario'],r['budget']):r for r in rows}
    q=dict(np.load(folder/'query_predictions.npz'));m=dict(np.load(folder/'support_predictions.npz'));maxq=maxz=maxac=0.
    for idx in read(folder/'query_index.json'):
        original=idx['original'];a=idx['start'];b=a+idx['size']
        assert np.array_equal(q['counts'][a:b],original['counts']) and np.array_equal(q['successes'][a:b],original['successes'])
        for j,n in enumerate([15,31]):
            r=lookup[(original['subject_id'],original['seed'],idx['scenario'],n)]
            for k,p in q.items():
                if k in ['counts','successes']:continue
                br=brier(p[a:b,j],q['counts'][a:b],q['successes'][a:b]);ac=accuracy(p[a:b,j],q['counts'][a:b],q['successes'][a:b])
                maxq=max(maxq,abs(br-r['query_brier'][k]));maxac=max(maxac,abs(ac-r['query_accuracy'][k]))
    risks=defaultdict(lambda:defaultdict(list));allrisk=defaultdict(lambda:defaultdict(list));counts=defaultdict(list)
    for idx in read(folder/'support_index.json'):
        a=idx['start'];b=a+idx['size'];key=(idx['subject_id'],idx['seed'],idx['scenario'],idx['budget']);n=idx['budget']
        assert set(idx['fit_indices']).isdisjoint(idx['hold_indices'])
        assert sorted(idx['fit_indices']+idx['hold_indices'])==list(range(n))
        assert len(idx['hold_indices'])==b-a and all(v['n']<=n for v in idx['peer_prefixes'])
        y=m['truth'][a:b]
        for k in METRICS:
            base=m['meanbase'][a:b] if k=='mean' else m['own'][a:b]
            risks[(key,idx['rep'])][k].extend(((m[k][a:b]-y)**2-(base-y)**2).tolist())
        for k in ['own','pooled','partial','mean','meanbase','combined']:
            assert np.isfinite(m[k][a:b]).all() and np.all((m[k][a:b]>=0)&(m[k][a:b]<=1))
            allrisk[key][k].extend(((m[k][a:b]-y)**2).tolist())
        counts[(key,idx['rep'])].extend(idx['hold_indices'])
    for key,r in lookup.items():
        for rep in range(2):assert sorted(counts[(key,rep)])==list(range(r['budget']))
        for k in METRICS:
            rz=[float(np.mean(risks[(key,rep)][k])) for rep in range(2)]
            maxz=max(maxz,float(np.max(abs(np.array(rz)-r['repeat_z'][k]))),abs(float(np.mean(rz))-r['z'][k]))
        r['support_brier']={k:float(np.mean(v)) for k,v in allrisk[key].items()}
    assert maxq<1e-12 and maxz<1e-12 and maxac<1e-12
    return rows,{'folder':str(folder),'diagnostic_records':len(rows),'support_prediction_rows':len(m['truth']),
                 'query_variant_rows':len(q['counts']),'max_query_brier_error':maxq,'max_query_accuracy_error':maxac,'max_support_z_error':maxz}

def as_frame(rows):
    return pd.DataFrame([{**{k:r[k] for k in ['selection_fold','benchmark','subject_id','seed','scenario','budget','query_delta_partial','query_delta_pooled','mean_gap','own_labels','peer_labels','active_peers']},
                           **{f'z_{k}':r['z'][k] for k in METRICS},**{f'r{j}_{k}':r['repeat_z'][k][j] for k in METRICS for j in range(2)},
                           **{f'support_brier_{k}':v for k,v in r['support_brier'].items()},
                           **{f'query_brier_{k}':v for k,v in r['query_brier'].items()},
                           **{f'query_accuracy_{k}':v for k,v in r['query_accuracy'].items()}} for r in rows])

def collapsed(df):
    return df.groupby(['benchmark','subject_id'],as_index=False).mean(numeric_only=True)

def association(df,metric,target='query_delta_partial'):
    d=collapsed(df);z='z_'+metric;rho=[];within=[];per=[]
    for bm,g in d.groupby('benchmark'):
        value=float(spearmanr(g[z],g[target]).statistic) if len(g)>=3 and g[z].nunique()>1 and g[target].nunique()>1 else None
        if value is not None and np.isfinite(value):rho.append(value)
        neg=g[g[z]<0];pos=g[g[z]>=0];gap=float(neg[target].mean()-pos[target].mean()) if len(neg) and len(pos) else None
        if gap is not None:within.append(gap)
        per.append({'benchmark':bm,'pairs':len(g),'spearman':value,'reliable_pairs':len(neg),'unreliable_pairs':len(pos),'within_bm_group_gap':gap,
                    'support_delta_mean':float(g[z].mean()),'query_delta_mean':float(g[target].mean())})
    rng=np.random.default_rng(20261023)
    def interval(v):
        v=np.asarray(v,float)
        return np.quantile(v[rng.integers(len(v),size=(10000,len(v)))].mean(1),[.025,.975]).tolist() if len(v) else [None,None]
    neg=d[d[z]<0];pos=d[d[z]>=0];rest=d[~d.benchmark.str.startswith('alignment_faking/')]
    def groupgap(a):return float(a.loc[a[z]<0,target].mean()-a.loc[a[z]>=0,target].mean()) if (a[z]<0).any() and (a[z]>=0).any() else None
    usable=d[abs(d[target])>1e-12];truth=usable[target]<0;pred=usable[z]<0
    acc=float((truth==pred).mean()) if len(usable) else None
    ba=float(np.mean([((pred[truth==label])==label).mean() for label in [False,True]])) if truth.nunique()==2 else None
    repeat_rho=float(spearmanr(d['r0_'+metric],d['r1_'+metric]).statistic) if d['r0_'+metric].nunique()>1 and d['r1_'+metric].nunique()>1 else None
    restwithin=[v['within_bm_group_gap'] for v in per if not v['benchmark'].startswith('alignment_faking/') and v['within_bm_group_gap'] is not None]
    return clean({'pairs':len(d),'benchmarks':d.benchmark.nunique(),'estimable_spearman_benchmarks':len(rho),'benchmark_equal_spearman':float(np.mean(rho)) if rho else None,
        'spearman_cluster_ci95':interval(rho),'positive_spearman_benchmarks':sum(v>0 for v in rho),
        'direction_accuracy':acc,'direction_balanced_accuracy':ba,'nonneutral_direction_pairs':len(usable),
        'reliable_pairs':len(neg),'unreliable_pairs':len(pos),'reliable_benchmarks':neg.benchmark.nunique(),'unreliable_benchmarks':pos.benchmark.nunique(),
        'reliable_query_delta':neg[target].mean(),'unreliable_query_delta':pos[target].mean(),'pair_group_gap':groupgap(d),
        'within_benchmark_group_gap':float(np.mean(within)) if within else None,'group_gap_cluster_ci95':interval(within),
        'without_alignment_pair_gap':groupgap(rest),'without_alignment_bm_gap':float(np.mean(restwithin)) if restwithin else None,
        'mask_repeat_spearman':repeat_rho,'mask_repeat_sign_agreement':float(((d['r0_'+metric]<0)==(d['r1_'+metric]<0)).mean()),'per_benchmark':per})

def local_criteria(a):
    def negative(v):return v is not None and v<0
    return {'rho_CI_positive':a['spearman_cluster_ci95'][0] is not None and a['spearman_cluster_ci95'][0]>0,
        'pair_group_gap_negative':negative(a['pair_group_gap']),'bm_group_gap_negative':negative(a['within_benchmark_group_gap']),
        'without_alignment_pair_negative':negative(a['without_alignment_pair_gap']),'without_alignment_bm_negative':negative(a['without_alignment_bm_gap']),
        'group_sample_minimum':min(a['reliable_pairs'],a['unreliable_pairs'])>=10 and min(a['reliable_benchmarks'],a['unreliable_benchmarks'])>=5}

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    start=time.perf_counter();root=ROOT/'R78';allrows=[];dedup=[];seen=set();checks=[]
    for f in range(3):
        for folder in sorted((root/f'source_fold{f}').iterdir()):
            if not (folder/'complete.json').exists():continue
            rows,check=replay_source(folder);allrows+=rows;checks.append(check)
            if rows[0]['benchmark'] not in seen:dedup+=rows;seen.add(rows[0]['benchmark'])
    frame=as_frame(allrows);unique=as_frame(dedup);frame.to_csv(root/'diagnostic_metrics.csv',index=False);unique.to_csv(root/'deduplicated_diagnostic_metrics.csv',index=False)
    assoc={};pool={};bybudget={};pooledtarget={};perbm=[]
    for scenario in sorted(unique.scenario.unique()):
        assoc[scenario]={};bybudget[scenario]={}
        for metric in METRICS:
            a=association(unique[unique.scenario==scenario],metric);assoc[scenario][metric]=a
            perbm += [{'scenario':scenario,'metric':metric,**r} for r in a['per_benchmark']]
        for n in [15,31]:bybudget[scenario][str(n)]={m:association(unique[(unique.scenario==scenario)&(unique.budget==n)],m) for m in METRICS}
        pooledtarget[scenario]=association(unique[unique.scenario==scenario],'pooled','query_delta_pooled')
    for f in range(3):pool[str(f)]={m:association(frame[(frame.selection_fold==f)&(frame.scenario=='main')],m) for m in METRICS}
    decision=[];selected=None;perfold={}
    for m in METRICS:
        a=assoc['main'][m];tests=local_criteria(a);tests['positive_in_two_source_pools']=sum((pool[str(f)][m]['benchmark_equal_spearman'] or -999)>0 for f in range(3))>=2
        passed=all(tests.values());decision.append({'metric':m,'tests':tests,'pass':passed})
        if passed and selected is None:selected=m
    for f in range(3):
        choices=[m for m in METRICS if all(local_criteria(pool[str(f)][m]).values())]
        perfold[str(f)]={'eligible_metrics':choices,'first_eligible':choices[0] if choices else None,'scope':'only this fold legal source Benchmark outcomes'}
    result={'status':'support_consistency_predictive_source_entry' if selected else 'support_consistency_not_predictive',
            'descriptive_first_metric':selected,'tests':decision,'outer_fold_source_only_choices':perfold,
            'R79_diagnostic_entry':selected is not None,'R79_outer_authorized':False,
            'note':'Any R79 still requires nested source strategy versus matched shrinkage; no global metric may be applied across outer folds.'}
    write_json(root/'decision.json',result);write_json(root/'associations.json',clean({'deduplicated':assoc,'source_pools':pool,'per_budget':bybudget,'pooled_target':pooledtarget}))
    pd.DataFrame(perbm).to_csv(root/'benchmark_associations.csv',index=False)
    validation={'checks':checks,'diagnostic_records':len(allrows),'deduplicated_records':len(dedup),'source_trajectories':len(allrows)//10,
                'unique_reporting_trajectories':len(dedup)//10,'query_variant_rows':sum(c['query_variant_rows'] for c in checks),
                'support_prediction_rows':sum(c['support_prediction_rows'] for c in checks),
                'max_query_brier_error':max(c['max_query_brier_error'] for c in checks),'max_query_accuracy_error':max(c['max_query_accuracy_error'] for c in checks),
                'max_support_z_error':max(c['max_support_z_error'] for c in checks),'finished':now()}
    write_json(root/'replay_validation.json',validation);write_json(root/'analysis_execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'finished':now()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for ax,scene in zip(axes,['main','permuted']):
        vals=[assoc[scene][m]['benchmark_equal_spearman'] for m in METRICS];ci=np.array([assoc[scene][m]['spearman_cluster_ci95'] for m in METRICS],float)
        vals=np.asarray(vals,float);ax.errorbar(range(4),vals,yerr=np.array([vals-ci[:,0],ci[:,1]-vals]),fmt='o');ax.axhline(0,c='gray',ls='--')
        ax.set_xticks(range(4),METRICS,rotation=20);ax.set_title(scene);ax.set_ylabel('Mean within-Benchmark Spearman (95% CI)')
    fig.tight_layout();dest=REPO/'reports/shared_reliability_figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'support_query_association.png',dpi=180);plt.close(fig)
    check_freeze(root/'contract')
    print(json.dumps(clean({'decision':result,'main':{m:{k:v for k,v in assoc['main'][m].items() if k!='per_benchmark'} for m in METRICS},'replay':{k:v for k,v in validation.items() if k!='checks'}})),flush=True)
if __name__=='__main__':main()
