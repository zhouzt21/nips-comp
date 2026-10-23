"""R79 nested legal-source binary policy; outer run remains conditional."""
import time
from shared_reliability_common import *
from analyze_support_sharing_reliability import association,local_criteria,clean
from support_sharing_diagnostic import METRICS

METHODS=['own','pooled','partial','gate','matched_mean','matched_cost']

def choose(df):
    checks={m:local_criteria(association(df,m)) for m in METRICS}
    choices=[m for m in METRICS if all(checks[m].values())]
    return (choices[0] if choices else None),checks

def compare(df,base):
    # Only B15/B31 differ. Their exact contribution to the six-budget ALC is .2/.1.
    d=df.copy();d['delta']=(d.brier_gate-d['brier_'+base])*d.budget.map({15:.2,31:.1})
    pair=d.groupby(['benchmark','subject_id','seed']).delta.sum().reset_index()
    bm=pair.groupby('benchmark').delta.agg(['sum','count','mean']);v=bm.to_numpy()
    draw=np.random.default_rng(20261023).integers(len(v),size=(10000,len(v)))
    boot=v[draw,0].sum(1)/v[draw,1].sum(1);rest=pair[~pair.benchmark.str.startswith('alignment_faking/')]
    return clean({'delta_ALC':pair.delta.mean(),'benchmark_equal_delta':bm['mean'].mean(),
        'cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),
        'without_alignment_delta':rest.delta.mean(),'without_alignment_bm_delta':rest.groupby('benchmark').delta.mean().mean(),
        'by_benchmark':bm['mean'].to_dict(),'trajectories':len(pair),'benchmarks':len(bm)})

def replay(root):
    rows=[];maxerr=0.;queries=0
    for folder in sorted((root/'source').glob('fold*/*')):
        p=dict(np.load(folder/'predictions.npz'));saved=read(folder/'metrics.json');lookup={(r['subject_id'],r['seed'],r['budget']):r for r in saved}
        for idx in read(folder/'index.json'):
            a=idx['start'];b=a+idx['size'];orig=idx['original'];queries+=b-a
            for j,n in enumerate([15,31]):
                r=lookup[(orig['subject_id'],orig['seed'],n)]
                for m in METHODS:
                    br=brier(p[m][a:b,j],p['counts'][a:b],p['successes'][a:b]);ac=accuracy(p[m][a:b,j],p['counts'][a:b],p['successes'][a:b])
                    maxerr=max(maxerr,abs(br-r['brier_'+m]),abs(ac-r['accuracy_'+m]))
                assert np.allclose(p['gate'][a:b,j],p['own'][a:b,j]+r['gate']*(p['partial'][a:b,j]-p['own'][a:b,j]),rtol=0,atol=0)
        rows+=saved
    assert maxerr<1e-12
    return pd.DataFrame(rows),{'max_brier_accuracy_error':maxerr,'query_variant_rows':queries,'budget_records':len(rows),'finished':now()}

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();r78=ROOT/'R78';root=ROOT/'R79'
    assert read(r78/'decision.json')['R79_diagnostic_entry'];root.mkdir(exist_ok=True)
    source_files=list(r78.glob('source_fold*/*/query_predictions.npz'))+list(r78.glob('source_fold*/*/query_index.json'))
    freeze(root/'source_contract',[Path(__file__),REPO/'research_loop/analyze_support_sharing_reliability.py',
        REPO/'research_loop/shared_reliability_common.py',REPO/'research_loop/SHARED_RELIABILITY_SPEC.md',
        REPO/'research_loop/SHARED_RELIABILITY_OOF_ADDENDUM.md',r78/'diagnostic_metrics.csv',r78/'decision.json',*source_files],
        {'source_half_rule':'stable_seed(7901|full_Benchmark_name) % 2','budgets':[15,31],
         'gate':'1[z<0] if fit-half eligible metric else unchanged partial','metric_priority':METRICS,
         'matched_mean':'fit-half equal trajectory mean gate by budget',
         'matched_cost':'sqrt(fit-half E[g*d^2]/E[d^2]) by budget; diagnostic only',
         'full_gate':'at least 2/3 pools improve vs partial AND matched_mean; dedup pair/BM/rest all negative vs both',
         'source_predictions_reused':True,'source_pools_overlap':True,'outer_scores_not_accessed':True})
    frame=pd.read_csv(r78/'diagnostic_metrics.csv');frame=frame[frame.scenario=='main'].copy()
    frame['half']=frame.benchmark.map(lambda b:stable_seed('7901|'+b)%2)
    # Geometry uses no outcomes and preserves equal trajectory / response weights.
    geometry={}
    for f in range(3):
        for folder in sorted((r78/f'source_fold{f}').iterdir()):
            if not (folder/'complete.json').exists():continue
            p=dict(np.load(folder/'query_predictions.npz'))
            for idx in read(folder/'query_index.json'):
                if idx['scenario']!='main':continue
                orig=idx['original'];a=idx['start'];b=a+idx['size'];w=p['counts'][a:b];w=w/w.sum()
                for j,n in enumerate([15,31]):geometry[(f,orig['benchmark'],orig['subject_id'],orig['seed'],n)]=float(w@((p['partial'][a:b,j]-p['own'][a:b,j])**2))
    frame['cost']=[geometry[(r.selection_fold,r.benchmark,r.subject_id,r.seed,r.budget)] for r in frame.itertuples()]
    policies={};refit={}
    for f in range(3):
        pool=frame[frame.selection_fold==f];policies[str(f)]={}
        for validation_half in [0,1]:
            fit=pool[pool.half!=validation_half];valid=pool[pool.half==validation_half]
            assert len(fit) and len(valid) and set(fit.benchmark).isdisjoint(valid.benchmark)
            metric,tests=choose(fit);coeff={}
            for n in [15,31]:
                part=fit[fit.budget==n];g=(part['z_'+metric]<0).astype(float) if metric else pd.Series(1.,index=part.index)
                cost=float(part.cost.sum());coeff[str(n)]={'mean':float(g.mean()),'cost':float(np.sqrt((g*part.cost).sum()/cost)) if cost>1e-30 else 1.}
            policies[str(f)][str(validation_half)]={'metric':metric,'checks':tests,'coefficients':coeff,
                'fit_benchmarks':sorted(fit.benchmark.unique()),'validation_benchmarks':sorted(valid.benchmark.unique())}
        metric,tests=choose(pool);refit[str(f)]={'metric':metric,'checks':tests,'scope':'legal source only, not authorization for outer'}
    write_json(root/'nested_source_policies.json',clean(policies));write_json(root/'source_refit_metrics.json',clean(refit))
    lookup={(int(r.selection_fold),r.benchmark,r.subject_id,int(r.seed),int(r.budget)):r for r in frame.itertuples()}
    for f in range(3):
        for folder in sorted((r78/f'source_fold{f}').iterdir()):
            if not (folder/'complete.json').exists():continue
            p=dict(np.load(folder/'query_predictions.npz'));bank={};index=[];rows=[];off=0
            for idx in read(folder/'query_index.json'):
                if idx['scenario']!='main':continue
                orig=idx['original'];a=idx['start'];b=a+idx['size'];bm=orig['benchmark'];half=stable_seed('7901|'+bm)%2
                policy=policies[str(f)][str(half)];metric=policy['metric'];q={m:p[m][a:b].copy() for m in ['own','pooled','partial']}
                q.update({m:np.empty_like(q['own']) for m in ['gate','matched_mean','matched_cost']})
                for j,n in enumerate([15,31]):
                    r=lookup[(f,bm,orig['subject_id'],orig['seed'],n)];g=float(getattr(r,'z_'+metric)<0) if metric else 1.
                    coeff=policy['coefficients'][str(n)];d=q['partial'][:,j]-q['own'][:,j]
                    for m,alpha in [('gate',g),('matched_mean',coeff['mean']),('matched_cost',coeff['cost'])]:q[m][:,j]=q['own'][:,j]+alpha*d
                    row={'selection_fold':f,'benchmark':bm,'subject_id':orig['subject_id'],'seed':orig['seed'],'budget':n,'validation_half':half,
                        'metric':metric,'gate':g,'matched_mean':coeff['mean'],'matched_cost':coeff['cost'],'own_labels':r.own_labels,'peer_labels':r.peer_labels,'active_peers':r.active_peers}
                    for m in METHODS:row['brier_'+m]=brier(q[m][:,j],p['counts'][a:b],p['successes'][a:b]);row['accuracy_'+m]=accuracy(q[m][:,j],p['counts'][a:b],p['successes'][a:b])
                    rows.append(row)
                for m in METHODS:bank.setdefault(m,[]).append(q[m])
                for m in ['counts','successes']:bank.setdefault(m,[]).append(p[m][a:b])
                index.append({'original':orig,'start':off,'size':b-a,'source_directory':str(folder)});off+=b-a
            dest=root/'source'/f'fold{f}'/folder.name;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'predictions.npz',**{m:np.concatenate(v) for m,v in bank.items()})
            write_json(dest/'index.json',index);write_json(dest/'metrics.json',rows)
    d,check=replay(root);d.to_csv(root/'source_metrics.csv',index=False);write_json(root/'source_replay_validation.json',check)
    first=d.groupby('benchmark').selection_fold.min().to_dict();unique=d[d.apply(lambda r:r.selection_fold==first[r.benchmark],axis=1)]
    pool={str(f):{m:compare(d[d.selection_fold==f],m) for m in ['partial','matched_mean','matched_cost']} for f in range(3)}
    combined={m:compare(unique,m) for m in ['partial','matched_mean','matched_cost']}
    passed=sum(all(pool[str(f)][m]['delta_ALC']<0 for m in ['partial','matched_mean']) for f in range(3))
    criteria={'two_source_pools':passed>=2,**{m+'_'+k:combined[m][k]<0 for m in ['partial','matched_mean'] for k in ['delta_ALC','benchmark_equal_delta','without_alignment_delta','without_alignment_bm_delta']}}
    authorize=all(criteria.values())
    decision={'status':'source_policy_entry_passed' if authorize else 'source_policy_no_independent_gain','R78_signal_present':True,
        'R79_source_executed':True,'R79_outer_authorized':authorize,'R79_outer_executed':False,'criteria':criteria,'source_pools_improved_vs_both':passed,
        'scope':'nested Benchmark source validation; not a new full outer result','no_threshold_scan':True,'finished':now()}
    write_json(root/'source_comparisons.json',clean({'source_pools':pool,'deduplicated':combined}));write_json(root/'decision.json',decision)
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'threads':4,'finished':now(),'stage':'source_only'})
    check_freeze(root/'source_contract');print(json.dumps(clean({'decision':decision,'comparisons':combined,'pool_deltas':{f:{m:v['delta_ALC'] for m,v in z.items()} for f,z in pool.items()},'replay':check})),flush=True)

if __name__=='__main__':main()
