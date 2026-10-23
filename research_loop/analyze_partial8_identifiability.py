"""R80 saved-probability replay and predeclared sign identifiability decision."""
import time
from collections import defaultdict
from scipy.stats import spearmanr
from partial8_next_common import *

KEY=['selection_fold','benchmark','subject_id','seed','budget']

def replay(folder):
    rows=read(folder/'metrics.json');look={(r['subject_id'],r['seed'],r['budget'],r['estimator']):r for r in rows}
    s=dict(np.load(folder/'support_predictions.npz'));q=dict(np.load(folder/'masked_query_predictions.npz'))
    sums=defaultdict(lambda:[0.,0.,0.]);coverage=defaultdict(list);err=0.;masks_n=0
    for idx in read(folder/'mask_index.json'):
        k=(idx['subject_id'],idx['seed'],idx['budget'],idx['estimator']);rep=idx['rep'];n=idx['budget'];reps=1 if k[-1]=='loio' else 2
        fit=idx['fit_indices'];hold=idx['hold_indices'];ids=idx['support_ids']
        assert set(ids[i] for i in fit).isdisjoint(ids[i] for i in hold)
        assert sorted(fit+hold)==list(range(n)) and all(v[1]<=n for v in idx['peer_prefixes'])
        a=idx['support_start'];b=a+idx['support_size'];c=idx['query_start'];end=c+idx['query_size'];y=s['truth'][a:b]
        delta=(s['partial'][a:b]-y)**2-(s['own'][a:b]-y)**2;weight=len(hold)/(n*reps)
        sums[k][0]+=float(delta.sum())/(n*reps);sums[k][1]+=weight*(brier(q['partial'][c:end],q['counts'][c:end],q['successes'][c:end])-brier(q['own'][c:end],q['counts'][c:end],q['successes'][c:end]));sums[k][2]+=weight
        assert np.array_equal(q['counts'][c:end],idx['original']['counts']) and np.array_equal(q['successes'][c:end],idx['original']['successes'])
        coverage[(k,rep)]+=hold;masks_n+=1
    for (k,rep),hold in coverage.items():assert sorted(hold)==list(range(k[2]))
    for k,(z,qd,w) in sums.items():
        r=look[k];err=max(err,abs(z-r['z']),abs(qd-r['masked_fit_query_delta']),abs(w-1),abs(r['estimation_error']-r['validation_gap']-r['fit_size_gap']))
    boot=read(folder/'bootstrap_metrics.json');bi=read(folder/'bootstrap_index.json');bp=dict(np.load(folder/'bootstrap_predictions.npz'))
    for r,i in zip(boot,bi):
        assert set(i['support_ids'][k] for k in i['fit_indices']).isdisjoint(i['support_ids'][k] for k in i['hold_indices'])
        a=i['start'];b=a+i['size'];y=bp['truth'][a:b];z=np.mean((bp['partial'][a:b]-y)**2-(bp['own'][a:b]-y)**2);err=max(err,abs(z-r['z']))
    assert err<1e-11
    return rows,boot,{'folder':str(folder),'max_error':err,'masks':masks_n,'support_rows':len(s['truth']),'masked_query_rows':len(q['counts']),'bootstrap_records':len(boot)}

def ba(d,z='z'):
    a=d[abs(d.query_delta)>1e-12];truth=a.query_delta>0;pred=a[z]>=0
    if truth.nunique()<2:return np.nan
    return float(np.mean([(pred[truth==x]==x).mean() for x in [False,True]]))

def paired_ba(d):
    a=d[abs(d.query_delta)>1e-12];values=[]
    for bm,g in a.groupby('benchmark'):
        truth=g.query_delta>0;vals=[]
        for label in [False,True]:
            ix=truth==label;vals += [int(ix.sum()),int(((g.z>=0)[ix]==label).sum()),int(((g.half_z>=0)[ix]==label).sum())]
        values.append(vals)
    v=np.asarray(values,float);draw=np.random.default_rng(20261023).integers(len(v),size=(10000,len(v)));x=v[draw].sum(1)
    good=(x[:,0]>0)&(x[:,3]>0);x=x[good]
    delta=.5*((x[:,1]-x[:,2])/x[:,0]+(x[:,4]-x[:,5])/x[:,3])
    return np.quantile(delta,[.025,.975]).tolist()

def group_ci(d):
    if len(d)==0:return [None,None]
    g=d.groupby('benchmark').query_delta.agg(['sum','count']);v=g.to_numpy();draw=np.random.default_rng(20261023).integers(len(v),size=(10000,len(v)))
    b=v[draw,0].sum(1)/v[draw,1].sum(1);return np.quantile(b,[.025,.975]).tolist()

def metrics(d):
    rho=[];feature=[]
    for bm,g in d.groupby('benchmark'):
        x=g.groupby('subject_id').mean(numeric_only=True)
        if len(x)>=3 and x.z.nunique()>1 and x.query_delta.nunique()>1:rho.append(float(spearmanr(x.z,x.query_delta).statistic))
        if len(x)>=3 and x.feature_mean_gap.nunique()>1 and x.estimation_error.abs().nunique()>1:feature.append(float(spearmanr(x.feature_mean_gap,x.estimation_error.abs()).statistic))
    harmful=d[d.lo>0];benefit=d[d.hi<0];unknown=d[(d.lo<=0)&(d.hi>=0)]
    boot=paired_ba(d)
    return plain({'requests':len(d),'pairs':d[['benchmark','subject_id']].drop_duplicates().shape[0],'benchmarks':d.benchmark.nunique(),
        'request_BA':ba(d),'half_BA':ba(d,'half_z'),'BA_improvement':ba(d)-ba(d,'half_z'),'BA_improvement_ci95':boot,
        'request_sign_accuracy':float(((d.z<0)==(d.query_delta<0))[abs(d.query_delta)>1e-12].mean()),
        'benchmark_mean_pair_spearman':np.mean(rho) if rho else None,'estimable_rho_BM':len(rho),
        'bias':d.estimation_error.mean(),'mae':d.estimation_error.abs().mean(),'rmse':np.sqrt(np.mean(d.estimation_error**2)),
        'mean_fit_size_gap':d.fit_size_gap.mean(),'mean_validation_gap':d.validation_gap.mean(),
        'mean_abs_fit_size_gap':d.fit_size_gap.abs().mean(),'mean_abs_validation_gap':d.validation_gap.abs().mean(),
        'feature_gap_error_BM_spearman':np.mean(feature) if feature else None,
        'mean_interval_width':(d.hi-d.lo).mean(),'harmful_fraction':len(harmful)/len(d),'beneficial_fraction':len(benefit)/len(d),'unknown_fraction':len(unknown)/len(d),
        'harmful_pairs':harmful[['benchmark','subject_id']].drop_duplicates().shape[0],'harmful_BM':harmful.benchmark.nunique(),
        'harmful_query_mean':harmful.query_delta.mean(),'harmful_query_ci95':group_ci(harmful),
        'nominal_interval_contains_query':((d.lo<=d.query_delta)&(d.hi>=d.query_delta)).mean(),
        'repeat_sign_agreement':d.repeat_agreement.mean(),'repeat_abs_gap':d.repeat_gap.mean()})

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R80';assert (root/'execution.json').exists();allrows=[];boot=[];checks=[]
    for f in range(3):
        for folder in sorted((root/f'source_fold{f}').iterdir()):
            rows,b,c=replay(folder);allrows+=rows;boot+=b;checks.append(c)
    flat=[]
    for r in allrows:
        z=r['repeat_z'];flat.append({**{k:v for k,v in r.items() if k not in ['interval','query_brier','query_accuracy','repeat_z']},
            'lo':r['interval']['lo'],'hi':r['interval']['hi'],'bootstrap_positive_fraction':r['interval']['bootstrap_positive_fraction'],
            'repeat_agreement':float((z[0]<0)==(z[1]<0)) if len(z)==2 else np.nan,'repeat_gap':abs(z[0]-z[1]) if len(z)==2 else np.nan,
            **{'query_brier_'+k:v for k,v in r['query_brier'].items()},**{'query_accuracy_'+k:v for k,v in r['query_accuracy'].items()}})
    d=pd.DataFrame(flat);half=d[d.estimator=='half'][KEY+['z']].rename(columns={'z':'half_z'});d=d.merge(half,on=KEY,validate='many_to_one')
    d.to_csv(root/'metrics.csv',index=False);first=d.groupby('benchmark').selection_fold.min().to_dict();u=d[d.apply(lambda r:r.selection_fold==first[r.benchmark],axis=1)];u.to_csv(root/'deduplicated_metrics.csv',index=False)
    stats={k:{str(n):metrics(g) for n,g in part.groupby('budget')} for k,part in u.groupby('estimator')}
    pools={str(f):{k:{str(n):metrics(g) for n,g in part.groupby('budget')} for k,part in pool.groupby('estimator')} for f,pool in d.groupby('selection_fold')}
    decisions=[];selected=None
    for k in ['loio','k5']:
        ss=stats[k];major=[n for n in ['15','31'] if ss[n]['BA_improvement']>=.03 and ss[n]['BA_improvement_ci95'][0]>0]
        tests={'BA_both_ge_060':all(ss[n]['request_BA']>=.60 for n in ['15','31']),
            'both_non_degrading':all(ss[n]['BA_improvement']>=0 for n in ['15','31']),'material_paired_improvement':bool(major)}
        good=[]
        for n in major:
            v=ss[n];sourcegood=sum(pools[str(f)][k][n]['BA_improvement']>0 for f in range(3))>=2
            harm=v['harmful_pairs']>=10 and v['harmful_BM']>=5 and v['harmful_query_ci95'][0] is not None and v['harmful_query_ci95'][0]>0
            if sourcegood and harm:good.append(n)
        tests['source_direction_and_harmful_group_confirmed']=bool(good)
        passed=all(tests.values());decisions.append({'estimator':k,'tests':tests,'passed':passed,'eligible_budgets':good})
        if passed and selected is None:selected=k
    bins=[-np.inf,-.02,-.005,0,.005,.02,np.inf];cal=u.copy();cal['risk_bin']=pd.cut(cal.z,bins,right=False)
    cal.groupby(['estimator','budget','risk_bin'],observed=True).agg(requests=('z','size'),z=('z','mean'),query_delta=('query_delta','mean')).to_csv(root/'calibration_bins.csv')
    u.groupby(['estimator','budget','benchmark']).mean(numeric_only=True).to_csv(root/'benchmark_metrics.csv')
    b=pd.DataFrame(boot);b.to_csv(root/'refit_bootstrap_metrics.csv',index=False)
    b.groupby(['selection_fold','benchmark','budget']).agg(z_mean=('z','mean'),z_std=('z','std'),min_z=('z','min'),max_z=('z','max'),positive_fraction=('z',lambda x:(x>0).mean()),
        sign_error_fraction=('z',lambda x:float(np.mean((x>=0)!=(b.loc[x.index,'query_delta']>=0)))),query_delta=('query_delta','first')).to_csv(root/'refit_bootstrap_summary.csv')
    decision={'status':'source_sign_entry_passed' if selected else 'sharing_sign_not_identifiable','selected_estimator':selected,'tests':decisions,'R82_authorized':selected is not None,'R83_authorized':False,
        'scope':'predeclared source evidence standard; failure does not prove zero information','finished':now()}
    write_json(root/'summary.json',{'deduplicated':stats,'source_pools':pools});write_json(root/'decision.json',decision)
    write_json(root/'replay_validation.json',{'max_error':max(c['max_error'] for c in checks),'source_trajectories':len(d)//6,'dedup_trajectories':len(u)//6,'checks':checks,'finished':now()})
    write_json(root/'analysis_execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'finished':now()});check_freeze(root/'contract')
    print(json.dumps(plain({'decision':decision,'summary':stats})),flush=True)
if __name__=='__main__':main()
