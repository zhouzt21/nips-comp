"""Read saved R79 artifacts against source inputs and independently check isolation."""
from shared_reliability_common import *

def main():
    root=ROOT/'R79';policies=read(root/'nested_source_policies.json')
    df=pd.read_csv(ROOT/'R78/diagnostic_metrics.csv');df=df[df.scenario=='main']
    lookup={(int(r.selection_fold),r.benchmark,r.subject_id,int(r.seed),int(r.budget)):r for r in df.itertuples()}
    trajectories=0;maxerr=0.;coefferr=0.;gates=0
    for f in range(3):
        allowed=source(f);pool=df[df.selection_fold==f]
        assert set(pool.benchmark)<=set(allowed['train_benchmarks'])
        assert set(pool.benchmark).isdisjoint(allowed['heldout_benchmarks'])
        for h,p in policies[str(f)].items():
            fit=set(p['fit_benchmarks']);val=set(p['validation_benchmarks'])
            assert fit.isdisjoint(val) and fit|val==set(pool.benchmark)
            assert all(stable_seed('7901|'+b)%2==int(h) for b in val)
            passing=[m for m in ['partial','pooled','mean','combined'] if all(p['checks'][m].values())]
            assert p['metric']==(passing[0] if passing else None)
            for n in [15,31]:
                train=pool[pool.benchmark.isin(fit)&(pool.budget==n)]
                alpha=float((train['z_'+p['metric']]<0).mean()) if p['metric'] else 1.
                coefferr=max(coefferr,abs(alpha-p['coefficients'][str(n)]['mean']))
        for folder in sorted((root/'source'/f'fold{f}').iterdir()):
            p=dict(np.load(folder/'predictions.npz'));src=ROOT/'R78'/f'source_fold{f}'/folder.name;q=dict(np.load(src/'query_predictions.npz'))
            original={(v['original']['subject_id'],v['original']['seed']):v for v in read(src/'query_index.json') if v['scenario']=='main'}
            for idx in read(folder/'index.json'):
                o=idx['original'];v=original[(o['subject_id'],o['seed'])];a=idx['start'];b=a+idx['size'];c=v['start'];end=c+v['size'];trajectories+=1
                assert o==v['original']
                for k in ['counts','successes','own','pooled','partial']:assert np.array_equal(p[k][a:b],q[k][c:end])
                policy=policies[str(f)][str(stable_seed('7901|'+o['benchmark'])%2)]
                for j,n in enumerate([15,31]):
                    r=lookup[(f,o['benchmark'],o['subject_id'],o['seed'],n)];m=policy['metric'];g=float(getattr(r,'z_'+m)<0) if m else 1.
                    expected=q['partial'][c:end,j] if g else q['own'][c:end,j]
                    maxerr=max(maxerr,float(abs(expected-p['gate'][a:b,j]).max()));gates+=1
                    for name,key in [('matched_mean','mean'),('matched_cost','cost')]:
                        alpha=policy['coefficients'][str(n)][key]
                        expected=q['own'][c:end,j]+alpha*(q['partial'][c:end,j]-q['own'][c:end,j])
                        maxerr=max(maxerr,float(abs(expected-p[name][a:b,j]).max()))
    assert maxerr<1e-12 and coefferr<1e-12 and trajectories==1146 and gates==2292
    decision=read(root/'decision.json');assert not decision['R79_outer_authorized'] and not decision['R79_outer_executed']
    assert not (root/'outer').exists() and not (root/'full').exists()
    check_freeze(root/'source_contract')
    result={'status':'passed','source_trajectories':trajectories,'budget_decisions':gates,'max_probability_error':maxerr,
        'max_matched_mean_fit_error':coefferr,'outer_Benchmark_isolation':True,'inner_Benchmark_isolation':True,
        'R79_full_not_executed':True,'verification_role':'main agent deterministic independent artifact replay','finished':now()}
    write_json(root/'source_isolation_validation.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
