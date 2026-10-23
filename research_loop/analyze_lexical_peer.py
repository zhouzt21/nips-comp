"""Numerical replay, paired comparisons, and mechanism tables from frozen runs."""
import argparse,time,hashlib
import pandas as pd
from historical_positive_common import *
from analyze_historical_positive import paired_comparison

def main(stage):
    start=time.perf_counter();root=ROOT/stage/'full';summary=read(root/'summary.json');allrows=[];maxerror=0.;maxacc=0.;chunks=0;qcount=0;energy_error=0.;group_records=[];peer_slices=[]
    for folder in sorted((root/'outer').glob('fold*/*')):
        rows=read(folder/'paired_scores.json');allrows+=rows;lookup={r['identity']:r for r in rows}
        if stage=='R72':
            ctx={(v['subject_id'],v['seed']):v for v in read(ROOT/'R72/contexts'/folder.name/'index.json')}
        for ip in sorted(folder.glob('index_*.json')):
            suf=ip.stem.split('_')[-1];arr=dict(np.load(folder/f'predictions_{suf}.npz'));index=read(ip);offset=0
            methods=[k for k in arr if k not in ['counts','successes','neighbor_agreement']]
            for idx in index:
                a=idx['start'];b=a+idx['size'];assert a==offset;offset=b
                meta={k:v for k,v in idx.items() if k not in ['identity','start','size']}
                assert hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()==idx['identity']
                row=lookup[idx['identity']];c=arr['counts'][a:b,None];s=arr['successes'][a:b,None]
                assert np.array_equal(c[:,0],idx['counts']);assert np.array_equal(s[:,0],idx['successes'])
                for m in methods:
                    p=arr[m][a:b];assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                    br=(c*p*p-2*s*p+s).sum(0)/c.sum();ac=np.where(p>=.5,s,c-s).sum(0)/c.sum()
                    maxerror=max(maxerror,float(np.max(np.abs(br-row['brier'][m]))));maxacc=max(maxacc,float(np.max(np.abs(ac-row['accuracy'][m]))))
                assert maxerror<1e-12 and maxacc<1e-12
                if stage=='R71':
                    p=arr['R25'][a:b];d=arr['R64'][a:b]-p;z=arr['neighbor_agreement'][a:b]
                    target=(c*(arr['R71_grouped'][a:b]-p)**2).sum(0)
                    for k in range(4):energy_error=max(energy_error,float(np.max(np.abs((c*(arr[f'random{k}_R71'][a:b]-p)**2).sum(0)-target))))
                    for j,n in enumerate(BUDGETS):
                        for g in [0,1]:
                            mask=z[:,j]==g;aa=((c[:,0]*p[:,j]-s[:,0])*d[:,j])[mask].sum()/c.sum();cc=(c[:,0]*d[:,j]**2)[mask].sum()/c.sum()
                            group_records.append({'fold':row['fold'],'benchmark':row['benchmark'],'subject_id':row['subject_id'],'seed':row['seed'],'budget':n,'group':g,
                                'A':float(aa),'C':float(cc),'response_share':float(c[mask].sum()/c.sum())})
                if stage=='R72':
                    from collections import Counter
                    target=ctx[(idx['subject_id'],idx['seed'])];peers=[ctx[(sid,idx['seed'])] for sid in target['peer_ids']]
                    p=arr['R25'][a:b]
                    for j,n in enumerate(BUDGETS):
                        visible=set()
                        for peer in peers:
                            for key,count in Counter(peer['support_keys'][:n]).items():
                                if count==1:visible.add(key)
                        hit=np.array([v in visible for v in target['query_keys']])
                        for m in ['R72_same3','R72_pooled3','R72_diff3','R72_morelabels_mean3']:
                            d=arr[m][a:b,j]-p[:,j]
                            for matched,mask in [(True,hit),(False,~hit)]:
                                aa=((c[:,0]*p[:,j]-s[:,0])*d)[mask].sum()/c.sum();cc=(c[:,0]*d*d)[mask].sum()/c.sum()
                                peer_slices.append({'benchmark':row['benchmark'],'fold':row['fold'],'budget':n,'method':m,'matched':matched,
                                    'response_share':float(c[mask].sum()/c.sum()),'alignment':float(2*aa),'cost':float(cc),'delta_contribution':float(2*aa+cc)})
                qcount+=len(c)
            assert offset==len(arr['counts']);chunks+=1
        print(json.dumps({'stage':'replay_'+stage,'benchmark':folder.name,'trajectories':len(rows),'max_brier_error':maxerror}),flush=True)
    assert len(allrows)==15726;assert len({(r['benchmark'],r['subject_id']) for r in allrows})==5242
    assert len({r['benchmark'] for r in allrows})==36
    comparisons={}
    methods=[m for m in summary['results'] if not m.startswith(('perm','random'))]
    baselines=['R25','R56_segment','R64','R70_independent','R70_monotone','R72_pooled3','R36_own_pooled3']
    for m in methods:
        comparisons[m]={b:paired_comparison(allrows,m,b) for b in baselines if b in summary['results'] and b!=m}
    control={}
    if stage=='R71':
        for m in ['R64','R70_independent','R70_monotone','R71_grouped']:
            true=paired_comparison(allrows,m,'R25');perms=[paired_comparison(allrows,f'perm{k}_{m}',f'perm{k}_R25') for k in range(4)]
            control[m]={'true_increment':true,'permuted_increment':perms,'permutation_minus_true_mean':float(np.mean([v['delta'] for v in perms])-true['delta'])}
        control['random_energy']={f'random{k}':paired_comparison(allrows,'R71_grouped',f'random{k}_R71') for k in range(4)}
        groups=pd.DataFrame(group_records);groups.to_csv(root/'group_alignment.csv',index=False)
        grouped=groups.groupby(['budget','group'])[['A','C','response_share']].mean().reset_index();grouped['posthoc_alpha']=np.clip(-grouped.A/np.maximum(grouped.C,1e-30),0,1)
        grouped.to_csv(root/'group_alignment_summary.csv',index=False)
        groups.groupby(['benchmark','budget','group'])[['A','C','response_share']].mean().reset_index().to_csv(root/'group_alignment_by_benchmark.csv',index=False)
    if stage=='R72':
        for m in ['R72_same3','R72_pooled3','R72_diff3','R36_own_pooled3']:
            control[m]={f'perm{k}':paired_comparison(allrows,m,f'perm{k}_{m}') for k in range(2)}
        sdf=pd.DataFrame(peer_slices)
        sdf.groupby(['budget','method','matched'])[['response_share','alignment','cost','delta_contribution']].mean().reset_index().to_csv(root/'matched_slice_summary.csv',index=False)
        sdf.groupby(['benchmark','budget','method','matched'])[['response_share','alignment','cost','delta_contribution']].mean().reset_index().to_csv(root/'matched_slice_by_benchmark.csv',index=False)
    write_json(root/'additional_comparisons.json',comparisons);write_json(root/'control_analysis.json',control)
    decisions={}
    for m in methods:
        if m in ['R25','R56_segment','R64','smoothed_mean_2','training_selected_smoothed','R72_nopeer']:continue
        gate={}
        for baseline in ['smoothed_mean_2','training_selected_smoothed']:
            c=summary['comparisons'][m][baseline]
            gate[baseline]={'gain_ge_005':c['delta_alc']<=-.005,'three_fold_gain':all(v<0 for v in c['fold_deltas']),
                'benchmark_equal_gain':c['benchmark_equal_delta']<0,'CI_upper_below_zero':c['cluster_ci95'][1]<0,
                'no_large_unexplained_regression':not c['regression_ge_03']}
        decisions[m]={'numerical_gate':all(all(v.values()) for v in gate.values()),'criteria':gate,
            'hosted_deployment':'unconfirmed_runtime' if stage=='R72' else 'stateless_inputs_available','official_submission':False}
    write_json(root/'gate.json',decisions)
    write_json(root/'replay_validation.json',{'trajectories':len(allrows),'pairs':5242,'benchmarks':36,'query_variant_rows':qcount,'chunks':chunks,
        'max_brier_error':maxerror,'max_accuracy_error':maxacc,'random_energy_error':energy_error,'seconds':time.perf_counter()-start,'finished':now()})
    print(json.dumps({'stage':stage,'max_brier_error':maxerror,'max_accuracy_error':maxacc,'seconds':time.perf_counter()-start,'controls':control}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['R70','R71','R72']);main(parser.parse_args().stage)
