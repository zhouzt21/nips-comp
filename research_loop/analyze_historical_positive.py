"""Re-score saved probabilities, exact low-budget decomposition, and decision tables."""
import argparse
import hashlib
import time
from historical_positive_common import *

def paired_comparison(rows,method,baseline):
    names=sorted({r['benchmark'] for r in rows})
    delta=np.array([float((np.array(r['brier'][method])-r['brier'][baseline])@ALC_WEIGHTS) for r in rows])
    counts=np.array([sum(r['benchmark']==b for r in rows) for b in names])
    sums=np.array([sum(d for r,d in zip(rows,delta) if r['benchmark']==b) for b in names])
    draws=np.random.default_rng(20261022).integers(len(names),size=(10000,len(names)))
    boot=sums[draws].sum(1)/counts[draws].sum(1)
    return {'delta':float(delta.mean()),'cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),
        'benchmark_equal_delta':float(np.mean(sums/counts)),
        'fold_delta':[float(np.mean([d for r,d in zip(rows,delta) if r['fold']==f])) for f in range(3)]}

def decomposition(p,c,s):
    mu=float(c@p/c.sum()); target=float(s.sum()/c.sum())
    var=float(c@((p-mu)**2)/c.sum());cov=float(((p-mu)@(s-c*target))/c.sum())
    mean=(mu-target)**2; variation=var-2*cov; constant=target*(1-target)
    assert abs(brier(p,c,s)-(constant+mean+variation))<1e-12
    return {'mean_bias_sq':mean,'prediction_variance':var,'covariance':cov,'variation_term':variation,
            'response_variance':constant,'mean_prediction':mu,'mean_outcome':target}

def main(stage):
    import pandas as pd
    start=time.perf_counter(); root=ROOT/stage if stage!='R64' else ROOT/'R64/full'; summary=read(root/'summary.json')
    allrows=[]; details=[]; maxerror=0.; chunks=0;predictions=0; gate_rows=[]
    paths=sorted((root/'outer').glob('fold*/*/paired_scores.json'))
    for path in paths:
        rows=read(path);allrows+=rows;lookup={(r['subject_id'],r['seed']):r for r in rows}
        for file in sorted(path.parent.glob('predictions_*.npz')):
            ix=read(file.with_name(file.name.replace('predictions_','index_')).with_suffix('.json'))
            with np.load(file,allow_pickle=False) as archive:
                bank={k:archive[k] for k in archive.files}
                methods=[m for m in bank if m not in ['counts','successes']]
                assert set(methods)==set(summary['results'])
                offset=0
                for index in ix:
                    meta={k:v for k,v in index.items() if k not in ['identity','start','size']}
                    ident=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
                    assert ident==index['identity']
                    assert index['start']==offset
                    a,b=offset,offset+index['size'];offset=b
                    row=lookup[(index['subject_id'],index['seed'])]
                    assert row['identity']==ident
                    c,s=bank['counts'][a:b],bank['successes'][a:b]
                    assert np.array_equal(c,index['counts']) and np.array_equal(s,index['successes'])
                    assert np.all(c>0) and np.all((s>=0)&(s<=c))
                    for method in methods:
                        p=bank[method][a:b]
                        assert p.shape==(len(c),6) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                        for j in range(6):
                            err=abs(brier(p[:,j],c,s)-row['brier'][method][j]);maxerror=max(maxerror,err)
                            assert err<1e-12
                            assert abs(accuracy(p[:,j],c,s)-row['accuracy'][method][j])<1e-12
                    if stage=='R62':
                        for method in ['R54_full','R56_full']:
                            for j in range(2):
                                value=decomposition(bank[method][a:b,j],c,s)
                                base=decomposition(bank['R25'][a:b,j],c,s)
                                record={k:row[k] for k in ['fold','benchmark','subject_id','seed']}
                                record.update(method=method,budget=BUDGETS[j],**value,
                                    delta_mean=value['mean_bias_sq']-base['mean_bias_sq'],
                                    delta_variation=value['variation_term']-base['variation_term'],
                                    delta_brier=row['brier'][method][j]-row['brier']['R25'][j])
                                assert abs(record['delta_mean']+record['delta_variation']-record['delta_brier'])<1e-12
                                details.append(record)
                        gate=row['gate_use_R56'];p=bank['R61_support1_gate'][a:b]
                        assert np.array_equal(p[:,[0,2,3,4,5]],bank['R25'][a:b][:,[0,2,3,4,5]])
                        assert np.array_equal(p[:,1],bank['R56_full' if gate else 'R25'][a:b,1])
                        assert gate==(row['R56_first_support_sqerr']<.25)
                        gate_rows.append({k:row[k] for k in ['fold','benchmark','subject_id','seed']} | {
                            'chosen':gate,'sqerr':row['R56_first_support_sqerr'],
                            'ungated_delta_B1':row['brier']['R56_full'][1]-row['brier']['R25'][1],
                            'gated_delta_B1':row['brier']['R61_support1_gate'][1]-row['brier']['R25'][1]})
                    else:
                        for method in methods:
                            if method in EPOCHS:
                                assert np.array_equal(bank[method][a:b,:2],bank['R25'][a:b,:2])
                                if EPOCHS[method][row['fold']]==0:
                                    assert np.array_equal(bank[method][a:b],bank['R25'][a:b])
                            if stage=='R64' and method not in BASELINES:
                                assert np.array_equal(bank[method][a:b,:2],bank['R25'][a:b,:2])
                    predictions+=len(c)
                assert offset==len(bank['counts'])
            chunks+=1
    assert len(allrows)==5242*3
    assert len({(r['benchmark'],r['subject_id']) for r in allrows})==5242
    assert len({r['benchmark'] for r in allrows})==36
    actual=aggregate(allrows)
    for m in summary['results']:
        assert np.max(np.abs(np.array(actual['results'][m]['brier_mean'])-summary['results'][m]['brier_mean']))<1e-12
    decisions={}
    for m,comp in summary['comparisons'].items():
        if m in BASELINES or m.startswith('permuted_template_'):continue
        tests={}
        for b in ['smoothed_mean_2','training_selected_smoothed']:
            c=comp[b]
            tests[b]={'gain_ge_005':c['delta_alc']<=-.005,'all_folds_improve':all(d<0 for d in c['fold_deltas']),
                'BM_improve':c['benchmark_equal_delta']<0,'CI_upper_below_zero':c['cluster_ci95'][1]<0,
                'no_ge_03_regression':not c['regression_ge_03']}
        simple_pass=all(all(v.values()) for v in tests.values())
        r25=comp['R25']
        decisions[m]={'simple_baseline_tests':tests,'simple_numeric_gate_pass':simple_pass,
            'r25_point_gain':r25['delta_alc']<0,'r25_all_folds_improve':all(d<0 for d in r25['fold_deltas']),
            'r25_CI_upper_below_zero':r25['cluster_ci95'][1]<0,
            'remaining_to_smoothed_target':summary['results'][m]['alc_mean']-(summary['results']['smoothed_mean_2']['alc_mean']-.005),
            'mechanism_gate':'requires bounded interpretation; numerical pass alone is insufficient',
            'promotion':'not_promoted' if not simple_pass else 'requires_mechanism_and_independent_review'}
    result={'stage':stage,'trajectories':len(allrows),'unique_pairs':5242,'benchmarks':36,
            'query_variant_predictions':predictions,'chunks':chunks,'rescore_max_error':maxerror,
            'seconds':time.perf_counter()-start,'decisions':decisions}
    if stage=='R62':
        df=pd.DataFrame(details);df.to_csv(root/'low_budget_decomposition.csv',index=False)
        gd=pd.DataFrame(gate_rows);gd.to_csv(root/'gate_decisions.csv',index=False)
        result['low_budget_decomposition']=[]
        for scope,part in [('all',df),('alignment_faking',df[df.benchmark.str.startswith('alignment_faking/')]),
                           ('others',df[~df.benchmark.str.startswith('alignment_faking/')])]:
            for method in sorted(df.method.unique()):
                cells=part[part.method==method].groupby('budget')[['delta_mean','delta_variation','delta_brier']].mean()
                values={col:float(cells.loc[0,col]*.1+cells.loc[1,col]*.2) for col in cells.columns}
                result['low_budget_decomposition'].append({'scope':scope,'method':method,**values})
        result['gate']={'chosen_fraction':float(gd.chosen.mean()),'by_choice':[]}
        for chosen,part in gd.groupby('chosen'):
            result['gate']['by_choice'].append({'chosen':bool(chosen),'trajectories':len(part),
                'ungated_mean_delta_B1':float(part.ungated_delta_B1.mean()),
                'gated_mean_delta_B1':float(part.gated_delta_B1.mean())})
    elif stage=='R63':
        result['query_vs_episode']=paired_comparison(allrows,'R55_resampled_query','R55_resampled_episode')
        result['resampling_vs_fixed']=paired_comparison(allrows,'R55_resampled_query','R43_query_best20')
        result['policy_epochs']=EPOCHS
        result['resampling_vs_fixed_interpretation']='Historical candidate contrast, also differs in full-source row order and selected epoch; not a pure resampling effect'
    else:
        result['template_vs_raw_average']=paired_comparison(allrows,'template_average','R46_raw_average')
        result['permutation_comparisons']=[paired_comparison(allrows,'template_average',f'permuted_template_{i}') for i in range(4)]
        result['permutation_boundary']='Only item-uniform residual mean before clipping is preserved; not pure causal correspondence effect'
        df=pd.read_csv(root/'residual_decomposition.csv')
        result['decomposition']={}
        for baseline in ['R25','R46_raw_average']:
            v=df[df.baseline==baseline].groupby('budget')[['residual_alignment_term','displacement_cost','delta_brier']].mean().reindex(BUDGETS)
            result['decomposition'][baseline]={col:float(v[col].to_numpy()@ALC_WEIGHTS) for col in v.columns}
        result['changed_query_counts_by_budget']=np.sum([r['changed_queries'] for r in allrows],axis=0).astype(int).tolist()
        result['active_episode_counts_by_budget']=np.sum([r['active_template'] for r in allrows],axis=0).astype(int).tolist()
        result['permutation_clipped_counts']=np.sum([r['permutation_clipped_count'] for r in allrows],axis=0).astype(int).tolist()
    check_freeze(root/'evaluation_contract');check_freeze(ROOT/'R62/isolation')
    write_json(root/'analysis_audit.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['R62','R63','R64']);main(p.parse_args().stage)
