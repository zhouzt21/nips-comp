"""Replay saved R75/R76 probabilities and produce auditable paired result tables."""
import time,hashlib
import pandas as pd
from historical_positive_common import *
from analyze_historical_positive import paired_comparison

def source_comparison(rows,method,baseline):
    names=sorted({r['benchmark'] for r in rows})
    delta=np.array([float((np.array(r['brier'][method])-r['brier'][baseline])@ALC_WEIGHTS) for r in rows])
    counts=np.array([sum(r['benchmark']==b for r in rows) for b in names])
    sums=np.array([sum(d for r,d in zip(rows,delta) if r['benchmark']==b) for b in names])
    draws=np.random.default_rng(20261022).integers(len(names),size=(10000,len(names)))
    boot=sums[draws].sum(1)/counts[draws].sum(1)
    return {'delta':float(delta.mean()),'cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),'benchmark_equal_delta':float(np.mean(sums/counts))}

def replay(folder,source_stage=False):
    rows=read(folder/'paired_scores.json');lookup={r['identity']:r for r in rows};maxbr=0.;maxac=0.;count=0
    files=[folder/'index.json'] if source_stage else sorted(folder.glob('index_*.json'))
    for ip in files:
        npz=folder/'predictions.npz' if source_stage else folder/f'predictions_{ip.stem.split("_")[-1]}.npz'
        arr=dict(np.load(npz));offset=0
        for idx in read(ip):
            a=idx['start'];b=a+idx['size'];assert a==offset;offset=b
            meta={k:v for k,v in idx.items() if k not in ['identity','start','size','selection_fold']}
            assert hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()==idx['identity']
            c=arr['counts'][a:b,None];s=arr['successes'][a:b,None]
            assert np.array_equal(c[:,0],idx['counts']) and np.array_equal(s[:,0],idx['successes'])
            row=lookup[idx['identity']]
            for method,pall in arr.items():
                if method in ['counts','successes']:continue
                p=pall[a:b];assert p.shape==(b-a,6) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                br=(c*p*p-2*s*p+s).sum(0)/c.sum();ac=np.where(p>=.5,s,c-s).sum(0)/c.sum()
                maxbr=max(maxbr,float(np.max(np.abs(br-row['brier'][method]))));maxac=max(maxac,float(np.max(np.abs(ac-row['accuracy'][method]))))
            count+=b-a
        assert offset==len(arr['counts'])
    assert maxbr<1e-12 and maxac<1e-12
    return rows,{'folder':str(folder),'trajectories':len(rows),'query_variant_rows':count,'max_brier_error':maxbr,'max_accuracy_error':maxac,'chunks':len(files)}

def main():
    start=time.perf_counter();root=ROOT/'R76/full';summary=read(root/'summary.json');rows=[];checks=[];cost_frames=[]
    for folder in sorted((root/'outer').glob('fold*/*')):
        rr,cc=replay(folder);rows+=rr;checks.append(cc);cost_frames.append(pd.read_csv(folder/'label_costs.csv'))
    assert len(rows)==15726 and len({(r['benchmark'],r['subject_id']) for r in rows})==5242
    assert sum(v['chunks'] for v in checks)==149
    comparisons={}
    for m in summary['results']:
        comparisons[m]={b:paired_comparison(rows,m,b) for b in ['R25','R56_segment','R72_pooled3','R76_peer3'] if b!=m}
    write_json(root/'additional_comparisons.json',comparisons)
    pair_detail=[]
    for m,b in [('R76_peer2','R76_peer1'),('R76_peer3','R76_peer2'),('R76_alt3','R76_peer3'),('R76_partial_selected','R76_peer3'),('R76_partial_selected','R76_partial_perm')]:
        c=paired_comparison(rows,m,b);pair_detail.append({'method':m,'baseline':b,**c})
    write_json(root/'mechanism_comparisons.json',pair_detail)
    costs=pd.concat(cost_frames,ignore_index=True)
    agg=costs.groupby(['method','budget'])[['own_labels','peer_labels','active_peers','visible_total','largest_dual_matrix_entries']].mean().reset_index()
    for alias in ['R76_partial_selected','R76_partial_perm']:
        copy=agg[agg.method=='R76_peer3'].copy();copy['method']=alias;agg=pd.concat([agg,copy],ignore_index=True)
    agg.to_csv(root/'label_cost_summary.csv',index=False)
    costs.groupby(['benchmark','method','budget'])[['peer_labels','active_peers','visible_total']].mean().reset_index().to_csv(root/'label_cost_by_benchmark.csv',index=False)
    costmax=costs.groupby(['method','budget'])[['peer_labels','visible_total','largest_dual_matrix_entries']].max().reset_index();costmax.to_csv(root/'label_cost_max.csv',index=False)
    gates={}
    for m in summary['results']:
        if not m.startswith('R76_'):continue
        criteria={}
        for b in ['smoothed_mean_2','training_selected_smoothed']:
            c=summary['comparisons'][m][b]
            criteria[b]={'gain_ge_005':c['delta_alc']<=-.005,'three_fold_gain':all(v<0 for v in c['fold_deltas']),
                         'benchmark_equal_gain':c['benchmark_equal_delta']<0,'CI_upper_below_zero':c['cluster_ci95'][1]<0,
                         'regression_ge_005_for_review':c['regression_ge_005'],'regression_ge_03':c['regression_ge_03']}
        # Numerical first four conditions are distinct from adjudication of severe regressions.
        numerical=all(all(v[k] for k in ['gain_ge_005','three_fold_gain','benchmark_equal_gain','CI_upper_below_zero']) for v in criteria.values())
        gates[m]={'four_numerical_conditions':numerical,'regression_review_required':any(v['regression_ge_005_for_review'] for v in criteria.values()),
                  'criteria':criteria,'scope':'conditional_simulation','hosted_deployment':False}
    write_json(root/'gate.json',gates)
    # Source rows are explicitly deduplicated by Benchmark, taking the first eligible
    # source-fold context (a fixed reporting convention, not best-score selection).
    source_rows=[];source_checks=[];selected_benchmarks=set();source_count=0;source_coverage=[]
    for f in range(3):
        allowed=set(source(f)['train_benchmarks']);forbidden=set(source(f)['heldout_benchmarks']);assert allowed.isdisjoint(forbidden)
        for folder in sorted((ROOT/f'R75/source_fold{f}').iterdir()):
            if not folder.is_dir() or not (folder/'complete.json').exists():continue
            rr,cc=replay(folder,True);source_checks.append(cc);source_count+=len(rr)
            name=folder.name+'/formatted_tables';assert name in allowed
            if name not in selected_benchmarks:source_rows+=rr;selected_benchmarks.add(name)
        completed={p.name+'/formatted_tables' for p in (ROOT/f'R75/source_fold{f}').iterdir() if p.is_dir() and (p/'complete.json').exists()}
        source_coverage.append({'selection_fold':f,'declared_training_benchmarks':len(allowed),'actually_scored_benchmarks':len(completed),
             'scored_benchmarks':sorted(completed),'not_scored':sorted(allowed-completed),'outer_overlap':sorted(completed & forbidden),
             'diagnosed_benchmarks':read(ROOT/'R75/selection.json')[str(f)]['diagnosed_benchmarks']})
    dest=ROOT/'R75/deduplicated_source';dest.mkdir(exist_ok=True);srcsummary=summarize(source_rows,dest)
    factor_pairs=[('pool_both','R25'),('pool_sem','own_sem'),('pool_both','pool_sem'),('pool_global_center','pool_both'),
                  ('pool_both_R36ridge','pool_both'),('pool_sem_R36ridge','pool_sem'),('partial8','pool_both'),('partial8','partial8_perm')]
    factors=[{'method':m,'baseline':b,**source_comparison(source_rows,m,b)} for m,b in factor_pairs]
    write_json(dest/'factor_comparisons.json',factors)
    write_json(ROOT/'R75/source_replay_validation.json',{'checks':source_checks,'source_trajectory_exposures':source_count,'unique_reporting_trajectories':len(source_rows),
               'dedup_rule':'first eligible source fold per Benchmark, never best score','max_brier_error':max(v['max_brier_error'] for v in source_checks),
               'max_accuracy_error':max(v['max_accuracy_error'] for v in source_checks)})
    write_json(ROOT/'R75/source_coverage.json',source_coverage)
    memory=pd.read_csv(ROOT/'R76/process_memory.csv')
    resource={'peak_working_set_mb':float(memory.lifetime_peak_working_set_mb.max()),'sampling_count':len(memory),'measurement':'Windows Get-Process lifetime PeakWorkingSet64',
              'rejected_in_process_metric':'ctypes metric returned 0 and is invalid; source process peak unavailable',
              'max_visible_labels':int(costs.visible_total.max()),'max_dual_matrix_size':int(np.sqrt(costs.largest_dual_matrix_entries.max())),
              'dual_kernel_float64_bytes':int(costs.largest_dual_matrix_entries.max()*8),'seconds_prediction':read(root/'execution.json')['seconds'],
              'seconds_source':read(ROOT/'R75/execution.json')['seconds']}
    write_json(root/'resources.json',resource)
    validation={'trajectories':len(rows),'pairs':5242,'benchmarks':36,'chunks':sum(v['chunks'] for v in checks),'query_variant_rows':sum(v['query_variant_rows'] for v in checks),
                'max_brier_error':max(v['max_brier_error'] for v in checks),'max_accuracy_error':max(v['max_accuracy_error'] for v in checks),
                'source_replay_passed':True,'seconds':time.perf_counter()-start,'finished':now(),'checks':checks}
    write_json(root/'replay_validation.json',validation)
    # Scientific figures use normal plotting tools and are standalone artifacts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=REPO/'reports/shared_slope_figures';figdir.mkdir(exist_ok=True)
    methods=['R25','R72_pooled3','R76_peer1','R76_peer2','R76_peer3','R76_alt3','R76_partial_selected']
    fig,ax=plt.subplots(figsize=(8,4.7))
    for m in methods:ax.plot(range(6),summary['results'][m]['brier_mean'],marker='o',label=m)
    ax.set_xticks(range(6),BUDGETS);ax.set_xlabel('Own support budget');ax.set_ylabel('Pair-equal Brier (lower is better)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/'budget_brier.png',dpi=180);plt.close(fig)
    methods=['R76_peer0','R76_peer1','R76_peer2','R76_peer3','R76_missing','R76_uneven','R76_alt3','R76_partial_selected']
    vals=[comparisons[m]['R25']['delta'] for m in methods];ci=np.array([comparisons[m]['R25']['cluster_ci95'] for m in methods])
    fig,ax=plt.subplots(figsize=(8,4.7));ax.errorbar(vals,range(len(vals)),xerr=np.array([np.array(vals)-ci[:,0],ci[:,1]-vals]),fmt='o');ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(len(vals)),methods);ax.set_xlabel('ALC difference vs R25, Benchmark-cluster 95% CI');fig.tight_layout();fig.savefig(figdir/'cohort_delta_ci.png',dpi=180);plt.close(fig)
    check_freeze(ROOT/'R75/contract');check_freeze(root/'contract');check_freeze(ROOT/'R72/full/contract');check_freeze(ROOT/'R72/coverage_contract');check_freeze(ROOT/'R62/isolation')
    print(json.dumps({'validation':{k:v for k,v in validation.items() if k!='checks'},'mechanisms':pair_detail,'resources':resource}),flush=True)

if __name__=='__main__':main()
