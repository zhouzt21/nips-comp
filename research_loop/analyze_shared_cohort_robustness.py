"""R77 independent replay and BM-cluster composition-aware inference."""
import time
from shared_reliability_common import *
from shared_slope_analyze import replay

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    start=time.perf_counter();root=ROOT/'R77';rows=[];checks=[]
    for stage in ['alternative','full']:
        rr=[];cc=[]
        for folder in sorted((root/stage/'outer').glob('fold*/*')):
            a,b=replay(folder);rr+=a;cc.append(b)
        assert len(rr)==15726
        write_json(root/stage/'replay_validation.json',{'pairs':5242,'trajectories':len(rr),'benchmarks':len(cc),'chunks':sum(v['chunks'] for v in cc),
             'query_variant_rows':sum(v['query_variant_rows'] for v in cc),'max_brier_error':max(v['max_brier_error'] for v in cc),
             'max_accuracy_error':max(v['max_accuracy_error'] for v in cc),'checks':cc})
        if stage=='full':rows=rr
        checks+=cc
    comp={}
    for c in COHORTS:
        comp[str(c)]=bm_comparison(rows,f'partial_{c}',f'pool_{c}')
        comp[str(c)]['partial_vs_shrink']=bm_comparison(rows,f'partial_{c}',f'shrink_{c}')
        comp[str(c)]['partial_vs_own']=bm_comparison(rows,f'partial_{c}','R25')
        comp[str(c)]['partial_vs_means']={m:bm_comparison(rows,f'partial_{c}',m) for m in ['smoothed_mean_2','training_selected_smoothed']}
    for r in rows:
        for field in ['brier','accuracy']:
            for kind in ['pool','partial','shrink']:
                r[field][kind+'_cohort_average']=np.mean([r[field][f'{kind}_{c}'] for c in COHORTS],axis=0).tolist()
    avg=bm_comparison(rows,'partial_cohort_average','pool_cohort_average')
    avg['definition']='mean risk across the five cohorts within each pair/seed; not probability ensembling'
    avg['partial_vs_shrink']=bm_comparison(rows,'partial_cohort_average','shrink_cohort_average')
    scenario={s:bm_comparison(rows,f'partial_{s}',f'pool_{s}') for s in ['peer0','peer1','peer2','missing','uneven','ambiguous']}
    good=sum(v['delta']<0 for v in comp.values());severe={c:v['regression_ge_005'] for c,v in comp.items() if v['regression_ge_005']}
    passes=good>=4 and avg['cluster_ci95'][1]<0 and not severe
    decision={'cohorts_improved':good,'cohorts':5,'average_increment':avg['delta'],'average_ci95':avg['cluster_ci95'],
        'new_severe_partial_vs_pool':severe,'status':'partial_sharing_composition_robust_in_frozen_scenarios' if passes else 'partial_sharing_composition_sensitive',
        'no_best_cohort_selection':True,'hosted':False,'claim_scope':'fixed public cohort sensitivity, not new Benchmark blind validation'}
    write_json(root/'comparisons.json',{'cohorts':comp,'cohort_average':avg,'scenarios':scenario});write_json(root/'decision.json',decision)
    matrix=pd.DataFrame({c:v['by_benchmark'] for c,v in comp.items()}).sort_index()
    matrix['min_delta']=matrix.min(axis=1);matrix['max_delta']=matrix[[str(c) for c in COHORTS]].max(axis=1)
    matrix['sign_flip']=(matrix.min_delta < -1e-10)&(matrix.max_delta > 1e-10)
    matrix.to_csv(root/'benchmark_cohort_delta.csv',index_label='benchmark')
    # Exact displacement accounting: use response weights and equal trajectories.
    geom=[]
    for folder in sorted((root/'full/outer').glob('fold*/*')):
        for ip in sorted(folder.glob('index_*.json')):
            arr=dict(np.load(folder/f'predictions_{ip.stem.split("_")[-1]}.npz'))
            for idx in read(ip):
                a=idx['start'];b=a+idx['size'];w=arr['counts'][a:b,None];w=w/w.sum();y=arr['successes'][a:b,None]/arr['counts'][a:b,None]
                old=arr['R25'][a:b]
                for c in COHORTS:
                    pool=arr[f'pool_{c}'][a:b];part=arr[f'partial_{c}'][a:b];d=pool-old;e=part-old;diff=part-pool
                    geom.append({'benchmark':idx['benchmark'],'subject_id':idx['subject_id'],'seed':idx['seed'],'cohort':c,
                        'pool_displacement_cost':float(((w*d*d).sum(0))@ALC_WEIGHTS),
                        'partial_displacement_cost':float(((w*e*e).sum(0))@ALC_WEIGHTS),
                        'partial_pool_cross':float(((w*d*e).sum(0))@ALC_WEIGHTS),
                        'increment_alignment':float(((2*w*(pool-y)*diff).sum(0))@ALC_WEIGHTS),
                        'increment_squared_displacement':float(((w*diff*diff).sum(0))@ALC_WEIGHTS)})
    pd.DataFrame(geom).to_csv(root/'prediction_geometry.csv',index=False)
    pd.DataFrame(geom).groupby(['benchmark','cohort']).mean(numeric_only=True).to_csv(root/'geometry_by_benchmark_cohort.csv')
    plot_results(comp,matrix)
    write_json(root/'analysis_execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'finished':now(),'matrix_sign_flips':int(matrix.sign_flip.sum())})
    print(json.dumps({'decision':decision,'cohorts':{c:v['delta'] for c,v in comp.items()},'scenarios':{c:v['delta'] for c,v in scenario.items()}}),flush=True)

def plot_results(comp,matrix):
    out=REPO/'reports/shared_reliability_figures';out.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    fig,ax=plt.subplots(figsize=(8,4.5))
    vals=np.array([comp[str(c)]['delta'] for c in COHORTS]);ci=np.array([comp[str(c)]['cluster_ci95'] for c in COHORTS])
    ax.errorbar(vals,range(5),xerr=np.array([vals-ci[:,0],ci[:,1]-vals]),fmt='o');ax.axvline(0,c='gray',ls='--');ax.set_yticks(range(5),COHORTS)
    ax.xaxis.set_major_locator(MaxNLocator(5));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x*1e4:.2f}'))
    ax.set_ylabel('Frozen peer cohort seed');ax.set_xlabel('Partial8 minus matched pooled ALC (×10⁻⁴; BM-cluster 95% CI)');fig.tight_layout();fig.savefig(out/'cohort_increment.png',dpi=180,bbox_inches='tight');plt.close(fig)
    data=matrix[[str(c) for c in COHORTS]];lim=float(abs(data.to_numpy()).max())
    fig,ax=plt.subplots(figsize=(9,10),layout='constrained');im=ax.imshow(data,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
    ax.set_xticks(range(5),COHORTS);ax.set_yticks(range(len(data)),[x.split('/')[0] for x in data.index]);ax.tick_params(axis='y',labelsize=8)
    fig.colorbar(im,ax=ax,label='Partial8 minus pooled ALC');fig.savefig(out/'benchmark_cohort_matrix.png',dpi=180,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
    import sys
    if '--plots-only' in sys.argv:
        plot_results(read(ROOT/'R77/comparisons.json')['cohorts'],pd.read_csv(ROOT/'R77/benchmark_cohort_delta.csv',index_col='benchmark'))
    else:main()
