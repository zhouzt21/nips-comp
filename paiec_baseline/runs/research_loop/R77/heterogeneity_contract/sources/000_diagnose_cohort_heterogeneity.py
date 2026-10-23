"""Secondary H3 diagnostic on fixed current B31 supports, never model selection."""
import time
from scipy.stats import spearmanr
from shared_reliability_common import *
from shared_slope_model import heterogeneity

def main():
    start=time.perf_counter();from threadpoolctl import threadpool_limits
    threadpool_limits(4);root=ROOT/'R77'
    freeze(root/'heterogeneity_contract',[Path(__file__),REPO/'research_loop/SHARED_RELIABILITY_HETEROGENEITY_NOTE.md',REPO/'research_loop/shared_slope_model.py'],
           {'budget':31,'seed':0,'cohorts':COHORTS,'group_rule':'first full group by visible subject key','reps':32,'no_model_selection':True})
    assert (root/'full/execution.json').exists()
    engine=CachedSemantic();rows=[]
    for f in range(3):
        for old in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
            bm=Benchmark(old,source(f)['semantic_ridge'],engine);records={v['subject_id']:v for v in bm.records if v['seed']==0}
            scores={v['subject_id']:v for v in read(root/f'full/outer/fold{f}'/old.name/'paired_scores.json') if v['seed']==0}
            for c in COHORTS:
                mapping=cohort_map(bm.records,c);eligible=sorted(records,key=lambda s:records[s]['subject_key'])
                anchor=next((s for s in eligible if len(mapping[s])==3),None)
                if anchor is None:continue
                group=[anchor,*mapping[anchor]]
                h=heterogeneity([bm.context(s,0,31) for s in group],reps=32,seed=stable_seed(f'7709|{bm.name}|{c}'))
                delta=float(np.mean([scores[s]['brier'][f'partial_{c}'][5]-scores[s]['brier'][f'pool_{c}'][5] for s in group]))
                rows.append({'fold':f,'benchmark':bm.name,'cohort':c,'subjects':group,'visible_keys':[records[s]['subject_key'] for s in group],
                             **h,'B31_partial_minus_pool':delta})
    write_json(root/'cohort_heterogeneity.json',rows)
    df=pd.DataFrame([{k:v for k,v in r.items() if k not in ['subjects','visible_keys','null_H','half_cosines']} for r in rows]);df.to_csv(root/'cohort_heterogeneity.csv',index=False)
    def corr(d):
        return {'benchmark_rows':len(d),'spearman_H_vs_query_increment':float(spearmanr(d.H,d.B31_partial_minus_pool).statistic),
                'interpretation':'negative rho would associate more observed heterogeneity with larger partial gain; model dependent, exploratory'}
    result={'by_cohort':{str(c):corr(g) for c,g in df.groupby('cohort')},'benchmark_average':corr(df.groupby('benchmark').mean(numeric_only=True)),
            'budget':31,'item_seed':0,'groups':len(df),'benchmarks':df.benchmark.nunique(),'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'finished':now()}
    write_json(root/'heterogeneity_association.json',result);check_freeze(root/'heterogeneity_contract');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
