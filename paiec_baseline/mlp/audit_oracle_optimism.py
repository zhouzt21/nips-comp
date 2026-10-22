"""R48 cross-panel audit of R33's outcome-fitted scale, not an actual predictor."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from .cohort_semantic import CachedSemantic
from .measurement_episodes import episodes,weighted_brier
from .run_metadata_round import prediction
from .oracle_scale import optimum_scale
from .oracle_crossfit import crossfit
from .summarize_candidate import summarize

NAMES=['crossfit','permuted_calibration','extra_label_mean','oracle_max4','R25','smoothed_mean_2']


def main():
    start=time.time();root=Path('runs/research_loop/R48');root.mkdir(parents=True,exist_ok=True)
    source=['mlp/audit_oracle_optimism.py','mlp/oracle_crossfit.py','mlp/oracle_scale.py','mlp/measurement_episodes.py','mlp/run_metadata_round.py','research_loop/R48_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in source:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    engine=CachedSemantic();full_result=None
    for fold in range(3):
        selection=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text())
        ref=json.loads(Path(f'runs/research_loop/R33/fold{fold}_evaluation.json').read_text())
        total={k:np.zeros(6) for k in NAMES};count=0;by={}
        for directory,old in ref['by_benchmark'].items():
            target=root/(directory.split('/')[0]+'_evaluation.json');dominant=directory=='alignment_faking/formatted_tables'
            if target.exists():result=json.loads(target.read_text())
            else:
                sums={scope:{k:np.zeros(6) for k in NAMES} for scope in ['capped','full']};nrows=0;panel_counts={'capped':[],'full':[]}
                seed_sums={seed:{k:np.zeros(6) for k in NAMES} for seed in [0,1,2]}
                with (root/(directory.split('/')[0]+'_pair_scores.csv')).open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.writer(stream);writer.writerow(['subject_id','seed','budget','scope',*NAMES,'coefficient_A','coefficient_B','query_items'])
                    for row in episodes(directory,engine,selection['semantic_ridge'],selection['metadata_ridge'],full=dominant):
                        scopes=[('capped',min(64,len(row['query']))),*([('full',len(row['query']))] if dominant else [])]
                        orders={scope:np.random.default_rng(_stable_seed(4801,directory,f"{row['subject_id']}:{row['seed']}:{scope}")).permutation(length) for scope,length in scopes}
                        for scope,length in scopes:panel_counts[scope].append(length//2)
                        for j,n in enumerate(BUDGETS):
                            a=row['labels'][:n];base=(1+a.sum())/(2+n);p=prediction(row['support'][:n],row['query'],a,1.)
                            for scope,length in scopes:
                                m=row['counts'][:length];s=row['successes'][:length];q=p[:length];d=q-base
                                vals,coeff=crossfit(base,d,m,s,orders[scope],_stable_seed(4803,directory,f"{row['subject_id']}:{row['seed']}:{scope}:{n}"))
                                vals['oracle_max4'],_=optimum_scale(base,d,m,s,4.)
                                vals['R25']=weighted_brier(q,m,s);vals['smoothed_mean_2']=weighted_brier(base,m,s)
                                assert vals['oracle_max4']<=vals['R25']+1e-10
                                for k,v in vals.items():
                                    sums[scope][k][j]+=v
                                    if scope=='capped':seed_sums[row['seed']][k][j]+=v
                                writer.writerow([row['subject_id'],row['seed'],n,scope,*[vals[k] for k in NAMES],*coeff,length])
                        nrows+=1
                pairs=nrows//3;assert nrows==3*old['pairs_per_seed'][0]
                avg={k:v/nrows for k,v in sums['capped'].items()}
                for k in ['oracle_max4','R25','smoothed_mean_2']:np.testing.assert_allclose(avg[k],old['brier_mean'][k],atol=1e-7,rtol=0)
                result={'pairs_per_seed':[pairs]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},
                        'alc_per_seed':{seed:{k:float(v/pairs@ALC_WEIGHTS) for k,v in values.items()} for seed,values in seed_sums.items()},
                        'calibration_item_count_quantiles':{scope:np.quantile(v,[0,.5,1]).tolist() for scope,v in panel_counts.items() if v}}
                if dominant:
                    full={k:v/nrows for k,v in sums['full'].items()}
                    expected=json.loads(Path('runs/research_loop/R33/summary.json').read_text())['dominant_full_query']
                    for k in ['oracle_max4','R25','smoothed_mean_2']:np.testing.assert_allclose(full[k],expected['full_brier'][k],atol=1e-7,rtol=0)
                    result['full']={'pairs':pairs,'full_brier':{k:v.tolist() for k,v in full.items()},'full_alc':{k:float(v@ALC_WEIGHTS) for k,v in full.items()}}
                target.write_text(json.dumps(result,indent=2),encoding='utf-8')
            if dominant:full_result=result['full']
            pairs=result['pairs_per_seed'][0];count+=pairs;by[directory]=result
            for k in NAMES:total[k]+=pairs*np.array(result['brier_mean'][k])
            print(directory,'pairs',pairs,flush=True)
        result={'pairs_per_seed':[count]*3,'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'crossfit',NAMES[1:])
    d.update(scope='DIAGNOSTIC ONLY: extra opposite-panel evaluation labels, NOT budget compliant; all5242capped plus full4278dominant; reused benchmarks',
             dominant_full_query=full_result,elapsed_seconds=time.time()-start,eligible_for_promotion=False)
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    for p in source:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps({k:d[k] for k in ['pair_equal_alc','dominant_full_query','elapsed_seconds','eligible_for_promotion']},indent=2),flush=True)


if __name__=='__main__':main()
