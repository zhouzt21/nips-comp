"""R46 frozen semantic/lexical probability average on all original outer pairs."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.cohort_semantic import CachedSemantic
from mlp.measurement_episodes import episodes,weighted_brier
from mlp.run.run_complementarity_pilot import LexicalWindows
from mlp.run.run_metadata_round import prediction
from mlp.evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from mlp.reporting.summarize_candidate import summarize

NAMES=['average','lexical','permuted_lexical','R25','smoothed_mean_2','training_selected_smoothed','empirical_mean']


def main():
    assert json.loads(Path('runs/research_loop/R45/summary.json').read_text())['admit_outer']
    start=time.time();root=Path('runs/research_loop/R46');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic();lexical=LexicalWindows(engine)
    source=['mlp/run/run_complementarity_outer.py','mlp/run/run_complementarity_pilot.py','mlp/measurement_episodes.py','mlp/run/run_metadata_round.py','../research_loop/R46_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in source:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        selection=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text())
        reference=json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text());total={k:np.zeros(6) for k in NAMES};by={};count=0
        for directory,old in reference['by_benchmark'].items():
            target=root/(directory.split('/')[0]+'_evaluation.json')
            if target.exists():current=json.loads(target.read_text())
            else:
                sums={k:np.zeros(6) for k in NAMES};seed_sums={s:{k:np.zeros(6) for k in NAMES} for s in [0,1,2]};nrows=0;dividend=np.zeros(6);error=0.
                with (root/(directory.split('/')[0]+'_pair_brier.csv')).open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.writer(stream);writer.writerow(['subject_id','seed','budget',*NAMES,'disagreement_dividend'])
                    for row in episodes(directory,engine,selection['semantic_ridge'],selection['metadata_ridge']):
                        lex_s=np.c_[np.stack([lexical.vector(v*np.sqrt(selection['semantic_ridge'])) for v in row['support'][:,:384]]),row['support'][:,384:]]
                        lex_q=np.c_[np.stack([lexical.vector(v*np.sqrt(selection['semantic_ridge'])) for v in row['query'][:,:384]]),row['query'][:,384:]]
                        rng=np.random.default_rng(_stable_seed(4601,directory,row['subject_id']+str(row['seed'])))
                        for j,n in enumerate(BUDGETS):
                            y=row['labels'][:n];p=prediction(row['support'][:n],row['query'],y,1.);q=prediction(lex_s[:n],lex_q,y,1.)
                            predictions={'average':.5*(p+q),'R25':p,'lexical':q}
                            for k,alpha in [('smoothed_mean_2',2.),('training_selected_smoothed',[2.,2.,4.][fold]),('empirical_mean',0.)]:
                                predictions[k]=(y.sum()+alpha*.5)/(n+alpha) if n+alpha else .5
                            values={k:weighted_brier(v,row['counts'],row['successes']) for k,v in predictions.items()}
                            values['permuted_lexical']=float(np.mean([weighted_brier(.5*(p+q[rng.permutation(len(q))]),row['counts'],row['successes']) for _ in range(4)]))
                            gain=float(.25*np.average((p-q)**2,weights=row['counts']));dividend[j]+=gain
                            error=max(error,abs(values['average']-(.5*values['R25']+.5*values['lexical']-gain)));assert error<1e-12
                            for k,v in values.items():sums[k][j]+=v;seed_sums[row['seed']][k][j]+=v
                            writer.writerow([row['subject_id'],row['seed'],n,*[values[k] for k in NAMES],gain])
                        nrows+=1
                assert nrows==3*old['pairs_per_seed'][0];pairs=nrows//3;avg={k:v/nrows for k,v in sums.items()}
                for k in NAMES[3:]:np.testing.assert_allclose(avg[k],old['brier_mean'][k],atol=1e-7,rtol=0)
                current={'pairs_per_seed':[pairs]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},
                         'alc_per_seed':{seed:{k:float(v/pairs@ALC_WEIGHTS) for k,v in ss.items()} for seed,ss in seed_sums.items()},
                         'disagreement_dividend':(dividend/nrows).tolist(),'identity_error':error}
                target.write_text(json.dumps(current,indent=2),encoding='utf-8')
            n=current['pairs_per_seed'][0];count+=n;by[directory]=current
            for k in NAMES:total[k]+=n*np.asarray(current['brier_mean'][k])
            lexical.features.clear();print(directory,'pairs',n,flush=True)
        result={'selection':selection,'pairs_per_seed':[count]*3,'by_benchmark':by,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],NAMES[0],NAMES[1:])
    d.update(scope='isolated pair,all5242pairs,three seeds,64queryitems; reused benchmarks; fixed equal average',elapsed_seconds=time.time()-start)
    d['mechanism_gate']=all(all(x<0 for x in d['comparisons'][k]['fold_deltas']) and d['comparisons'][k]['cluster_bootstrap_ci95'][1]<0 for k in ['R25','permuted_lexical'])
    d['numeric_gate_passes']=all(-d['comparisons'][k]['pair_equal_delta']>=.005 and all(x<0 for x in d['comparisons'][k]['fold_deltas']) and d['comparisons'][k]['cluster_bootstrap_ci95'][1]<0 and d['benchmark_equal_alc']['average']<d['benchmark_equal_alc'][k] and d['comparisons'][k]['worst_benchmark_delta']<.03 for k in ['smoothed_mean_2','training_selected_smoothed'])
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    for p in source:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps({k:d[k] for k in ['pair_equal_alc','mechanism_gate','numeric_gate_passes','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
