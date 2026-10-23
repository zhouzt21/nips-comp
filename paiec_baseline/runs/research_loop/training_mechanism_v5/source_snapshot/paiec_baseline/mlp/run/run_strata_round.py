"""R40 frozen input strata, model-evidence update and correspondence controls."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.cohort_semantic import CachedSemantic
from mlp.measurement_episodes import episodes,weighted_brier
from mlp.local_strata import partition,count_predict
from mlp.run.run_metadata_round import prediction
from mlp.evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from mlp.reporting.summarize_candidate import summarize

NAMES=['strata','random_direction','permuted_labels','R25','smoothed_mean_2','training_selected_smoothed','empirical_mean']


def main():
    start=time.time();root=Path('runs/research_loop/R40');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic()
    sources=['mlp/run/run_strata_round.py','mlp/local_strata.py','mlp/measurement_episodes.py','mlp/run/run_metadata_round.py','../research_loop/R40_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        selection=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text())
        reference=json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text());total={k:np.zeros(6) for k in NAMES};by={};count=0
        for directory,old in reference['by_benchmark'].items():
            target=root/(directory.split('/')[0]+'_evaluation.json')
            if target.exists():current=json.loads(target.read_text())
            else:
                sums={k:np.zeros(6) for k in NAMES};seed_sums={s:{k:np.zeros(6) for k in NAMES} for s in [0,1,2]};nrows=0;degenerate=np.zeros(6,int);weights=np.zeros(6)
                with (root/(directory.split('/')[0]+'_pair_brier.csv')).open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.writer(stream);writer.writerow(['subject_id','seed','budget',*NAMES,'degenerate','split_weight'])
                    for row in episodes(directory,engine,selection['semantic_ridge'],selection['metadata_ridge']):
                        axes=np.random.default_rng(4001).normal(size=(4,row['support'].shape[1]));axes/=np.linalg.norm(axes,axis=1,keepdims=True)
                        rng=np.random.default_rng(_stable_seed(4003,directory,row['subject_id']+str(row['seed'])))
                        for j,n in enumerate(BUDGETS):
                            y=row['labels'][:n];x=row['support'][:n];q=row['query'];base=(1+y.sum())/(2+n)
                            groups=partition(x,q);degenerate[j]+=groups is None
                            if groups is None:p=np.full(len(q),base);weight=0.
                            else:p,weight=count_predict(*groups,y)
                            weights[j]+=weight
                            predictions={'strata':p,'R25':prediction(x,q,y,1.)}
                            for k,alpha in [('smoothed_mean_2',2.),('training_selected_smoothed',[2.,2.,4.][fold]),('empirical_mean',0.)]:
                                predictions[k]=(y.sum()+alpha*.5)/(n+alpha) if n+alpha else .5
                            values={k:weighted_brier(v,row['counts'],row['successes']) for k,v in predictions.items()}
                            random_losses=[];perm_losses=[]
                            for axis in axes:
                                rg=partition(x,q,axis);rp=np.full(len(q),base) if rg is None else count_predict(*rg,y)[0]
                                pp=np.full(len(q),base) if groups is None else count_predict(*groups,y[rng.permutation(n)])[0]
                                random_losses.append(weighted_brier(rp,row['counts'],row['successes']));perm_losses.append(weighted_brier(pp,row['counts'],row['successes']))
                            values['random_direction']=float(np.mean(random_losses));values['permuted_labels']=float(np.mean(perm_losses))
                            for k,v in values.items():sums[k][j]+=v;seed_sums[row['seed']][k][j]+=v
                            writer.writerow([row['subject_id'],row['seed'],n,*[values[k] for k in NAMES],groups is None,weight])
                        nrows+=1
                assert nrows==3*old['pairs_per_seed'][0];pairs=nrows//3;avg={k:v/nrows for k,v in sums.items()}
                for k in NAMES[3:]:np.testing.assert_allclose(avg[k],old['brier_mean'][k],atol=1e-7,rtol=0)
                current={'pairs_per_seed':[pairs]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},
                         'alc_per_seed':{seed:{k:float(v/pairs@ALC_WEIGHTS) for k,v in ss.items()} for seed,ss in seed_sums.items()},
                         'degenerate_counts':degenerate.tolist(),'mean_split_weight':(weights/nrows).tolist()}
                target.write_text(json.dumps(current,indent=2),encoding='utf-8')
            n=current['pairs_per_seed'][0];count+=n;by[directory]=current
            for k in NAMES:total[k]+=n*np.asarray(current['brier_mean'][k])
            print(directory,'pairs',n,flush=True)
        result={'selection':selection,'pairs_per_seed':[count]*3,'by_benchmark':by,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],NAMES[0],NAMES[1:])
    d.update(scope='isolated pair,all5242pairs,three seeds,64queryitems; reused benchmarks; frozen prior odds1:1; no tuning',elapsed_seconds=time.time()-start)
    d['mechanism_gate']=all(all(x<0 for x in d['comparisons'][k]['fold_deltas']) and d['comparisons'][k]['cluster_bootstrap_ci95'][1]<0 for k in ['R25','random_direction','permuted_labels'])
    d['numeric_gate_passes']=all(-d['comparisons'][k]['pair_equal_delta']>=.005 and all(x<0 for x in d['comparisons'][k]['fold_deltas']) and d['comparisons'][k]['cluster_bootstrap_ci95'][1]<0 and d['benchmark_equal_alc']['strata']<d['benchmark_equal_alc'][k] and d['comparisons'][k]['worst_benchmark_delta']<.03 for k in ['smoothed_mean_2','training_selected_smoothed'])
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    for p in sources:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps({k:d[k] for k in ['pair_equal_alc','mechanism_gate','numeric_gate_passes','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
