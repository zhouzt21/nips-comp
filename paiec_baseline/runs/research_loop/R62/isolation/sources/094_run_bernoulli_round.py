"""R39 isolated-pair binary likelihood with matched local curvature and fixed features."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.cohort_semantic import CachedSemantic
from mlp.measurement_episodes import episodes,weighted_brier
from mlp.local_bernoulli import fit_predict
from mlp.run.run_metadata_round import prediction
from mlp.evaluate import BUDGETS,ALC_WEIGHTS
from mlp.reporting.summarize_candidate import summarize

NAMES=['bernoulli','one_step','R25','smoothed_mean_2','training_selected_smoothed','empirical_mean']


def main():
    start=time.time();root=Path('runs/research_loop/R39');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic()
    sources=['mlp/run/run_bernoulli_round.py','mlp/local_bernoulli.py','mlp/measurement_episodes.py','mlp/run/run_metadata_round.py','../research_loop/R39_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    diagnostics=[]
    for fold in range(3):
        selection=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text())
        reference=json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text());total={k:np.zeros(6) for k in NAMES};by={};count=0
        for directory,old in reference['by_benchmark'].items():
            target=root/(directory.split('/')[0]+'_evaluation.json')
            if target.exists():current=json.loads(target.read_text())
            else:
                sums={k:np.zeros(6) for k in NAMES};seed_sums={s:{k:np.zeros(6) for k in NAMES} for s in [0,1,2]};nrows=0;iterations=[];gradient=0.;fits=0
                with (root/(directory.split('/')[0]+'_pair_brier.csv')).open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.writer(stream);writer.writerow(['subject_id','seed','budget',*NAMES,'iterations','gradient_inf'])
                    for row in episodes(directory,engine,selection['semantic_ridge'],selection['metadata_ridge']):
                        for j,n in enumerate(BUDGETS):
                            y=row['labels'][:n];p,first,diag=fit_predict(row['support'][:n],row['query'],y)
                            assert diag['converged'],(directory,row['subject_id'],row['seed'],n,diag)
                            assert np.all(np.diff(diag['objective_history'])<=1e-9)
                            iterations.append(diag['iterations']);gradient=max(gradient,diag['gradient_inf']);fits+=1
                            predictions={'bernoulli':p,'one_step':first,'R25':prediction(row['support'][:n],row['query'],y,1.)}
                            for k,alpha in [('smoothed_mean_2',2.),('training_selected_smoothed',[2.,2.,4.][fold]),('empirical_mean',0.)]:
                                predictions[k]=(y.sum()+alpha*.5)/(n+alpha) if n+alpha else .5
                            values={k:weighted_brier(v,row['counts'],row['successes']) for k,v in predictions.items()}
                            for k,v in values.items():sums[k][j]+=v;seed_sums[row['seed']][k][j]+=v
                            writer.writerow([row['subject_id'],row['seed'],n,*[values[k] for k in NAMES],diag['iterations'],diag['gradient_inf']])
                        nrows+=1
                assert nrows==3*old['pairs_per_seed'][0];pairs=nrows//3;avg={k:v/nrows for k,v in sums.items()}
                for k in NAMES[2:]:np.testing.assert_allclose(avg[k],old['brier_mean'][k],atol=1e-7,rtol=0)
                current={'pairs_per_seed':[pairs]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},
                         'alc_per_seed':{seed:{k:float(v/pairs@ALC_WEIGHTS) for k,v in ss.items()} for seed,ss in seed_sums.items()},
                         'solver':{'fits':fits,'nonconverged':0,'maximum_iterations':int(max(iterations)),'mean_iterations':float(np.mean(iterations)),'maximum_gradient_inf':gradient,'fallbacks':0}}
                target.write_text(json.dumps(current,indent=2),encoding='utf-8')
            n=current['pairs_per_seed'][0];count+=n;by[directory]=current;diagnostics.append(current['solver'])
            for k in NAMES:total[k]+=n*np.asarray(current['brier_mean'][k])
            print(directory,'pairs',n,'max iterations',current['solver']['maximum_iterations'],flush=True)
        result={'selection':selection,'pairs_per_seed':[count]*3,'by_benchmark':by,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],NAMES[0],NAMES[1:])
    d.update(scope='isolated pair,all5242pairs,three seeds,64queryitems; reused benchmarks; no new hyperparameter selection',
             elapsed_seconds=time.time()-start,solver_fits=sum(x['fits'] for x in diagnostics),maximum_iterations=max(x['maximum_iterations'] for x in diagnostics),
             maximum_gradient_inf=max(x['maximum_gradient_inf'] for x in diagnostics),nonconverged=0,fallbacks=0)
    d['gate_checks']={k:{'gain_at_least_005':-v['pair_equal_delta']>=.005,'all_folds':all(x<0 for x in v['fold_deltas']),
        'benchmark_equal_improves':d['benchmark_equal_alc']['bernoulli']<d['benchmark_equal_alc'][k],
        'cluster_ci_below_zero':v['cluster_bootstrap_ci95'][1]<0,'no_large_regression':v['worst_benchmark_delta']<.03}
        for k,v in d['comparisons'].items() if k in ['smoothed_mean_2','training_selected_smoothed']}
    d['numeric_gate_passes']=all(all(v.values()) for v in d['gate_checks'].values())
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    for p in sources:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps({k:d[k] for k in ['pair_equal_alc','numeric_gate_passes','elapsed_seconds','maximum_iterations','maximum_gradient_inf']},indent=2),flush=True)


if __name__=='__main__':main()
