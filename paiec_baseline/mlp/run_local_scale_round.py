"""R34 source-selected support-only scaling, matched fixed-scale and correspondence control."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .cohort_semantic import CachedSemantic
from .evaluate import BUDGETS, ALC_WEIGHTS, _stable_seed
from .measurement_episodes import episodes, weighted_brier
from .local_scale import smoother, coefficient
from .run_metadata_round import prediction
from .summarize_candidate import summarize

TAUS = [None, .1, 1., 10.]
CONSTANTS = [.5, 1., 2., 4.]
NAMES = ['adaptive_scale','fixed_scale','scale_permuted','R25','smoothed_mean_2','training_selected_smoothed','empirical_mean']


def components(row, n):
    a = row['labels'][:n]; b = (1+a.sum())/(2+n)
    p = prediction(row['support'][:n],row['query'],a,1.)
    h = smoother(row['support'][:n]) if n >= 3 else None
    return a,b,p,h


def main():
    start = time.time(); root = Path('runs/research_loop/R34'); root.mkdir(parents=True,exist_ok=True)
    sources = ['mlp/run_local_scale_round.py','mlp/local_scale.py','mlp/measurement_episodes.py',
               'mlp/run_metadata_round.py','research_loop/R34_protocol.md']
    hashes = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if (root/'provenance.json').exists(): assert json.loads((root/'provenance.json').read_text()) == hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    snap = root/'code'; snap.mkdir(exist_ok=True)
    for p in sources: (snap/Path(p).name).write_bytes(Path(p).read_bytes())
    engine = CachedSemantic(); selections = []
    # All source selections finish before any outer scoring.
    for fold in range(3):
        dest = root/f'fold{fold}_selection.json'
        if dest.exists(): selections.append(json.loads(dest.read_text())); continue
        source = json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text())
        adaptive = np.zeros(len(TAUS)); fixed = np.zeros(len(CONSTANTS)); count = 0
        for directory in source['train_benchmarks']:
            for row in episodes(directory,engine,source['semantic_ridge'],source['metadata_ridge'],seeds=(73,),pair_cap=25):
                for j,n in enumerate(BUDGETS):
                    a,b,p,h = components(row,n); d = p-b
                    for i,tau in enumerate(TAUS):
                        c = coefficient(h,a,tau)
                        adaptive[i] += ALC_WEIGHTS[j]*weighted_brier(np.clip(b+c*d,0,1),row['counts'],row['successes'])
                    for i,c in enumerate(CONSTANTS):
                        fixed[i] += ALC_WEIGHTS[j]*weighted_brier(np.clip(b+c*d,0,1),row['counts'],row['successes'])
                count += 1
        selection = {**source,'source_seed':73,'max_pairs_per_source_benchmark':25,'source_pairs':count,
                     'taus':TAUS,'constant_scales':CONSTANTS,'adaptive_source_alc':(adaptive/count).tolist(),
                     'fixed_source_alc':(fixed/count).tolist(),'tau':TAUS[int(np.argmin(adaptive))],
                     'fixed_scale':CONSTANTS[int(np.argmin(fixed))]}
        selections.append(selection); dest.write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('selected',fold,'tau',selection['tau'],'fixed',selection['fixed_scale'],'source_pairs',count,flush=True)
    for fold,selection in enumerate(selections):
        reference = json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text())
        total = {k:np.zeros(6) for k in NAMES}; by = {}; count = 0
        for directory,old in reference['by_benchmark'].items():
            target = root/(directory.split('/')[0]+'_evaluation.json')
            if target.exists(): current = json.loads(target.read_text())
            else:
                sums = {k:np.zeros(6) for k in NAMES}; nrows = 0; scales = [[] for _ in BUDGETS]
                seed_sums = {seed:{k:np.zeros(6) for k in NAMES} for seed in [0,1,2]}
                with (root/(directory.split('/')[0]+'_pair_brier.csv')).open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.writer(stream); writer.writerow(['subject_id','seed','budget',*NAMES,'scale'])
                    for row in episodes(directory,engine,selection['semantic_ridge'],selection['metadata_ridge']):
                        for j,n in enumerate(BUDGETS):
                            a,b,p,h = components(row,n); d = p-b; m,s = row['counts'],row['successes']
                            c = coefficient(h,a,selection['tau']); scales[j].append(c)
                            values = {
                                'adaptive_scale':weighted_brier(np.clip(b+c*d,0,1),m,s),
                                'fixed_scale':weighted_brier(np.clip(b+selection['fixed_scale']*d,0,1),m,s),
                                'R25':weighted_brier(p,m,s),'smoothed_mean_2':weighted_brier(b,m,s)}
                            for name,alpha in [('training_selected_smoothed',[2,2,4][fold]),('empirical_mean',0)]:
                                q = (a.sum()+alpha*.5)/(n+alpha) if n+alpha else .5
                                values[name] = weighted_brier(q,m,s)
                            if n<3 or selection['tau'] is None: values['scale_permuted']=values['R25']
                            else:
                                rng=np.random.default_rng(_stable_seed(3404,directory,f'{row["subject_id"]}:{row["seed"]}:{n}'))
                                control=[]
                                for _ in range(4):
                                    shuffled_c=coefficient(h,a[rng.permutation(n)],selection['tau'])
                                    control.append(weighted_brier(np.clip(b+shuffled_c*d,0,1),m,s))
                                values['scale_permuted']=float(np.mean(control))
                            for k,v in values.items(): sums[k][j]+=v; seed_sums[row['seed']][k][j]+=v
                            writer.writerow([row['subject_id'],row['seed'],n,*[values[k] for k in NAMES],c])
                        nrows += 1
                pairs=nrows//3; assert nrows==3*old['pairs_per_seed'][0]
                avg={k:v/nrows for k,v in sums.items()}
                for k in ['R25','smoothed_mean_2','training_selected_smoothed','empirical_mean']:
                    np.testing.assert_allclose(avg[k],old['brier_mean'][k],atol=1e-7,rtol=0)
                current={'pairs_per_seed':[pairs]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},
                         'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},
                         'alc_per_seed':{seed:{k:float(v/pairs@ALC_WEIGHTS) for k,v in sums.items()} for seed,sums in seed_sums.items()},
                         'scale_mean':[float(np.mean(v)) for v in scales],
                         'scale_above1_fraction':[float(np.mean(np.asarray(v)>1.00001)) for v in scales]}
                target.write_text(json.dumps(current,indent=2),encoding='utf-8')
            by[directory]=current; pairs=current['pairs_per_seed'][0]; count+=pairs
            for k in NAMES: total[k]+=pairs*np.asarray(current['brier_mean'][k])
            print('outer',fold,directory,'pairs',pairs,flush=True)
        result={'selection':selection,'pairs_per_seed':[count]*3,'by_benchmark':by,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    summary=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],NAMES[0],NAMES[1:])
    summary.update(elapsed_seconds=time.time()-start,scope='all5242pairs,three original seeds,64queryitems; reused public benchmarks; source-only scale choices')
    summary['gate_checks']={k:{'gain_at_least_005':-v['pair_equal_delta']>=.005,
        'all_fold_directions':all(x<0 for x in v['fold_deltas']),
        'benchmark_equal_improves':summary['benchmark_equal_alc']['adaptive_scale']<summary['benchmark_equal_alc'][k],
        'cluster_ci_below_zero':v['cluster_bootstrap_ci95'][1]<0,
        'no_large_regression':v['worst_benchmark_delta']<.03} for k,v in summary['comparisons'].items() if k in ['smoothed_mean_2','training_selected_smoothed']}
    summary['numeric_gate_passes']=all(all(v.values()) for v in summary['gate_checks'].values())
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for p in sources: assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps({k:summary[k] for k in ['pair_equal_alc','numeric_gate_passes','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__': main()
