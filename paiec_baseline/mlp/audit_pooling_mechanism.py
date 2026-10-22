"""R12 matched prior-strength control for interpreting R11."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .run_shared_round import records
from .run_hierarchical_round import score,SOLVERS
from .evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from .summarize_candidate import summarize


def plugin(rows,directory,alpha,kappa=8):
    order=np.arange(len(rows));np.random.default_rng(_stable_seed(193,directory,'cohort')).shuffle(order)
    values=[]
    for offset in range(0,len(order),4):
        cohort=[rows[i] for i in order[offset:offset+4]]
        for target in cohort:
            b=[]
            for n in BUDGETS:
                own_s=own_n=peer_s=peer_n=0
                for row in cohort:
                    if row['key']==target['key']:own_s+=sum(row['a'][:n]);own_n+=n
                    else:peer_s+=sum(row['a'][:n]);peer_n+=n
                prior=(kappa*.5+peer_s)/(kappa+peer_n)
                p=(alpha*prior+own_s)/(alpha+own_n) if peer_n else (1+own_s)/(2+own_n)
                q=target['evaluation_mean'];b.append(p*p-2*p*q+q)
            values.append(b)
    return np.array(values)


def main():
    root=Path('runs/research_loop/R12');root.mkdir(parents=True,exist_ok=True)
    selections=[];alphas=[1.,2.,4.,8.,16.]
    for fold in range(3):
        sel=json.loads(Path(f'runs/research_loop/R11/fold{fold}_selection.json').read_text());sums=np.zeros(5);count=0
        kappa=int(sel['hierarchy_selected']['fixed'].split('_')[1])
        for directory in sel['train_benchmarks']:
            rows=records(directory,73,25)
            if not rows:continue
            sums+=[float(np.sum(plugin(rows,directory,a,kappa)@ALC_WEIGHTS)) for a in alphas];count+=len(rows)
        sel.update(plugin_alpha_grid=alphas,plugin_training_alc=(sums/count).tolist(),plugin_alpha=alphas[int(np.argmin(sums))],plugin_kappa=kappa)
        (root/f'fold{fold}_selection.json').write_text(json.dumps(sel,indent=2),encoding='utf-8');selections.append(sel)
    for fold,sel in enumerate(selections):
        names=['fixed_hierarchy','adaptive_hierarchy','plugin_matched4','plugin_selected','plugin2','smoothed_mean_2'];totals={k:np.zeros(6) for k in names}
        old=json.loads(Path(f'runs/research_loop/R11_verify/fold{fold}_evaluation.json').read_text());by_benchmark={};count=0
        for directory in sel['heldout_benchmarks']:
            avg={k:[] for k in names}
            for seed in [0,1,2]:
                rows=records(directory,seed,100)
                if not rows:continue
                for name,alpha in [('plugin_matched4',4.),('plugin_selected',sel['plugin_alpha']),('plugin2',2.)]:
                    avg[name].append(plugin(rows,directory,alpha,sel['plugin_kappa']).mean(0))
            if not avg['plugin2']:continue
            pairs=len(rows);count+=pairs
            for dest,source in [('fixed_hierarchy','fixed'),('adaptive_hierarchy','adaptive'),('smoothed_mean_2','smoothed_mean_2')]:avg[dest]=[old['by_benchmark'][directory]['brier_mean'][source]]
            avg={k:np.mean(v,axis=0) for k,v in avg.items()}
            np.testing.assert_allclose(avg['plugin2'],old['by_benchmark'][directory]['brier_mean']['shared_prior'],atol=1e-12)
            by_benchmark[directory]={'pairs_per_seed':[pairs]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():totals[k]+=pairs*v
        output={'selection':sel,'seeds':[0,1,2],'pairs_per_seed':[count]*3,'by_benchmark':by_benchmark,
            'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    summary=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'fixed_hierarchy',names[1:])
    summary['scope']='Matched-strength attribution audit; conditional shared-evidence cohort4, reused public folds'
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['pair_equal_alc','comparisons']},indent=2))


if __name__=='__main__':main()
