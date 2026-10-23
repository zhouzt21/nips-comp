"""Freeze R11 choices; verify shared-evidence trajectories without retuning."""
import json
from pathlib import Path
import time
import numpy as np
from mlp.run.run_shared_round import records,score_rows
from mlp.run.run_hierarchical_round import score,SOLVERS
from mlp.evaluate import ALC_WEIGHTS
from mlp.reporting.summarize_candidate import summarize


def main():
    root=Path('runs/research_loop/R11_verify');root.mkdir(parents=True,exist_ok=True);start=time.time()
    names=['adaptive','fixed','shared_prior','smoothed_mean_2','selected_smoothed','empirical_mean']
    for fold in range(3):
        sel=json.loads(Path(f'runs/research_loop/R11/fold{fold}_selection.json').read_text())
        previous=json.loads(Path(f'runs/research_loop/R11/cohort4/fold{fold}_evaluation.json').read_text())
        counts=0;by_benchmark={};totals={k:np.zeros(6) for k in names}
        for directory in sel['heldout_benchmarks']:
            collected={k:[] for k in names};per_seed={}
            for seed in [0,1,2]:
                rows=records(directory,seed,100)
                if not rows:continue
                out=score_rows(rows,directory,4,sel['selected_kappa'],sel['selected_alpha'])
                for mode in ['adaptive','fixed']:out[mode]=score(rows,directory,4,SOLVERS[sel['hierarchy_selected'][mode]])[0]
                per_seed[str(seed)]={k:float(v.mean(0)@ALC_WEIGHTS) for k,v in out.items()}
                for k,v in out.items():
                    collected[k].append(v.mean(0))
                    if seed==0:np.testing.assert_allclose(v.mean(0),previous['by_benchmark'][directory]['brier_mean'][k],atol=1e-12)
            if not collected['adaptive']:continue
            count=len(rows);counts+=count;avg={k:np.mean(v,axis=0) for k,v in collected.items()}
            by_benchmark[directory]={'pairs_per_seed':[count]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},
                'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},'alc_per_seed':per_seed}
            for k,v in avg.items():totals[k]+=count*v
            print(f'fold{fold} {directory}',flush=True)
        output={'selection':sel,'seeds':[0,1,2],'pairs_per_seed':[counts]*3,'by_benchmark':by_benchmark,
            'results':{k:{'brier_mean':(v/counts).tolist(),'alc_mean':float(v/counts@ALC_WEIGHTS)} for k,v in totals.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    summary=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'adaptive',names[1:])
    summary.update(scope='Conditional same-benchmark cohort4; reused public benchmarks; not isolated-pair gate',
        bootstrap_unit='benchmark after seed averaging',elapsed_seconds=time.time()-start,seeds=[0,1,2])
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['pair_equal_alc','comparisons']},indent=2))


if __name__=='__main__':main()
