"""R11: matched conditional pooling, fixed versus uncertain heterogeneity."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from .run_shared_round import records,score_rows
from .summarize_candidate import summarize
from shared_mean.hierarchical import HierarchicalCounts

ROOT=Path('runs/research_loop/R11')
SOLVERS={f'{mode}_{k}':HierarchicalCounts(k,(4.,) if mode=='fixed' else (1.,4.,16.))
         for mode in ['fixed','adaptive'] for k in [2,8]}


def score(rows,directory,size,solver):
    order=np.arange(len(rows));np.random.default_rng(_stable_seed(193,directory,'cohort')).shuffle(order)
    scores=[];concentrations=[]
    for offset in range(0,len(order),size):
        cohort=[rows[i] for i in order[offset:offset+size]]
        keys=list(dict.fromkeys(r['key'] for r in cohort));positions={k:i for i,k in enumerate(keys)}
        result=np.empty((len(cohort),6));diag=[]
        for j,n in enumerate(BUDGETS):
            counts=[[0,0] for k in keys]
            for r in cohort:
                v=counts[positions[r['key']]];v[0]+=sum(r['a'][:n]);v[1]+=n
            p,c=solver.predict_counts(counts);diag.append(c)
            for i,r in enumerate(cohort):
                pred=p[positions[r['key']]];q=r['evaluation_mean'];result[i,j]=pred*pred-2*pred*q+q
        scores.extend(result);concentrations.extend([diag]*len(cohort))
    return np.array(scores),np.array(concentrations)


def main():
    ROOT.mkdir(parents=True,exist_ok=True);start=time.time();selections=[]
    sources=['mlp/run_hierarchical_round.py','shared_mean/hierarchical.py','mlp/run_shared_round.py','mlp/summarize_candidate.py','research_loop/R11_protocol.md']
    snap=ROOT/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        prior=json.loads(Path(f'runs/research_loop/R9/fold{fold}_selection.json').read_text())
        sums={k:0. for k in SOLVERS};count=0
        for directory in prior['train_benchmarks']:
            rows=records(directory,73,25)
            if not rows:continue
            for name,solver in SOLVERS.items():sums[name]+=float(np.sum(score(rows,directory,4,solver)[0]@ALC_WEIGHTS))
            count+=len(rows)
        selected={mode:min([k for k in SOLVERS if k.startswith(mode)],key=sums.get) for mode in ['fixed','adaptive']}
        sel={**prior,'hierarchy_selected':selected,'hierarchy_training_alc':{k:v/count for k,v in sums.items()}}
        (ROOT/f'fold{fold}_selection.json').write_text(json.dumps(sel,indent=2),encoding='utf-8');selections.append(sel)
        print(f'fold{fold}: {selected}',flush=True)
    names=['adaptive','fixed','shared_prior','smoothed_mean_2','selected_smoothed','empirical_mean']
    for size in [1,2,4,6]:
        folder=ROOT/f'cohort{size}';folder.mkdir(exist_ok=True)
        for fold,sel in enumerate(selections):
            prior=json.loads(Path(f'runs/research_loop/R9/cohort{size}/fold{fold}_evaluation.json').read_text())
            by_benchmark={};total={k:np.zeros(6) for k in names};count=0;concentration={}
            for directory in sel['heldout_benchmarks']:
                rows=records(directory,0,100)
                if not rows:continue
                out=score_rows(rows,directory,size,sel['selected_kappa'],sel['selected_alpha'])
                concentration[directory]={}
                for mode in ['fixed','adaptive']:
                    out[mode],diag=score(rows,directory,size,SOLVERS[sel['hierarchy_selected'][mode]])
                    concentration[directory][mode]=[float(np.mean(col[np.isfinite(col)])) if np.isfinite(col).any() else None for col in diag.T]
                for name in names[2:]:np.testing.assert_allclose(out[name].mean(0),prior['by_benchmark'][directory]['brier_mean'][name],atol=1e-12)
                if size==1:
                    for name in ['fixed','adaptive']:np.testing.assert_allclose(out[name],out['smoothed_mean_2'],atol=1e-12)
                by_benchmark[directory]={'pairs_per_seed':[len(rows)],'brier_mean':{k:v.mean(0).tolist() for k,v in out.items()},
                                         'alc_mean':{k:float(v.mean(0)@ALC_WEIGHTS) for k,v in out.items()}}
                count+=len(rows)
                for k,v in out.items():total[k]+=v.sum(0)
            output={'selection':sel,'cohort_size':size,'pairs_per_seed':[count],'by_benchmark':by_benchmark,
                    'posterior_concentration':concentration,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
            (folder/f'fold{fold}_evaluation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
        summary=summarize([folder/f'fold{i}_evaluation.json' for i in range(3)],'adaptive',names[1:])
        summary['scope']='Conditional shared-evidence scenario, organizer cohorts unverified; not the isolated-pair promotion gate'
        (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(f'cohort{size}: {summary["pair_equal_alc"]}',flush=True)
    provenance={'elapsed_seconds':time.time()-start,'source_revision':'54074453da7b9911be0f48e4c560be547bc84d21',
                'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}}
    (ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__=='__main__':main()
