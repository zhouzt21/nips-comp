"""R27: parameter-free sqrt-budget shrinkage, not a PAIEC minimax theorem."""
import hashlib
import json
from math import comb, sqrt
from pathlib import Path
import time
import numpy as np
from .evaluate import BUDGETS, ALC_WEIGHTS
from .run_shared_round import records
from .summarize_candidate import summarize


def estimate(successes, n):
    return (successes + sqrt(n)/2)/(n + sqrt(n)) if n else .5


def verify_risk_identity():
    errors=[]
    for n in BUDGETS[1:]:
        for q in np.linspace(0,1,21):
            risk=sum(comb(n,s)*q**s*(1-q)**(n-s)*(estimate(s,n)-q)**2 for s in range(n+1))
            errors.append(abs(risk-1/(4*(sqrt(n)+1)**2)))
    assert max(errors)<1e-12
    return {'max_parameter_risk_identity_error':max(errors),
            'assumption':'iid Bernoulli parameter squared error, not full Brier minimax'}


def main():
    root=Path('runs/research_loop/R27');root.mkdir(parents=True,exist_ok=True);start=time.time()
    check=verify_risk_identity();names=['budget_mean','empirical_mean','smoothed_mean_2','training_selected_smoothed','R25']
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text())
        old=json.loads(Path(f'runs/research_loop/R25_verify/fold{fold}_evaluation.json').read_text())
        total={k:np.zeros(6) for k in names};by={};count=0
        for directory in split['development_benchmarks']:
            collected={k:[] for k in names[:3]};per_seed={};counts=[]
            for seed in [0,1,2]:
                rows=records(directory,seed,100)
                if not rows:continue
                values={k:[] for k in collected}
                for row in rows:
                    q=row['evaluation_mean'];cur={k:[] for k in collected}
                    for n in BUDGETS:
                        s=sum(row['a'][:n]);ps=[estimate(s,n),s/n if n else .5,(1+s)/(2+n)]
                        for k,p in zip(cur,ps):cur[k].append(p*p-2*p*q+q)
                    for k,v in cur.items():values[k].append(v)
                per_seed[str(seed)]={k:float(np.mean(v,axis=0)@ALC_WEIGHTS) for k,v in values.items()}
                for k,v in values.items():collected[k].append(np.mean(v,axis=0))
                counts.append(len(rows))
            if not counts:continue
            assert len(set(counts))==1 and len(counts)==3
            avg={k:np.mean(v,axis=0) for k,v in collected.items()};historic=old['by_benchmark'][directory]
            np.testing.assert_allclose(avg['smoothed_mean_2'],historic['brier_mean']['smoothed_mean_2'],atol=1e-7,rtol=0)
            avg['training_selected_smoothed']=np.asarray(historic['brier_mean']['training_selected_smoothed'])
            avg['R25']=np.asarray(historic['brier_mean']['separate'])
            n=counts[0];count+=n
            by[directory]={'pairs_per_seed':counts,'brier_mean':{k:v.tolist() for k,v in avg.items()},
                           'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},'alc_per_seed':per_seed}
            for k,v in avg.items():total[k]+=n*v
        d={'seeds':[0,1,2],'pairs_per_seed':[count]*3,'by_benchmark':by,
           'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
        print(f'fold {fold} complete',flush=True)
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'budget_mean',names[1:])
    d.update(elapsed_seconds=time.time()-start,mathematical_check=check)
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/run_budget_mean_round.py','mlp/run_shared_round.py','mlp/summarize_candidate.py','research_loop/R27_protocol.md']
    snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
