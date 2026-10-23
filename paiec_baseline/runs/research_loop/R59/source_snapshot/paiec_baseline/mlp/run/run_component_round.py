"""R13 factorial: paired local discrimination and peer prevalence estimates."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.cohort_semantic import CachedSemantic,episodes
from mlp.evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from mlp.reporting.summarize_candidate import summarize

NAMES=['combined','local','peer','base']


def score(rows,directory,size,ridge,alpha,kappa,cohort_seed=193):
    order=np.arange(len(rows));np.random.default_rng(_stable_seed(cohort_seed,directory,'cohort')).shuffle(order)
    outputs={k:[] for k in NAMES}
    for offset in range(0,len(order),size):
        cohort=[rows[i] for i in order[offset:offset+size]]
        values={k:np.empty((len(cohort),6)) for k in NAMES}
        for j,n in enumerate(BUDGETS):
            total_s=sum(r['a'][:n].sum() for r in cohort);total_n=n*len(cohort)
            groups={}
            for r in cohort:groups.setdefault(r['key'],[]).append(r)
            state={}
            for key,group in groups.items():
                labels=np.concatenate([r['a'][:n] for r in group]);own_n=len(labels);own_s=labels.sum()
                b=(1+own_s)/(2+own_n);peer_n=total_n-own_n
                mean=(alpha*(kappa*.5+total_s-own_s)/(kappa+peer_n)+own_s)/(alpha+own_n) if peer_n else b
                if own_n<2 or ridge is None:state[key]=(b,mean,None,None)
                else:
                    a=np.concatenate([r['support'][:n] for r in group]);center=a.mean(0);ac=a-center
                    coef=np.linalg.solve(ac@ac.T+ridge*np.eye(own_n),labels-labels.mean())
                    state[key]=(b,mean,center,ac.T@coef)
            for i,r in enumerate(cohort):
                b,mean,center,slope=state[r['key']]
                correction=0 if slope is None else (r['query']-center)@slope
                predictions={'base':b,'peer':mean,'local':np.clip(b+correction,0,1),'combined':np.clip(mean+correction,0,1)}
                for name,p in predictions.items():values[name][i,j]=np.mean((p-r['y'])**2)
        for k,v in values.items():outputs[k].extend(v)
    return {k:np.asarray(v) for k,v in outputs.items()}


def main():
    root=Path('runs/research_loop/R13');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic();start=time.time()
    sources=['mlp/run/run_component_round.py','mlp/cohort_semantic.py','mlp/reporting/summarize_candidate.py','../research_loop/R13_protocol.md']
    snapshot=root/'code';snapshot.mkdir(exist_ok=True)
    for source in sources:(snapshot/Path(source).name).write_bytes(Path(source).read_bytes())
    for size in [1,4,6]:
        folder=root/f'cohort{size}';folder.mkdir(exist_ok=True)
        for fold in range(3):
            local=json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text())
            peer=json.loads(Path(f'runs/research_loop/R12/fold{fold}_selection.json').read_text())
            old_local=json.loads(Path(f'runs/research_loop/R7_verify/fold{fold}_evaluation.json').read_text())
            old_peer=json.loads(Path(f'runs/research_loop/R12/fold{fold}_evaluation.json').read_text())
            totals={k:np.zeros(6) for k in NAMES};count=0;by_benchmark={}
            for directory in local['heldout_benchmarks']:
                all_seeds={k:[] for k in NAMES};per_seed={}
                for seed in [0,1,2]:
                    rows=episodes(directory,seed,engine)
                    if not rows:continue
                    out=score(rows,directory,size,local['selected_ridge']['semantic'],peer['plugin_alpha'],peer['plugin_kappa'])
                    per_seed[str(seed)]={k:float(v.mean(0)@ALC_WEIGHTS) for k,v in out.items()}
                    for k,v in out.items():all_seeds[k].append(v.mean(0))
                if not all_seeds['base']:continue
                n=len(rows);avg={k:np.mean(v,axis=0) for k,v in all_seeds.items()};count+=n
                if size==1:
                    np.testing.assert_allclose(avg['local'],old_local['by_benchmark'][directory]['brier_mean']['semantic'],atol=1e-7,rtol=0)
                    np.testing.assert_allclose(avg['peer'],avg['base'],atol=1e-12)
                if size==4:np.testing.assert_allclose(avg['peer'],old_peer['by_benchmark'][directory]['brier_mean']['plugin_selected'],atol=1e-12,rtol=0)
                by_benchmark[directory]={'pairs_per_seed':[n]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},
                    'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},'alc_per_seed':per_seed}
                for k,v in avg.items():totals[k]+=n*v
                print(f'cohort{size} fold{fold} {directory}',flush=True)
            output={'seeds':[0,1,2],'pairs_per_seed':[count]*3,'selection':{'local':local,'peer':peer},'by_benchmark':by_benchmark,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
            (folder/f'fold{fold}_evaluation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
        summary=summarize([folder/f'fold{i}_evaluation.json' for i in range(3)],'combined',NAMES[1:])
        a=summary['pair_equal_alc'];summary['interaction_alc']=a['combined']-a['local']-a['peer']+a['base']
        summary['scope']='Factorial diagnostic, conditional shared-evidence cohorts; cohort1 isolated replay; repeated public folds'
        (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(f'cohort{size} ALC {a}',flush=True)
    (root/'provenance.json').write_text(json.dumps({'elapsed_seconds':time.time()-start,
        'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},'model_fingerprint':engine.fingerprint},indent=2),encoding='utf-8')


if __name__=='__main__':main()
