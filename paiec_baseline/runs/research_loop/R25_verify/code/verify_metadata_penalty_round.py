"""R25 fixed-coefficient three-trajectory verification."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .metadata_episodes import episodes
from .cohort_semantic import CachedSemantic
from .run_metadata_penalty_round import score
from .evaluate import ALC_WEIGHTS,_stable_seed
from .summarize_candidate import summarize


def main():
    root=Path('runs/research_loop/R25_verify');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic();start=time.time()
    names=['separate','metadata_permuted','semantic','smoothed_mean_2','training_selected_smoothed']
    for fold in range(3):
        selection=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text());sem=selection['semantic_ridge'];meta=selection['metadata_ridge']
        old=json.loads(Path(f'runs/research_loop/R7_verify/fold{fold}_evaluation.json').read_text())
        initial=json.loads(Path(f'runs/research_loop/R25/fold{fold}_evaluation.json').read_text())
        total={k:np.zeros(6) for k in names};count=0;by={}
        for directory in selection['heldout_benchmarks']:
            collected={k:[] for k in names[:3]};per_seed={}
            for seed in [0,1,2]:
                rows=episodes(directory,seed,100,engine)
                if not rows:continue
                rng=np.random.default_rng(_stable_seed(119,directory,'separate_metadata'))
                current={'separate':np.mean([score(r,sem,meta) for r in rows],axis=0),
                         'metadata_permuted':np.mean([score(r,sem,meta,8,rng) for r in rows],axis=0),
                         'semantic':np.mean([score(r,sem,None) for r in rows],axis=0)}
                if seed==0:
                    for k,v in current.items():np.testing.assert_allclose(v,initial['by_benchmark'][directory]['brier_mean'][k],atol=1e-7,rtol=0)
                per_seed[str(seed)]={k:float(v@ALC_WEIGHTS) for k,v in current.items()}
                for k,v in current.items():collected[k].append(v)
            if not collected['separate']:continue
            n=len(rows);count+=n;avg={k:np.mean(v,axis=0) for k,v in collected.items()};historic=old['by_benchmark'][directory]
            np.testing.assert_allclose(avg['semantic'],historic['brier_mean']['semantic'],atol=1e-7,rtol=0)
            for k in names[3:]:avg[k]=np.asarray(historic['brier_mean'][k])
            by[directory]={'pairs_per_seed':[n]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},'alc_per_seed':per_seed}
            for k,v in avg.items():total[k]+=n*v
            print(directory,flush=True)
        result={'selection':selection,'seeds':[0,1,2],'pairs_per_seed':[count]*3,'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'separate',names[1:]);d['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/verify_metadata_penalty_round.py','mlp/run_metadata_penalty_round.py','research_loop/R25_verify_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
