"""R20 three-seed verification without parameter reselection."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .run_pooled_slope_round import score
from .cohort_semantic import CachedSemantic,episodes
from .evaluate import ALC_WEIGHTS
from .summarize_candidate import summarize


def main():
    root=Path('runs/research_loop/R20_verify');root.mkdir(parents=True,exist_ok=True);start=time.time();engine=CachedSemantic()
    names=['pooled','own_selected','own_original','peer','base','peer_permuted']
    for fold in range(3):
        selection=json.loads(Path(f'runs/research_loop/R20/fold{fold}_selection.json').read_text())
        original=json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text())
        old=json.loads(Path(f'runs/research_loop/R20/fold{fold}_evaluation.json').read_text())
        totals={k:np.zeros(6) for k in names};count=0;by={}
        for directory in selection['heldout_benchmarks']:
            collected={k:[] for k in names};per_seed={}
            for seed in [0,1,2]:
                rows=episodes(directory,seed,engine)
                if not rows:continue
                out=score(rows,directory,selection['selected_ridge']['pooled'],selection['selected_ridge']['own_selected'],original['selected_ridge']['semantic'],8)
                if seed==0:
                    for k,v in out.items():np.testing.assert_allclose(v.mean(0),old['by_benchmark'][directory]['brier_mean'][k],atol=1e-12,rtol=0)
                per_seed[str(seed)]={k:float(v.mean(0)@ALC_WEIGHTS) for k,v in out.items()}
                for k,v in out.items():collected[k].append(v.mean(0))
            if not collected['base']:continue
            n=len(rows);count+=n;avg={k:np.mean(v,axis=0) for k,v in collected.items()}
            by[directory]={'pairs_per_seed':[n]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},'alc_per_seed':per_seed}
            for k,v in avg.items():totals[k]+=n*v
            print(directory,flush=True)
        result={'selection':selection,'seeds':[0,1,2],'pairs_per_seed':[count]*3,'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'pooled',names[1:]);d['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/verify_pooled_slope_round.py','mlp/run_pooled_slope_round.py','mlp/pooled_item_slope.py','research_loop/R20_verify_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
