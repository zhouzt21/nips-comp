"""R14 fixed predictors under alternative allowed peer assignments."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .cohort_semantic import CachedSemantic,episodes
from .run_component_round import score,NAMES
from .evaluate import ALC_WEIGHTS
from .summarize_candidate import summarize


def main():
    root=Path('runs/research_loop/R14');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic();start=time.time()
    fold_payloads={c:[] for c in [193,271,811]}
    for fold in range(3):
        old=json.loads(Path(f'runs/research_loop/R13/cohort4/fold{fold}_evaluation.json').read_text())
        sel=old['selection'];ridge=sel['local']['selected_ridge']['semantic'];alpha=sel['peer']['plugin_alpha'];kappa=sel['peer']['plugin_kappa']
        by_benchmark={c:{} for c in fold_payloads};totals={c:{k:np.zeros(6) for k in NAMES} for c in fold_payloads};count=0
        for directory in sel['local']['heldout_benchmarks']:
            data=[episodes(directory,seed,engine) for seed in [0,1,2]]
            if not data[0]:continue
            n=len(data[0]);count+=n
            for cohort_seed in fold_payloads:
                scores=[score(rows,directory,4,ridge,alpha,kappa,cohort_seed) for rows in data]
                avg={k:np.mean([s[k].mean(0) for s in scores],axis=0) for k in NAMES}
                if cohort_seed==193:
                    for k in NAMES:np.testing.assert_allclose(avg[k],old['by_benchmark'][directory]['brier_mean'][k],atol=1e-12)
                by_benchmark[cohort_seed][directory]={'pairs_per_seed':[n]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
                for k,v in avg.items():totals[cohort_seed][k]+=n*v
            print(f'fold{fold} {directory}',flush=True)
        for c in fold_payloads:
            folder=root/f'cohort_seed{c}';folder.mkdir(exist_ok=True)
            payload={'cohort_seed':c,'item_seeds':[0,1,2],'pairs_per_seed':[count]*3,'by_benchmark':by_benchmark[c],
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in totals[c].items()}}
            (folder/f'fold{fold}_evaluation.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');fold_payloads[c].append(payload)
        avg_bench={}
        for directory in by_benchmark[193]:
            n=by_benchmark[193][directory]['pairs_per_seed'][0]
            avg={k:np.mean([by_benchmark[c][directory]['brier_mean'][k] for c in fold_payloads],axis=0) for k in NAMES}
            avg_bench[directory]={'pairs_per_seed':[n]*9,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
        avg={k:np.mean([totals[c][k]/count for c in fold_payloads],axis=0) for k in NAMES}
        output={'pairs_per_seed':[count]*9,'by_benchmark':avg_bench,'results':{k:{'brier_mean':v.tolist(),'alc_mean':float(v@ALC_WEIGHTS)} for k,v in avg.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for c in fold_payloads:
        folder=root/f'cohort_seed{c}';s=summarize([folder/f'fold{i}_evaluation.json' for i in range(3)],'combined',NAMES[1:])
        (folder/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    s=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'combined',NAMES[1:])
    s.update(scope='Conditional cohort4 across three peer assignments and three item trajectories; reused public benchmarks',
        cohort_seeds=list(fold_payloads),item_seeds=[0,1,2],bootstrap_unit='benchmark after averaging all9draws',elapsed_seconds=time.time()-start)
    (root/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    sources=['mlp/stress_cohort_composition.py','mlp/run_component_round.py','mlp/cohort_semantic.py','research_loop/R14_protocol.md']
    snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps({k:s[k] for k in ['pair_equal_alc','comparisons']},indent=2))


if __name__=='__main__':main()
