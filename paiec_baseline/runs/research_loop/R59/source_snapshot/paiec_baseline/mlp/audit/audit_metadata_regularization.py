"""R25 exact attribution of R24 representation and common-ridge changes."""
import json
from pathlib import Path
import time
import numpy as np
from mlp.cohort_semantic import CachedSemantic
from mlp.metadata_episodes import episodes
from mlp.run.run_metadata_round import score
from mlp.evaluate import ALC_WEIGHTS
from mlp.reporting.summarize_candidate import summarize


def main():
    root=Path('runs/research_loop/R25');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic();start=time.time()
    names=['combined','same_ridge_semantic','original_semantic']
    for fold in range(3):
        selection=json.loads(Path(f'runs/research_loop/R24/fold{fold}_selection.json').read_text());old=json.loads(Path(f'runs/research_loop/R24/fold{fold}_evaluation.json').read_text())
        total={k:np.zeros(6) for k in names};count=0;by={}
        for directory in selection['heldout_benchmarks']:
            rows=episodes(directory,0,100,engine)
            if not rows:continue
            n=len(rows);count+=n;historical=old['by_benchmark'][directory]
            same=np.mean([score(r,'semantic',selection['selected_ridge']['combined']) for r in rows],axis=0)
            avg={'same_ridge_semantic':same,'combined':np.asarray(historical['brier_mean']['combined']),
                 'original_semantic':np.asarray(historical['brier_mean']['semantic'])}
            by[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},
                'pairs_with_constant_support_metadata':int(sum(np.max(np.abs(r['support_meta']-r['support_meta'][0]))<1e-12 for r in rows))}
            for k,v in avg.items():total[k]+=n*v
        result={'pairs_per_seed':[count],'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_attribution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_attribution.json' for i in range(3)],'combined',names[1:]);a=d['pair_equal_alc']
    d['components']={'added_field_delta':a['combined']-a['same_ridge_semantic'],
                     'regularization_delta':a['same_ridge_semantic']-a['original_semantic'],
                     'net_delta':a['combined']-a['original_semantic']};d['elapsed_seconds']=time.time()-start
    (root/'attribution_summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    print(json.dumps(d['components'],indent=2))


if __name__=='__main__':main()
