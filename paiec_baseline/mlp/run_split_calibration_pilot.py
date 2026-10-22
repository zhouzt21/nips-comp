"""R49 source-only bounded support-role experiment."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .cohort_semantic import CachedSemantic
from .metadata_episodes import episodes
from .split_calibration import predictions
from .evaluate import BUDGETS,ALC_WEIGHTS

NAMES=['split_calibrated','split_fixed','permuted_calibration','R25','smoothed_mean_2']


def main():
    start=time.time();root=Path('runs/research_loop/R49');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic()
    sources=['mlp/run_split_calibration_pilot.py','mlp/split_calibration.py','mlp/oracle_scale.py','mlp/run_metadata_round.py','research_loop/R49_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    results=[]
    for fold in range(3):
        target=root/f'fold{fold}_source.json'
        if target.exists():results.append(json.loads(target.read_text()));continue
        sel=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text());total={k:np.zeros(6) for k in NAMES};by={};count=0
        for directory in sel['train_benchmarks']:
            rows=episodes(directory,73,25,engine)
            if not rows:continue
            sums={k:np.zeros(6) for k in NAMES};scales=[[] for _ in BUDGETS]
            for row in rows:
                s=np.c_[row['support']/np.sqrt(sel['semantic_ridge']),row['support_meta']/np.sqrt(sel['metadata_ridge'])]
                q=np.c_[row['query']/np.sqrt(sel['semantic_ridge']),row['query_meta']/np.sqrt(sel['metadata_ridge'])]
                for j,n in enumerate(BUDGETS):
                    y=row['a'][:n];preds,c=predictions(s[:n],q,y);preds['smoothed_mean_2']=(1+y.sum())/(2+n)
                    for k,p in preds.items():sums[k][j]+=np.mean((p-row['y'])**2)
                    scales[j].extend(c)
            size=len(rows);count+=size
            by[directory]={'pairs':size,'brier':{k:(v/size).tolist() for k,v in sums.items()},'alc':{k:float(v/size@ALC_WEIGHTS) for k,v in sums.items()},
                           'coefficient_mean':[float(np.mean(v)) if v else None for v in scales]}
            for k,v in sums.items():total[k]+=v
        alc={k:float(v/count@ALC_WEIGHTS) for k,v in total.items()}
        old=json.loads(Path(f'runs/research_loop/R45/fold{fold}_source.json').read_text())
        assert abs(alc['R25']-old['pair_equal_alc']['semantic'])<1e-7
        result={'fold':fold,'scope':'source only','source_benchmarks':sel['train_benchmarks'],'pairs':count,'pair_equal_alc':alc,'pair_equal_brier':{k:(v/count).tolist() for k,v in total.items()},
                'benchmark_equal_alc':{k:float(np.mean([v['alc'][k] for v in by.values()])) for k in NAMES},'by_benchmark':by}
        target.write_text(json.dumps(result,indent=2),encoding='utf-8');results.append(result);print(fold,alc,flush=True)
    admit=all(r['pair_equal_alc']['split_calibrated']<min(r['pair_equal_alc'][k] for k in ['R25','split_fixed','permuted_calibration']) and r['benchmark_equal_alc']['split_calibrated']<r['benchmark_equal_alc']['R25'] for r in results)
    d={'scope':'source pilot only; no independent OOD estimate','admit_outer':admit,'folds':results,'elapsed_seconds':time.time()-start}
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    for p in sources:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print('admit_outer',admit,'seconds',d['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
