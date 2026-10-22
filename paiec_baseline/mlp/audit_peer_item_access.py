"""R19a uses input matches only, never outcomes, to measure evidence access."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .visible_item_evidence import episodes,cohorts
from .evaluate import BUDGETS


def main():
    root=Path('runs/research_loop/R19');root.mkdir(parents=True,exist_ok=True);start=time.time();by={};all_rows=[]
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text())
        for directory in split['development_benchmarks']:
            rows=episodes(directory,0);coverage=[]
            for cohort in cohorts(rows,directory):
                for target in cohort:
                    values=[]
                    for n in BUDGETS:
                        peer_keys={k for r in cohort if r['key']!=target['key'] for k in r['support_keys'][:n]}
                        values.append(np.mean([k in peer_keys for k in target['query_keys']]))
                    coverage.append(values)
            if coverage:
                by[directory]={'pairs':len(rows),'coverage':np.mean(coverage,axis=0).tolist()};all_rows.extend(coverage)
            print(directory,len(rows),flush=True)
    output={'scope':'input-only peer match coverage; conditional cohort4; original seeds/splits','budgets':list(BUDGETS),
            'pairs':len(all_rows),'pair_equal_coverage':np.mean(all_rows,axis=0).tolist(),'by_benchmark':by,'elapsed_seconds':time.time()-start}
    (root/'access_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    sources=['mlp/visible_item_evidence.py','mlp/audit_peer_item_access.py','research_loop/R19_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'access_provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print('coverage',output['pair_equal_coverage'],flush=True)


if __name__=='__main__':main()
