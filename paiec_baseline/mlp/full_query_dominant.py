"""R32c exact full-query census on the dominant benchmark, cached features only."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .audit_full_queries import evaluate_pair,NAMES
from .cohort_semantic import CachedSemantic
from .metadata_features import encode
from .evaluate import _load_directory,ALC_WEIGHTS


def main():
    start=time.time();root=Path('runs/research_loop/R32c');root.mkdir(parents=True,exist_ok=True);directory='alignment_faking/formatted_tables';fold=2
    response,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory);engine=CachedSemantic();vectors={};metas={}
    for i,item in items.items():vectors[i]=engine.vector(item['content']);metas[i]=encode(item['item_features'])
    selection=json.loads(Path('runs/research_loop/R25/fold2_selection.json').read_text())
    old=pd.read_csv('runs/research_loop/R31/alignment_faking_pair_brier.csv',dtype={'subject_id':str}).set_index(['subject_id','seed','budget']).sort_index()
    full={k:np.zeros(6) for k in NAMES};capped={k:np.zeros(6) for k in NAMES};n=0;noise=0;queries=0;responses=0;worst=0
    with (root/'pair_budget_scores.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['subject_id','budget',*[f'{scope}:{k}' for scope in ['full','capped'] for k in NAMES]])
        for sid,g in response.groupby('subject_id',sort=False):
            if g.item_id.nunique()<80 or sid not in subjects:continue
            runs=[]
            for seed in [0,1,2]:
                r=evaluate_pair(g,items,vectors,metas,sid,directory,seed,selection['semantic_ridge'],selection['metadata_ridge'],fold);runs.append(r)
                expected=old.loc[(str(sid),seed)]
                for k in NAMES:np.testing.assert_allclose(r['capped_brier'][k],expected[k].to_numpy(),atol=1e-7,rtol=0)
            f={k:np.mean([r['full_brier'][k] for r in runs],axis=0) for k in NAMES};c={k:np.mean([r['capped_brier'][k] for r in runs],axis=0) for k in NAMES}
            for k in NAMES:full[k]+=f[k];capped[k]+=c[k]
            for i,budget in enumerate([0,1,3,7,15,31]):writer.writerow([str(sid),budget,*[v[k][i] for v in [f,c] for k in NAMES]])
            n+=1;noise+=np.mean([r['empirical_within_item_variance'] for r in runs]);queries+=sum(r['full_items'] for r in runs);responses+=sum(r['full_responses'] for r in runs);worst=max(worst,max(r['identity_max_error'] for r in runs))
            if n%500==0:print('completed',n,'pairs',flush=True)
    assert n==4278
    d={'scope':'actual all4278pairs,all evaluation items/responses,three original seeds; one reused benchmark, not independent generalization',
       'pairs':n,'seeds':[0,1,2],'full_brier':{k:(v/n).tolist() for k,v in full.items()},'capped_brier':{k:(v/n).tolist() for k,v in capped.items()},
       'full_alc':{k:float(v/n@ALC_WEIGHTS) for k,v in full.items()},'capped_alc':{k:float(v/n@ALC_WEIGHTS) for k,v in capped.items()},
       'within_item_empirical_variance':noise/n,'total_pair_seed_query_items':queries,'total_pair_seed_responses':responses,'identity_max_error':worst,'elapsed_seconds':time.time()-start}
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/full_query_dominant.py','mlp/audit_full_queries.py','research_loop/R32c_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8');print(json.dumps(d,indent=2),flush=True)


if __name__=='__main__':main()
