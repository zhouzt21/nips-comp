"""Dense R7 source features with the exact source sampling, using only disk cache."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import _load_directory,_stable_seed
from .visible_item_evidence import plain
from .model import _SUBJECT_FIELDS


def episodes(directory,engine):
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+engine.fingerprint.encode()).hexdigest()
    path=Path('D:/paiec_research_cache/source_semantic')/(hashlib.sha256((directory+fingerprint).encode()).hexdigest()+'.npz')
    names=['a','y','support','query']
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['meta']));return [{**m,**{k:data[f'{i}_{k}'] for k in names}} for i,m in enumerate(meta)]
    responses,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory)
    groups=[(s,g) for s,g in responses.groupby('subject_id',sort=False) if g['item_id'].nunique()>=80 and s in subjects]
    if len(groups)>25:
        rng=np.random.default_rng(_stable_seed(2026,directory,'pair_sample'));groups=[groups[i] for i in np.sort(rng.choice(len(groups),25,replace=False))]
    out=[];vectors={}
    def vector(i):
        if i not in vectors:vectors[i]=engine.vector(items[i]['content'])
        return vectors[i]
    for sid,group in groups:
        ids=pd.unique(group['item_id']).copy();rng=np.random.default_rng(_stable_seed(73,directory,sid));rng.shuffle(ids)
        cut=len(ids)//2;acquired_ids=ids[:31];evaluation_ids=ids[cut:cut+64]
        acquired=group[group['item_id'].isin(acquired_ids)].sample(frac=1,random_state=int(rng.integers(2**31)))
        acquired=acquired.drop_duplicates('item_id').set_index('item_id').loc[acquired_ids].reset_index()
        evaluation=group[group['item_id'].isin(evaluation_ids)]
        subject={f:plain(subjects[sid].get(f)) for f in _SUBJECT_FIELDS}
        out.append({'key':json.dumps(subject,sort_keys=True),'subject':subject,'a':acquired['response'].to_numpy(float),
                    'y':evaluation['response'].to_numpy(float),'support':np.stack([vector(i) for i in acquired['item_id']]),'query':np.stack([vector(i) for i in evaluation['item_id']])})
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,meta=json.dumps([{k:r[k] for k in ['key','subject']} for r in out]),**{f'{i}_{k}':r[k] for i,r in enumerate(out) for k in names})
    return out
