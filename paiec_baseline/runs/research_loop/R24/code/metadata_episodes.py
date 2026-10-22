"""Same source/outer episodes with additional permitted item metadata vectors."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import _load_directory,_stable_seed
from .metadata_features import encode


def episodes(directory,seed,cap,engine):
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path('mlp/metadata_features.py').read_bytes()+engine.fingerprint.encode()).hexdigest()
    path=Path('D:/paiec_research_cache/metadata_episodes')/(hashlib.sha256(f'{directory}|{seed}|{cap}|{fingerprint}'.encode()).hexdigest()+'.npz')
    names=['a','y','support','query','support_meta','query_meta']
    if path.exists():
        with np.load(path,allow_pickle=False) as data:return [{k:data[f'{i}_{k}'] for k in names} for i in range(int(data['count']))]
    responses,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory)
    groups=[(s,g) for s,g in responses.groupby('subject_id',sort=False) if g['item_id'].nunique()>=80 and s in subjects]
    if len(groups)>cap:
        rng=np.random.default_rng(_stable_seed(2026,directory,'pair_sample'));groups=[groups[i] for i in np.sort(rng.choice(len(groups),cap,replace=False))]
    vectors={};metas={};out=[]
    def vector(i):
        if i not in vectors:vectors[i]=engine.vector(items[i]['content']);metas[i]=encode(items[i]['item_features'])
        return vectors[i]
    for sid,group in groups:
        ids=pd.unique(group['item_id']).copy();rng=np.random.default_rng(_stable_seed(seed,directory,sid));rng.shuffle(ids)
        cut=len(ids)//2;acquired_ids=ids[:31];evaluation_ids=ids[cut:cut+64]
        acquired=group[group['item_id'].isin(acquired_ids)].sample(frac=1,random_state=int(rng.integers(2**31)))
        acquired=acquired.drop_duplicates('item_id').set_index('item_id').loc[acquired_ids].reset_index()
        evaluation=group[group['item_id'].isin(evaluation_ids)]
        support=np.stack([vector(i) for i in acquired['item_id']]);query=np.stack([vector(i) for i in evaluation['item_id']])
        out.append({'a':acquired['response'].to_numpy(float),'y':evaluation['response'].to_numpy(float),'support':support,'query':query,
                    'support_meta':np.stack([metas[i] for i in acquired['item_id']]),'query_meta':np.stack([metas[i] for i in evaluation['item_id']])})
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,count=len(out),**{f'{i}_{k}':r[k] for i,r in enumerate(out) for k in names})
    return out
