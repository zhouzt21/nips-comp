"""Local disk features for matched cohort experiments; no model/network fallback."""
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from .semantic_features import SemanticFeatures,MODEL_REVISION
from .evaluate import _load_directory,_stable_seed
from .model import _SUBJECT_FIELDS


class CachedSemantic:
    def __init__(self):
        self.fingerprint=hashlib.sha256(MODEL_REVISION.encode()+Path('mlp/semantic_features.py').read_bytes()+Path('mlp/structured_text.py').read_bytes()).hexdigest()
        path=Path('D:/paiec_research_cache/semantic')/(self.fingerprint+'.sqlite')
        self.db=sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True)

    def vector(self,text):
        key=hashlib.sha256(SemanticFeatures.canonical(text).encode()).hexdigest()
        row=self.db.execute('SELECT vector FROM features WHERE key=?',(key,)).fetchone()
        if row is None:raise RuntimeError('Required R7 semantic cache row absent')
        return np.frombuffer(row[0],dtype=np.float32).astype(float)


def episodes(directory,seed,engine):
    key=hashlib.sha256(f'{directory}|{seed}|100|{engine.fingerprint}'.encode()+Path(__file__).read_bytes()).hexdigest()
    path=Path('D:/paiec_research_cache/cohort_semantic')/(key+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['meta']))
            return [{**m,**{k:data[f'{i}_{k}'] for k in ['a','y','support','query']}} for i,m in enumerate(meta)]
    responses,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory)
    groups=[(s,g) for s,g in responses.groupby('subject_id',sort=False) if g['item_id'].nunique()>=80 and s in subjects]
    if len(groups)>100:
        rng=np.random.default_rng(_stable_seed(2026,directory,'pair_sample'));groups=[groups[i] for i in np.sort(rng.choice(len(groups),100,replace=False))]
    vectors={};out=[]
    def vector(item_id):
        if item_id not in vectors:vectors[item_id]=engine.vector(items[item_id]['content'])
        return vectors[item_id]
    def plain(value):
        if value is None or pd.isna(value):return ''
        text=str(value).strip();return '' if text.lower() in {'nan','none','<na>','nat'} else text
    for sid,group in groups:
        ids=pd.unique(group['item_id']).copy();rng=np.random.default_rng(_stable_seed(seed,directory,sid));rng.shuffle(ids)
        cut=len(ids)//2;acquired_ids=ids[:31];evaluation_ids=ids[cut:cut+64]
        acquired=group[group['item_id'].isin(acquired_ids)].sample(frac=1,random_state=int(rng.integers(2**31)))
        acquired=acquired.drop_duplicates('item_id').set_index('item_id').loc[acquired_ids].reset_index()
        evaluation=group[group['item_id'].isin(evaluation_ids)]
        subject={field:plain(subjects[sid].get(field)) for field in _SUBJECT_FIELDS}
        out.append({'subject':subject,'key':json.dumps(subject,sort_keys=True),'a':acquired['response'].to_numpy(float),
                    'y':evaluation['response'].to_numpy(float),'support':np.stack([vector(i) for i in acquired['item_id']]),
                    'query':np.stack([vector(i) for i in evaluation['item_id']])})
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,meta=json.dumps([{k:r[k] for k in ['subject','key']} for r in out]),
                        **{f'{i}_{k}':r[k] for i,r in enumerate(out) for k in ['a','y','support','query']})
    return out
