"""Visible item identities for conditional peer diagnostics; no private ID features."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import _load_directory,_stable_seed
from .model import _SUBJECT_FIELDS


def plain(value):
    if value is None or pd.isna(value):return ''
    text=str(value).strip()
    return '' if text.lower() in {'nan','none','<na>','nat'} else text


def item_key(item,interactors):
    fields=[plain(item['content']),plain(item['item_features']),plain(interactors)]
    return hashlib.sha256(json.dumps(fields,ensure_ascii=False).encode()).hexdigest()


def episodes(directory,seed,cap=100):
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    key=hashlib.sha256(f'{directory}|{seed}|{cap}|{fingerprint}|54074453da7b9911be0f48e4c560be547bc84d21'.encode()).hexdigest()
    path=Path('D:/paiec_research_cache/visible_item_evidence')/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    responses,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory)
    groups=[(s,g) for s,g in responses.groupby('subject_id',sort=False) if g['item_id'].nunique()>=80 and s in subjects]
    if len(groups)>cap:
        rng=np.random.default_rng(_stable_seed(2026,directory,'pair_sample'))
        groups=[groups[i] for i in np.sort(rng.choice(len(groups),cap,replace=False))]
    out=[]
    for sid,group in groups:
        ids=pd.unique(group['item_id']).copy();rng=np.random.default_rng(_stable_seed(seed,directory,sid));rng.shuffle(ids)
        cut=len(ids)//2;acquired_ids=ids[:31];evaluation_ids=ids[cut:cut+64]
        acquired=group[group['item_id'].isin(acquired_ids)].sample(frac=1,random_state=int(rng.integers(2**31)))
        acquired=acquired.drop_duplicates('item_id').set_index('item_id').loc[acquired_ids].reset_index()
        evaluation=group[group['item_id'].isin(evaluation_ids)]
        assert not set(acquired['item_id']) & set(evaluation['item_id'])
        subject={field:plain(subjects[sid].get(field)) for field in _SUBJECT_FIELDS}
        def keys(rows):return [item_key(items[i],v) for i,v in rows[['item_id','interactors']].itertuples(index=False,name=None)]
        out.append({'subject':subject,'key':json.dumps(subject,sort_keys=True),'support_keys':keys(acquired),'query_keys':keys(evaluation),
                    'a':acquired['response'].astype(int).tolist(),'y':evaluation['response'].astype(int).tolist()})
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out),encoding='utf-8')
    return out


def cohorts(rows,directory,size=4,cohort_seed=193):
    order=np.arange(len(rows));np.random.default_rng(_stable_seed(cohort_seed,directory,'cohort')).shuffle(order)
    for offset in range(0,len(order),size):yield [rows[i] for i in order[offset:offset+size]]
