"""Original source seed73 trajectories plus input text, never query labels for encoding."""
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import _load_directory,_stable_seed
from .metadata_features import encode
from .semantic_features import SemanticFeatures


def episodes(directory,tokenizer,old_engine):
    response,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory)
    groups=[(sid,g) for sid,g in response.groupby('subject_id',sort=False) if sid in subjects and g.item_id.nunique()>=80]
    if len(groups)>25:
        rng=np.random.default_rng(_stable_seed(2026,directory,'pair_sample'));groups=[groups[i] for i in np.sort(rng.choice(len(groups),25,replace=False))]
    pending=[];needed=set()
    for sid,g in groups:
        ids=pd.unique(g.item_id).copy();rng=np.random.default_rng(_stable_seed(73,directory,sid));rng.shuffle(ids)
        aid=ids[:31];eid=ids[len(ids)//2:len(ids)//2+64];needed.update(aid);needed.update(eid)
        acquired=g[g.item_id.isin(aid)].sample(frac=1,random_state=int(rng.integers(2**31))).drop_duplicates('item_id').set_index('item_id').loc[aid]
        query=g[g.item_id.isin(eid)].groupby('item_id',sort=False).response.agg(['size','sum']).loc[eid]
        pending.append((str(sid),aid,eid,acquired.response.to_numpy(float),query['size'].to_numpy(float),query['sum'].to_numpy(float)))
    ids=list(needed)
    if not ids:return []
    canonical=[SemanticFeatures.canonical(items[i]['content']) for i in ids]
    tokens=tokenizer(canonical,add_special_tokens=False,truncation=False,return_attention_mask=False)['input_ids']
    tokens={i:tuple(t) for i,t in zip(ids,tokens)};sem={i:old_engine.vector(items[i]['content']) for i in ids};meta={i:encode(items[i]['item_features']) for i in ids}
    return [{'subject_id':sid,'labels':y,'counts':m,'successes':s,'tokens':[tokens[i] for i in [*a,*q]],'support_n':len(a),
             'old_semantic':np.stack([sem[i] for i in [*a,*q]]),'metadata':np.stack([meta[i] for i in [*a,*q]])} for sid,a,q,y,m,s in pending]
