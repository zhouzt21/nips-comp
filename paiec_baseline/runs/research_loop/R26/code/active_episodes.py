"""Fixed random candidate stream, original first31 draws retained exactly."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import _load_directory,_stable_seed
from .metadata_features import encode


def episodes(directory,seed,cap,engine):
    key=hashlib.sha256(f'{directory}|{seed}|{cap}|{engine.fingerprint}'.encode()+Path(__file__).read_bytes()+Path('mlp/metadata_features.py').read_bytes()).hexdigest()
    path=Path('D:/paiec_research_cache/active_episodes')/(key+'.npz');names=['a','y','support','query','support_meta','query_meta']
    if path.exists():
        with np.load(path,allow_pickle=False) as data:return [{k:data[f'{i}_{k}'] for k in names} for i in range(int(data['count']))]
    responses,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory)
    groups=[(s,g) for s,g in responses.groupby('subject_id',sort=False) if g['item_id'].nunique()>=80 and s in subjects]
    if len(groups)>cap:
        rng=np.random.default_rng(_stable_seed(2026,directory,'pair_sample'));groups=[groups[i] for i in np.sort(rng.choice(len(groups),cap,replace=False))]
    pending=[];texts={}
    for sid,group in groups:
        ids=pd.unique(group['item_id']).copy();rng=np.random.default_rng(_stable_seed(seed,directory,sid));rng.shuffle(ids)
        cut=len(ids)//2;initial_ids=ids[:31];extra_ids=ids[31:min(64,cut)];evaluation_ids=ids[cut:cut+64]
        def one_per_item(chosen):
            data=group[group['item_id'].isin(chosen)].sample(frac=1,random_state=int(rng.integers(2**31)))
            return data.drop_duplicates('item_id').set_index('item_id').loc[chosen].reset_index()
        initial=one_per_item(initial_ids);extra=one_per_item(extra_ids);acquired=pd.concat([initial,extra],ignore_index=True)
        evaluation=group[group['item_id'].isin(evaluation_ids)]
        assert not set(acquired['item_id'])&set(evaluation['item_id'])
        item_ids=[*acquired['item_id'],*evaluation['item_id']]
        for i in item_ids:texts[i]=items[i]['content']
        pending.append((list(acquired['item_id']),list(evaluation['item_id']),acquired['response'].to_numpy(float),evaluation['response'].to_numpy(float)))
    encoded=engine.encode_many(list(texts.values())) if texts else {};vectors={i:encoded[engine.key(t)]['semantic'] for i,t in texts.items()};metadata={i:encode(items[i]['item_features']) for i in texts}
    out=[{'a':a,'y':y,'support':np.stack([vectors[i] for i in ids]),'query':np.stack([vectors[i] for i in query]),
          'support_meta':np.stack([metadata[i] for i in ids]),'query_meta':np.stack([metadata[i] for i in query])} for ids,query,a,y in pending]
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,count=len(out),**{f'{i}_{k}':r[k] for i,r in enumerate(out) for k in names})
    print('candidate cache',directory,seed,len(out),'new embeddings',engine.encoded,flush=True)
    return out
