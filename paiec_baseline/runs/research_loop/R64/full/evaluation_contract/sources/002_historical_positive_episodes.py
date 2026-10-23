"""Exact historical full sampler, with variant-aware MLP inputs and replay IDs."""
import hashlib
import json
import numpy as np
import pandas as pd
from historical_positive_common import *
from mlp.evaluate import _load_directory, _stable_seed, _vectors
from mlp.metadata_features import encode as meta_encode
from mlp.model import encode_mode
from mlp.reproduction_episodes import DATA_ROOT, DATA_REVISION

def episodes(directory,engine,ridge,fold,include_mlp=False,seeds=(0,1,2),pair_cap=None):
    assert read(DATA_ROOT/'corpus_manifest.json')['revision']==DATA_REVISION
    response,subjects,items=_load_directory(DATA_ROOT,directory)
    groups=response.groupby('subject_id',sort=False)
    eligible=[sid for sid,g in groups if g.item_id.nunique()>=80 and sid in subjects]
    if pair_cap is not None and len(eligible)>pair_cap:
        rng=np.random.default_rng(_stable_seed(2026,directory,'pair_sample'))
        eligible=[eligible[i] for i in np.sort(rng.choice(len(eligible),pair_cap,replace=False))]
    vectors={}
    def vector(i):
        if i not in vectors:
            vectors[i]=np.r_[engine.vector(items[i]['content'])/np.sqrt(ridge),
                              meta_encode(items[i]['item_features'])/4.]
        return vectors[i]
    for pair_index,sid in enumerate(eligible):
        g=groups.get_group(sid)
        mlp_cache={}
        def mlp_vectors(rows):
            output=[]
            for i,interactors in rows[['item_id','interactors']].itertuples(index=False,name=None):
                key=(i,str(interactors))
                if key not in mlp_cache:
                    item=items[i]
                    mlp_cache[key]=encode_mode([subjects[sid],{'item_content':item['content'],
                        'item_features':item['item_features'],'interactors':interactors}],'all')
                output.append(mlp_cache[key])
            return np.stack(output)
        for seed in seeds:
            ids=pd.unique(g.item_id).copy()
            rng=np.random.default_rng(_stable_seed(seed,directory,sid)); rng.shuffle(ids)
            aid,eid=ids[:31],ids[len(ids)//2:len(ids)//2+64]
            assert len(aid)==31 and set(aid).isdisjoint(eid)
            acquired=g[g.item_id.isin(aid)].sample(frac=1,random_state=int(rng.integers(2**31)))
            acquired=acquired.drop_duplicates('item_id').set_index('item_id').loc[aid].reset_index()
            qr=g[g.item_id.isin(eid)].copy()
            qr['_variant']=qr.interactors.map(str)
            keys=['item_id','_variant']
            aggregated=qr.groupby(keys,sort=False,dropna=False).response.agg(['size','sum'])
            reps=qr.drop_duplicates(keys).set_index(keys).loc[aggregated.index].reset_index()
            counts=aggregated['size'].to_numpy(float); successes=aggregated['sum'].to_numpy(float)
            labels=acquired.response.to_numpy(float)
            query_map={i:j for j,i in enumerate(eid)}
            vi=np.asarray([query_map[i] for i in reps.item_id],dtype=np.int64)
            item_counts=qr.groupby('item_id',sort=False).response.agg(['size','sum']).loc[eid]
            assert np.array_equal(np.bincount(vi,weights=counts,minlength=len(eid)),item_counts['size'])
            assert np.array_equal(np.bincount(vi,weights=successes,minlength=len(eid)),item_counts['sum'])
            meta={'fold':fold,'benchmark':directory,'subject_id':str(sid),'seed':int(seed),
                'support_ids':list(map(str,aid)),'query_ids':list(map(str,eid)),
                'support_labels':labels.tolist(),
                'query_variant_item_indices':vi.tolist(),
                'query_variant_sha256':[hashlib.sha256(str(v).encode()).hexdigest() for v in reps['_variant']],
                'support_variant_sha256':[hashlib.sha256(str(v).encode()).hexdigest() for v in acquired.interactors],
                'counts':counts.tolist(),'successes':successes.tolist()}
            identity=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
            ep={**meta,'identity':identity,'metadata':meta,'labels':labels,
                'counts':counts,'successes':successes,'variant_indices':vi,
                'support':np.stack([vector(i) for i in aid]),'query':np.stack([vector(i) for i in eid])}
            if include_mlp:
                ep['mlp_support']=mlp_vectors(acquired); ep['mlp_query']=mlp_vectors(reps)
                # Deterministic first pair audit of the cache against the unchanged encoder.
                if pair_index==0 and seed==seeds[0]:
                    assert np.array_equal(ep['mlp_support'],_vectors(subjects[sid],items,acquired))
                    assert np.array_equal(ep['mlp_query'],_vectors(subjects[sid],items,reps))
            yield ep
