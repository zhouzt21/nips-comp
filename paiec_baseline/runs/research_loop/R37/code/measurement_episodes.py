"""Stream count-aggregated episodes with the unchanged historical item trajectories."""
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import _load_directory, _stable_seed
from .metadata_features import encode


def episodes(directory, engine, semantic_ridge, metadata_ridge, full=False, seeds=(0, 1, 2), pair_cap=None):
    response, subjects, items = _load_directory(Path('D:/paiec_payload/data'), directory)
    vectors = {}
    def vector(i):
        if i not in vectors:
            vectors[i] = np.r_[engine.vector(items[i]['content'])/np.sqrt(semantic_ridge),
                               encode(items[i]['item_features'])/np.sqrt(metadata_ridge)]
        return vectors[i]
    groups = response.groupby('subject_id', sort=False)
    eligible = [sid for sid, g in groups if g.item_id.nunique() >= 80 and sid in subjects]
    if pair_cap is not None and len(eligible) > pair_cap:
        rng = np.random.default_rng(_stable_seed(2026, directory, 'pair_sample'))
        eligible = [eligible[i] for i in np.sort(rng.choice(len(eligible), pair_cap, replace=False))]
    for sid in eligible:
        g = groups.get_group(sid)
        for seed in seeds:
            ids = pd.unique(g['item_id']).copy()
            rng = np.random.default_rng(_stable_seed(seed, directory, sid)); rng.shuffle(ids)
            cut = len(ids)//2
            aid, eid = ids[:31], (ids[cut:] if full else ids[cut:cut+64])
            acquired = g[g.item_id.isin(aid)].sample(frac=1, random_state=int(rng.integers(2**31)))
            acquired = acquired.drop_duplicates('item_id').set_index('item_id').loc[aid]
            query = g[g.item_id.isin(eid)].groupby('item_id', sort=False)['response'].agg(['size', 'sum']).loc[eid]
            yield {'subject_id': str(sid), 'seed': seed,
                   'support': np.stack([vector(i) for i in aid]), 'query': np.stack([vector(i) for i in eid]),
                   'labels': acquired.response.to_numpy(float), 'counts': query['size'].to_numpy(float),
                   'successes': query['sum'].to_numpy(float)}


def weighted_brier(p, counts, successes):
    return float(np.sum(counts*p*p-2*successes*p+successes)/counts.sum())
