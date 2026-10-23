"""Frozen PAIEC source data for the five plan_2 interventions.

Historical feature modules are imported unchanged. Cache files are outside Git.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import ALC_WEIGHTS, BUDGETS, _load_directory, _stable_seed, _vectors
from .evidence_reliability import statistics
from .metadata_features import encode as metadata_encode
from .reliability_training import source_data
from .reproduction_episodes import DATA_ROOT, DATA_REVISION, source_episodes
from .train import _load_examples, _pair_weights

CACHE = Path('D:/paiec_research_cache/training_mechanism_v5')
FIELDS = ('z', 'base', 'delta', 'fraction', 'weight')


def digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else str(part).encode())
    return h.hexdigest()


def cache_path(kind, name, extra=''):
    fingerprint = digest(DATA_REVISION, Path(__file__).read_bytes(),
                         Path('mlp/model.py').read_bytes(), Path('mlp/evaluate.py').read_bytes())
    folder = CACHE / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (digest(fingerprint, name, extra) + '.npz')


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def row_weights(groups):
    pair = _pair_weights(groups)
    names = np.asarray([g.split('\0', 1)[0] for g in groups])
    unique, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    benchmarks = sorted(set(names))
    pairs_per = {b: sum(g.split('\0', 1)[0] == b for g in unique) for b in benchmarks}
    benchmark = np.asarray([len(groups) / (len(benchmarks) * pairs_per[b] * counts[i])
                            for b, i in zip(names, inv)], dtype=np.float32)
    mixture = (pair + benchmark) / 2
    assert abs(pair.mean() - 1) < 1e-6 and abs(benchmark.mean() - 1) < 1e-6
    return pair, mixture, {
        'rows': len(groups), 'pairs': len(unique), 'benchmarks': len(benchmarks),
        'mass': {b: {'pair': float(pair[names == b].sum() / len(groups)),
                     'benchmark': float(benchmark[names == b].sum() / len(groups)),
                     'mixture': float(mixture[names == b].sum() / len(groups)),
                     'pairs': int(pairs_per[b])} for b in benchmarks},
        'range': {'pair': [float(pair.min()), float(pair.max())],
                  'mixture': [float(mixture.min()), float(mixture.max())]}}


def mlp_training(names):
    rows = []
    for name in names:
        path = cache_path('mlp_train', name)
        if not path.exists():
            x, y, groups = _load_examples(DATA_ROOT, [name], 400, 2026, 'all', 'response')
            np.savez_compressed(path, x=x, y=y, groups=groups)
        rows.append(load_npz(path))
    result = {k: np.concatenate([r[k] for r in rows]) for k in ('x', 'y', 'groups')}
    result['pair_weight'], result['mixture_weight'], result['weight_audit'] = row_weights(result['groups'])
    result['data_hash'] = digest(*(result[k].tobytes() for k in ('x', 'y', 'groups')))
    return result


def raw_episodes(directory, engine, ridge, seeds=(73,), include_mlp=False):
    """The measurement_episodes sampler, with identities and optional row features."""
    manifest = json.loads((DATA_ROOT / 'corpus_manifest.json').read_text(encoding='utf-8'))
    assert manifest['revision'] == DATA_REVISION
    response, subjects, items = _load_directory(DATA_ROOT, directory)
    groups = response.groupby('subject_id', sort=False)
    eligible = [sid for sid, g in groups if g.item_id.nunique() >= 80 and sid in subjects]
    if len(eligible) > 25:
        rng = np.random.default_rng(_stable_seed(2026, directory, 'pair_sample'))
        eligible = [eligible[i] for i in np.sort(rng.choice(len(eligible), 25, replace=False))]
    vectors = {}

    def vector(i):
        if i not in vectors:
            vectors[i] = np.r_[engine.vector(items[i]['content']) / np.sqrt(ridge),
                               metadata_encode(items[i]['item_features']) / 4.]
        return vectors[i]

    for sid in eligible:
        group = groups.get_group(sid)
        for seed in seeds:
            ids = pd.unique(group.item_id).copy()
            rng = np.random.default_rng(_stable_seed(seed, directory, sid))
            rng.shuffle(ids)
            aid, eid = ids[:31], ids[len(ids)//2:len(ids)//2 + 64]
            acquired = group[group.item_id.isin(aid)].sample(
                frac=1, random_state=int(rng.integers(2**31)))
            acquired = acquired.drop_duplicates('item_id').set_index('item_id').loc[aid].reset_index()
            query_rows = group[group.item_id.isin(eid)].copy()
            query = query_rows.groupby('item_id', sort=False).response.agg(['size', 'sum']).loc[eid]
            assert len(aid) == 31 and set(aid).isdisjoint(eid)
            result = {'benchmark': directory, 'subject_id': str(sid), 'seed': int(seed),
                      'support_ids': list(map(str, aid)), 'query_ids': list(map(str, eid)),
                      'labels': acquired.response.to_numpy(float),
                      'counts': query['size'].to_numpy(float), 'successes': query['sum'].to_numpy(float)}
            if include_mlp:
                # Different interactors can encode different features for one item.
                query_rows['_variant'] = query_rows.interactors.map(str)
                keys = ['item_id', '_variant']
                aggregated = query_rows.groupby(keys, sort=False, dropna=False).response.agg(['size', 'sum'])
                representatives = query_rows.drop_duplicates(keys).set_index(keys).loc[aggregated.index].reset_index()
                result.update(support=_vectors(subjects[sid], items, acquired),
                              query=_vectors(subjects[sid], items, representatives),
                              counts=aggregated['size'].to_numpy(float),
                              successes=aggregated['sum'].to_numpy(float))
                assert result['counts'].sum() == len(query_rows)
            else:
                result.update(support=np.stack([vector(i) for i in aid]),
                              query=np.stack([vector(i) for i in eid]))
            result['identity_hash'] = digest(json.dumps({k: result[k] for k in
                ('benchmark', 'subject_id', 'seed', 'support_ids', 'query_ids')}, sort_keys=True),
                result['labels'].tobytes(), result['support'].tobytes(), result['query'].tobytes())
            yield result


def mlp_validation(name, engine):
    path = cache_path('mlp_validation', name)
    if not path.exists():
        rows = list(raw_episodes(name, engine, 1., include_mlp=True))
        fields = ('support', 'query', 'labels', 'counts', 'successes')
        meta = [{k: v for k, v in row.items() if k not in fields} for row in rows]
        np.savez_compressed(path, metadata=json.dumps(meta),
                            **{f'{i}_{k}': row[k] for i, row in enumerate(rows) for k in fields})
    data = load_npz(path)
    return [{**m, **{k: data[f'{i}_{k}'] for k in ('support', 'query', 'labels', 'counts', 'successes')}}
            for i, m in enumerate(json.loads(str(data['metadata'])))]


def reliability_fixed(name, engine, ridge):
    result = source_data(name, engine, ridge)
    episodes = source_episodes(name, engine)
    # Exact old float32 statistics; metadata only provides row counting.
    meta = []
    offset = 0
    for episode in episodes:
        q = len(episode['counts'])
        meta.append({'benchmark': name, 'subject_id': episode['subject_id'], 'seed': 73,
                     'labels': episode['labels'].tolist(), 'counts': episode['counts'].tolist(),
                     'successes': episode['successes'].tolist(), 'start': offset, 'q': q})
        offset += 4*q
    assert offset == len(result['z'])
    result['meta'] = meta
    return result


def make_bank(episodes):
    blocks = {k: [] for k in FIELDS}
    meta, baseline = [], np.zeros(6)
    offset = 0
    for episode in episodes:
        counts, successes = episode['counts'], episode['successes']
        q = len(counts)
        for j, n in enumerate(BUDGETS):
            y = episode['labels'][:n]
            b = (1 + y.sum()) / (2 + n)
            p = np.full(q, b)
            if n >= 3:
                z, b, d = statistics(episode['support'][:n], episode['query'], y)
                p = np.clip(b + d, 0, 1)
                for key, value in zip(FIELDS, (z, np.full(q, b), d,
                        successes/counts, ALC_WEIGHTS[j]*counts/counts.sum())):
                    blocks[key].append(value)
            baseline[j] += np.sum(counts*p*p - 2*successes*p + successes)/counts.sum()
        meta.append({'benchmark': episode['benchmark'], 'subject_id': episode['subject_id'],
                     'seed': episode['seed'], 'identity_hash': episode['identity_hash'],
                     'support_ids': episode['support_ids'], 'query_ids': episode['query_ids'],
                     'labels': episode['labels'].tolist(), 'counts': counts.tolist(),
                     'successes': successes.tolist(), 'start': offset, 'q': q})
        offset += 4*q
    result = {k: np.concatenate(v).astype(np.float32) for k, v in blocks.items()}
    result.update(pairs=len(meta), constant=float(baseline[:2] @ ALC_WEIGHTS[:2]),
                  baseline_brier=baseline/len(meta), meta=meta)
    return result


def resampled_banks(name, engine, ridge):
    paths = {seed: cache_path('resampled', name, f'{ridge}|{seed}|{engine.fingerprint}')
             for seed in range(5501, 5521)}
    missing = [seed for seed, path in paths.items() if not path.exists()]
    if missing:
        by_seed = {seed: [] for seed in missing}
        # Generate one pair at a time; retain only small statistics, not all 640D frames.
        for ep in raw_episodes(name, engine, ridge, seeds=missing):
            by_seed[ep['seed']].append(make_bank([ep]))
        for seed in missing:
            bank = concatenate(by_seed[seed])
            np.savez_compressed(paths[seed], **{k: v for k, v in bank.items() if k != 'meta'},
                                metadata=json.dumps(bank['meta']))
    return paths


def bank_load(path):
    result = load_npz(path)
    result['meta'] = json.loads(str(result.pop('metadata')))
    return result


def concatenate(rows):
    result = {k: np.concatenate([r[k] for r in rows]) for k in FIELDS}
    meta, offset = [], 0
    for row in rows:
        meta.extend([{**m, 'start': m['start'] + offset} for m in row['meta']])
        offset += len(row['z'])
    result.update(meta=meta, pairs=sum(int(r['pairs']) for r in rows),
                  constant=sum(float(r['constant']) for r in rows))
    return result
