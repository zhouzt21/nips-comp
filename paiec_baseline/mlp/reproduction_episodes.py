"""Auditable, response-counted episodes for paper-core transfer experiments.

Historical samplers and their byte-addressed caches are deliberately imported,
not edited. IDs identify split membership only and never enter model tensors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort_semantic import CachedSemantic
from .evaluate import ALC_WEIGHTS, BUDGETS, _load_directory, _stable_seed
from .metadata_features import encode as encode_metadata
from .model import _SUBJECT_FIELDS, _string, encode as encode_original
from .run.run_metadata_round import prediction

DATA_ROOT = Path('D:/paiec_payload/data')
CACHE_ROOT = Path('D:/paiec_research_cache/reproduction_v4')
DATA_REVISION = '54074453da7b9911be0f48e4c560be547bc84d21'
ARRAY_FIELDS = ('support', 'query', 'labels', 'counts', 'successes', 'subject')


def subject_features(subject):
    """Fold the existing permitted-field hash into 256 coordinates; no IDs."""
    vector = encode_original([subject, {}], include_item=False).reshape(8, 256).sum(0)
    norm = np.linalg.norm(vector)
    return np.asarray(vector / norm if norm else vector, dtype=np.float32)


def cache_fingerprint(engine):
    parts = [DATA_REVISION.encode(), engine.fingerprint.encode(),
             Path(__file__).read_bytes(), Path('mlp/model.py').read_bytes(),
             Path('mlp/metadata_features.py').read_bytes(), Path('mlp/evaluate.py').read_bytes()]
    return hashlib.sha256(b''.join(parts)).hexdigest()


def generate_episodes(directory, engine, seeds=(73,), pair_cap=25, full=False):
    manifest = json.loads((DATA_ROOT / 'corpus_manifest.json').read_text(encoding='utf-8'))
    if manifest['revision'] != DATA_REVISION:
        raise ValueError('Unexpected data revision')
    response, subjects, items = _load_directory(DATA_ROOT, directory)
    groups = response.groupby('subject_id', sort=False)
    eligible = [sid for sid, g in groups if g.item_id.nunique() >= 80 and sid in subjects]
    if pair_cap is not None and len(eligible) > pair_cap:
        rng = np.random.default_rng(_stable_seed(2026, directory, 'pair_sample'))
        eligible = [eligible[i] for i in np.sort(rng.choice(len(eligible), pair_cap, replace=False))]
    vectors = {}

    def vector(item_id):
        if item_id not in vectors:
            item = items[item_id]
            vectors[item_id] = np.r_[engine.vector(item['content']),
                                     encode_metadata(item['item_features'])].astype(np.float32)
        return vectors[item_id]

    for sid in eligible:
        group = groups.get_group(sid)
        subject = {field: _string(subjects[sid].get(field, '')) for field in _SUBJECT_FIELDS}
        subject_key = hashlib.sha256(json.dumps(subject, sort_keys=True).encode()).hexdigest()
        for seed in seeds:
            ids = pd.unique(group.item_id).copy()
            rng = np.random.default_rng(_stable_seed(seed, directory, sid))
            rng.shuffle(ids)
            aid = ids[:31]
            cut = len(ids) // 2
            eid = ids[cut:] if full else ids[cut:cut + 64]
            acquired = group[group.item_id.isin(aid)].sample(
                frac=1, random_state=int(rng.integers(2**31)))
            acquired = acquired.drop_duplicates('item_id').set_index('item_id').loc[aid]
            query = group[group.item_id.isin(eid)].groupby('item_id', sort=False).response.agg(['size', 'sum']).loc[eid]
            assert set(aid).isdisjoint(eid) and len(aid) == 31
            yield {
                'benchmark': directory, 'subject_id': str(sid), 'subject_key': subject_key,
                'seed': int(seed), 'support_item_ids': [str(i) for i in aid],
                'query_item_ids': [str(i) for i in eid],
                'support': np.stack([vector(i) for i in aid]),
                'query': np.stack([vector(i) for i in eid]),
                'labels': acquired.response.to_numpy(np.float32),
                'counts': query['size'].to_numpy(np.float64),
                'successes': query['sum'].to_numpy(np.float64),
                'subject': subject_features(subject),
            }


def source_episodes(directory, engine):
    fingerprint = cache_fingerprint(engine)
    key = hashlib.sha256(f'{directory}|73|25|{fingerprint}'.encode()).hexdigest()
    path = CACHE_ROOT / 'source' / (key + '.npz')
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            return [{**meta, **{field: data[f'{i}_{field}'] for field in ARRAY_FIELDS}}
                    for i, meta in enumerate(metadata)]
    episodes = list(generate_episodes(directory, engine))
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [{key: value for key, value in episode.items() if key not in ARRAY_FIELDS}
                for episode in episodes]
    np.savez_compressed(path, metadata=json.dumps(metadata),
                        **{f'{i}_{field}': episode[field]
                           for i, episode in enumerate(episodes) for field in ARRAY_FIELDS})
    return episodes


def brier(probabilities, counts, successes):
    p = np.asarray(probabilities, dtype=np.float64)
    return float(np.sum(counts * p * p - 2 * successes * p + successes) / counts.sum())


def accuracy(probabilities, counts, successes):
    """Response-weighted correctness at fixed p>=0.5 threshold; secondary only."""
    correct = np.where(np.asarray(probabilities) >= 0.5, successes, counts - successes)
    return float(np.sum(correct) / counts.sum())


def r25_predict(episode, budget, semantic_ridge, metadata_ridge=16.0):
    scale = np.r_[np.full(384, np.sqrt(semantic_ridge)),
                  np.full(256, np.sqrt(metadata_ridge))]
    return prediction(episode['support'][:budget].astype(float) / scale,
                      episode['query'].astype(float) / scale,
                      episode['labels'][:budget].astype(float), 1.0)


def baseline_predictions(episode, budget, semantic_ridge, selected_alpha):
    successes = episode['labels'][:budget].sum(dtype=np.float64)
    shape = len(episode['counts'])
    return {
        'R25': r25_predict(episode, budget, semantic_ridge),
        'smoothed_mean_2': np.full(shape, (1 + successes) / (2 + budget)),
        'training_selected_smoothed': np.full(shape, (.5 * selected_alpha + successes) / (selected_alpha + budget)),
        'constant_0.5': np.full(shape, .5),
    }


def score_baselines(episodes, semantic_ridge, selected_alpha):
    rows = []
    for episode in episodes:
        scores, accs = {}, {}
        for budget in BUDGETS:
            for name, p in baseline_predictions(episode, budget, semantic_ridge, selected_alpha).items():
                scores.setdefault(name, []).append(brier(p, episode['counts'], episode['successes']))
                accs.setdefault(name, []).append(accuracy(p, episode['counts'], episode['successes']))
        rows.append({'benchmark': episode['benchmark'], 'subject_id': episode['subject_id'],
                     'seed': episode['seed'], 'brier': scores, 'accuracy': accs})
    return rows


def aggregate(rows):
    if not rows:
        raise ValueError('Cannot aggregate an empty evaluation')
    methods = list(rows[0]['brier'])
    result = {'pairs': len(rows), 'results': {}, 'by_benchmark': {}}
    for name in methods:
        curve = np.mean([row['brier'][name] for row in rows], axis=0)
        result['results'][name] = {
            'brier_mean': curve.tolist(), 'alc_mean': float(curve @ ALC_WEIGHTS),
            'accuracy_at_0_5': np.mean([row['accuracy'][name] for row in rows], axis=0).tolist(),
        }
    for benchmark in sorted({row['benchmark'] for row in rows}):
        group = [row for row in rows if row['benchmark'] == benchmark]
        result['by_benchmark'][benchmark] = {
            'pairs': len(group),
            'brier_mean': {name: np.mean([row['brier'][name] for row in group], axis=0).tolist() for name in methods},
            'alc_mean': {name: float(np.mean([row['brier'][name] for row in group], axis=0) @ ALC_WEIGHTS) for name in methods},
        }
    result['benchmark_equal_alc'] = {
        name: float(np.mean([group['alc_mean'][name] for group in result['by_benchmark'].values()]))
        for name in methods}
    return result


def source_fold(fold, engine=None):
    engine = engine or CachedSemantic()
    old = json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text())
    prior = json.loads(Path(f'runs/mlp_failure_2026_10_22/stat_prior/fold{fold}_prior.json').read_text())
    # These frozen values are from the historical training-only prior selection.
    alpha = (2.0, 2.0, 4.0)[fold]
    data = {name: source_episodes(name, engine) for name in old['train_benchmarks']}
    data = {name: episodes for name, episodes in data.items() if episodes}
    episodes = [episode for name in sorted(data) for episode in data[name]]
    assert len(episodes) == old['training_pairs']
    reference = old['training_alc'][old['metadata_grid'].index(16.0)]
    summary = aggregate(score_baselines(episodes, old['semantic_ridge'], alpha))
    observed = summary['results']['R25']['alc_mean']
    assert abs(observed - reference) < 1e-7, (observed, reference)
    old_inner = json.loads(Path(f'runs/research_loop/R43/fold{fold}_selection.json').read_text())
    halves = old_inner['inner_halves']
    assert set(halves[0]).isdisjoint(halves[1])
    assert set(halves[0] + halves[1]) == set(data)
    assert set(old['heldout_benchmarks']).isdisjoint(data)
    return data, {**old, 'selected_alpha': alpha, 'strong_mean_source': prior,
                  'inner_halves': halves, 'baseline_summary': summary,
                  'replay_error': observed - reference, 'cache_fingerprint': cache_fingerprint(engine)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        out = Path('runs/research_loop/reproduction_v4'); out.mkdir(parents=True, exist_ok=True)
        engine = CachedSemantic()
        for fold in range(3):
            data, info = source_fold(fold, engine)
            summary = {key: info[key] for key in ('train_benchmarks', 'heldout_benchmarks', 'training_pairs',
                       'semantic_ridge', 'selected_alpha', 'inner_halves', 'replay_error', 'cache_fingerprint', 'baseline_summary')}
            (out / f'source_fold{fold}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(json.dumps({'fold': fold, 'pairs': info['training_pairs'],
                              'baseline_alc': info['baseline_summary']['results']['R25']['alc_mean'],
                              'replay_error': info['replay_error']}), flush=True)
            del data, info


if __name__ == '__main__':
    main()
