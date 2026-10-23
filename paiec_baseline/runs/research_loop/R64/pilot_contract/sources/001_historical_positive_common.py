"""Shared contracts and reporting for the frozen R62--R64 follow-up."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'paiec_baseline'
sys.path.insert(0, str(BASE))
os.chdir(BASE)
import numpy as np
from mlp.training_mechanism_data import write_json, load_npz
from mlp.evaluate import BUDGETS, ALC_WEIGHTS
from mlp.reproduction_episodes import aggregate, brier, accuracy

ROOT = BASE/'runs/research_loop'
V5 = ROOT/'training_mechanism_v5'
ARCHIVES = [Path('D:/nips-comp/paiec-dataset/paiec_git_backups')/p/'workspace/paiec_baseline'
            for p in ('layout_20261022_120427', 'nips-comp_20261022_113534')]
EPOCHS = {'R54': [1,2,1], 'R56': [1,2,1], 'R43_query_best20': [0,0,20],
          'R55_resampled_query': [0,0,17], 'R55_resampled_episode': [1,20,17]}
BASELINES = ['R25', 'smoothed_mean_2', 'training_selected_smoothed']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def source(fold):
    return read(ROOT/f'reproduction_v4/source_fold{fold}.json')

def freeze(folder, paths, extra=None):
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {str(Path(p).resolve()): sha(p) for p in paths}
    path = folder/'contract.json'
    if path.exists():
        old = read(path)
        assert old['sha256'] == hashes, f'Frozen contract changed: {path}'
        assert old['extra'] == json.loads(json.dumps(extra)), f'Changed contract metadata: {path}'
        return old
    import subprocess, platform
    info = {'frozen_utc': now(), 'sha256': hashes, 'extra': extra,
            'command': [sys.executable, *sys.argv], 'cwd': str(BASE),
            'python': sys.version, 'platform': platform.platform(),
            'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'git_status': subprocess.check_output(['git','status','--short'],cwd=REPO,text=True)}
    write_json(path, info)
    for i,p in enumerate(paths):
        p = Path(p)
        dest = folder/'sources'/f'{i:03d}_{p.name}'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(p.read_bytes())
    return info

def check_freeze(folder):
    for p,h in read(folder/'contract.json')['sha256'].items():
        assert sha(p) == h, p

def score_episode(ep, predictions, extra=None):
    c,s = ep['counts'], ep['successes']
    for name,p in predictions.items():
        assert p.shape == (len(c),6), (name,p.shape,len(c))
        assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    return {'fold':ep['fold'], 'benchmark':ep['benchmark'], 'subject_id':ep['subject_id'],
            'seed':int(ep['seed']), 'identity':ep['identity'],
            'response_count':int(c.sum()), 'query_variants':len(c),
            'brier':{m:[brier(p[:,j],c,s) for j in range(6)] for m,p in predictions.items()},
            'accuracy':{m:[accuracy(p[:,j],c,s) for j in range(6)] for m,p in predictions.items()},
            **(extra or {})}

def summarize(rows, folder, expected_pairs=None):
    """Equal pairs, equal split seeds; cluster-resample Benchmark weighted sums."""
    summary = aggregate(rows)
    identities = {(r['benchmark'],r['subject_id']) for r in rows}
    seeds = sorted({r['seed'] for r in rows})
    assert len({(r['benchmark'],r['subject_id'],r['seed']) for r in rows})==len(rows)
    assert len(rows)==len(identities)*len(seeds)
    if expected_pairs is not None:
        assert len(identities)==expected_pairs
    summary.update(unique_pairs=len(identities), trajectories=len(rows), seeds=seeds,
                   folds={}, by_seed={}, comparisons={})
    methods = list(rows[0]['brier'])
    folds = sorted({r['fold'] for r in rows})
    for f in folds:
        summary['folds'][str(f)] = aggregate([r for r in rows if r['fold']==f])
    for s in seeds:
        summary['by_seed'][str(s)] = aggregate([r for r in rows if r['seed']==s])
    names = sorted(summary['by_benchmark'])
    counts = np.array([summary['by_benchmark'][b]['pairs'] for b in names],float)
    draws = np.random.default_rng(20261022).integers(len(names),size=(10000,len(names)))
    for method in methods:
        summary['comparisons'][method] = {}
        for baseline in BASELINES:
            if baseline not in methods or method==baseline:
                continue
            delta = np.array([summary['by_benchmark'][b]['alc_mean'][method]-
                              summary['by_benchmark'][b]['alc_mean'][baseline] for b in names])
            weighted = delta*counts
            boot = weighted[draws].sum(1)/counts[draws].sum(1)
            fd = [summary['folds'][str(f)]['results'][method]['alc_mean']-
                  summary['folds'][str(f)]['results'][baseline]['alc_mean'] for f in folds]
            alignment = np.array(['alignment_faking/' in b for b in names])
            comparison = {'delta_alc': float(weighted.sum()/counts.sum()),
                'benchmark_equal_delta':float(delta.mean()), 'fold_deltas':fd,
                'cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),
                'benchmarks_improved':int((delta < -1e-10).sum()),
                'benchmarks_unchanged':int((np.abs(delta)<=1e-10).sum()),
                'worst_benchmark':names[int(delta.argmax())], 'worst_delta':float(delta.max()),
                'regression_ge_005':{b:float(d) for b,d in zip(names,delta) if d>=.005},
                'regression_ge_03':{b:float(d) for b,d in zip(names,delta) if d>=.03},
                'alignment_faking_contribution':float(weighted[alignment].sum()/counts.sum()),
                'others_contribution':float(weighted[~alignment].sum()/counts.sum()),
                'without_alignment_delta':float(weighted[~alignment].sum()/counts[~alignment].sum()),
                'per_budget_delta':(np.array(summary['results'][method]['brier_mean'])-
                                    summary['results'][baseline]['brier_mean']).tolist()}
            summary['comparisons'][method][baseline] = comparison
    write_json(folder/'summary.json', summary)
    import pandas as pd
    metrics, bm, paired = [],[],[]
    for method in methods:
        for j,n in enumerate(BUDGETS):
            metrics.append({'method':method,'budget':n, 'pairs':len(identities),
                'brier':summary['results'][method]['brier_mean'][j],
                'accuracy_at_0_5':summary['results'][method]['accuracy_at_0_5'][j],
                'alc':summary['results'][method]['alc_mean'],
                'benchmark_equal_alc':summary['benchmark_equal_alc'][method]})
        for name in names:
            v=summary['by_benchmark'][name]
            bm.append({'benchmark':name,'method':method,'pairs':v['pairs']//len(seeds),
                       'alc':v['alc_mean'][method],
                       'delta_R25':v['alc_mean'][method]-v['alc_mean']['R25']})
    for row in rows:
        for method in methods:
            paired.append({k:row[k] for k in ('fold','benchmark','subject_id','seed','identity')} | {
                'method':method,'alc':float(np.array(row['brier'][method])@ALC_WEIGHTS),
                **{f'brier_B{n}':row['brier'][method][j] for j,n in enumerate(BUDGETS)},
                **{f'accuracy_B{n}':row['accuracy'][method][j] for j,n in enumerate(BUDGETS)}})
    pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(bm).to_csv(folder/'benchmark_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(folder/'paired_scores.csv',index=False)
    return summary
