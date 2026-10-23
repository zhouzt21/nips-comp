"""R51 source-only admission; do not run outer scores when the gate fails."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mlp.cohort_semantic import CachedSemantic
from mlp.evaluate import ALC_WEIGHTS
from mlp.reproduction_episodes import aggregate, score_baselines, source_fold
from mlp.reproduction_training import evaluate_checkpoint, merge_rows, train_neural_process, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/research_loop/R51')
    args = parser.parse_args(); root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    started = time.time(); torch.set_num_threads(4)
    paths = ['mlp/neural_process.py', 'mlp/reproduction_episodes.py', 'mlp/reproduction_training.py',
             'mlp/run/run_neural_process_round.py', '../research_loop/R51_protocol.md',
             '../research_loop/literature/reproduction_v4/cnp_anp.md',
             'mlp/evaluate.py', 'mlp/model.py', 'mlp/metadata_features.py', 'mlp/cohort_semantic.py',
             'mlp/semantic_features.py', 'mlp/structured_text.py', 'mlp/run/run_metadata_round.py',
             'mlp/local_ridge.py', '../doc/复现与组合.md', 'D:/paiec_payload/data/corpus_manifest.json']
    for fold in range(3):
        paths.extend([f'runs/research_loop/R25/fold{fold}_selection.json',
                      f'runs/research_loop/R43/fold{fold}_selection.json',
                      f'runs/mlp_failure_2026_10_22/stat_prior/fold{fold}_prior.json',
                      f'runs/mlp_cv/fold{fold}_split.json'])
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}
    if (root/'provenance.json').exists():
        assert json.loads((root/'provenance.json').read_text()) == hashes, 'Source changed: use a new run directory'
    else:
        write_json(root/'provenance.json', hashes)
        (root/'code').mkdir(exist_ok=True)
        for index, path in enumerate(paths):
            (root/'code'/f'{index:02d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    engine = CachedSemantic(); folds = []
    for fold in range(3):
        selection_path = root/f'fold{fold}_selection.json'
        if selection_path.exists():
            folds.append(json.loads(selection_path.read_text())); continue
        data, info = source_fold(fold, engine); halves = info['inner_halves']
        all_episodes = [episode for name in sorted(data) for episode in data[name]]
        baselines = score_baselines(all_episodes, info['semantic_ridge'], info['selected_alpha'])
        selection = {'fold': fold, 'source_benchmarks': info['train_benchmarks'],
                     'heldout_benchmarks': info['heldout_benchmarks'], 'inner_halves': halves,
                     'pairs': len(all_episodes), 'source_replay_error': info['replay_error'],
                     'source_fingerprint': info['cache_fingerprint'], 'models': {}}
        for name in ('cnp', 'attentive_cnp'):
            histories, counts = [], []
            for inner in range(2):
                fit = [ep for benchmark in halves[inner] for ep in data[benchmark]]
                validation = [ep for benchmark in halves[1-inner] for ep in data[benchmark]]
                history = train_neural_process(fit, validation, name, root/f'fold{fold}_{name}_inner{inner}',
                                              seed=5107+100*fold+inner, source_fingerprint=info['cache_fingerprint'])
                histories.append(history); counts.append(len(validation))
            curves = np.average([[row['validation_brier'] for row in history] for history in histories],
                                weights=counts, axis=0)
            alcs = curves@ALC_WEIGHTS; epoch = int(np.argmin(alcs)); rows = []
            checkpoints = []
            for inner in range(2):
                checkpoint = root/f'fold{fold}_{name}_inner{inner}'/f'epoch_{epoch:03d}.pt'
                validation = [ep for benchmark in halves[1-inner] for ep in data[benchmark]]
                rows.extend(evaluate_checkpoint(checkpoint, validation))
                checkpoints.append(str(checkpoint))
            rows = merge_rows(baselines, rows); summary = aggregate(rows)
            observed = summary['results'][name]['alc_mean']
            assert abs(observed-alcs[epoch]) < 2e-7
            comparisons = {baseline: observed-summary['results'][baseline]['alc_mean']
                           for baseline in ('R25', 'smoothed_mean_2', 'training_selected_smoothed', name+'_permuted')}
            benchmark_delta = summary['benchmark_equal_alc'][name]-summary['benchmark_equal_alc']['R25']
            admitted = all(value < 0 for value in comparisons.values()) and benchmark_delta < 0
            selection['models'][name] = {'selected_epoch': epoch, 'pooled_inner_alc': alcs.tolist(),
                                         'summary': summary, 'comparisons': comparisons,
                                         'benchmark_equal_delta_R25': benchmark_delta,
                                         'source_pool_admitted': admitted, 'checkpoints': checkpoints}
            write_json(root/f'fold{fold}_{name}_selected_pair_scores.json', rows)
            print(json.dumps({'fold': fold, 'model': name, 'selected_epoch': epoch,
                              'alc': observed, 'deltas': comparisons, 'admit_pool': admitted}), flush=True)
        write_json(selection_path, selection); folds.append(selection)
        del data, all_episodes, baselines
    admissions = {name: all(fold['models'][name]['source_pool_admitted'] for fold in folds)
                  for name in ('cnp', 'attentive_cnp')}
    write_json(root/'summary.json', {'scope': 'nested source benchmark development; reused public data; not outer scores',
                                    'admit_outer': admissions, 'folds': folds,
                                    'elapsed_seconds': time.time()-started})
    for path in paths: assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == hashes[path]
    print(json.dumps({'admit_outer': admissions, 'elapsed_seconds': time.time()-started}), flush=True)


if __name__ == '__main__': main()
