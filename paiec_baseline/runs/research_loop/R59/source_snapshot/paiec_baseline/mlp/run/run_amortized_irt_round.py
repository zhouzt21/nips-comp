"""R52 fixed author-calibration source admission; never runs outer evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from mlp.cohort_semantic import CachedSemantic
from mlp.irt_training import IncompleteIRTFit, low_budget_gate, train_amortized_irt
from mlp.reproduction_episodes import aggregate, score_baselines, source_fold
from mlp.reproduction_training import evaluate_checkpoint, merge_rows, write_json


def source_paths():
    paths = ['mlp/amortized_irt.py', 'mlp/irt_training.py', 'mlp/run/run_amortized_irt_round.py',
             'mlp/reproduction_episodes.py', 'mlp/reproduction_training.py', 'mlp/neural_process.py',
             '../research_loop/R52_protocol.md', '../research_loop/REPRODUCTION_V4_SPEC.md',
             '../research_loop/literature/reproduction_v4/amortized_irt.md',
             '../research_loop/literature/reproduction_v4/amortized_irt_sources.json',
             'mlp/evaluate.py', 'mlp/model.py', 'mlp/metadata_features.py', 'mlp/cohort_semantic.py',
             'mlp/semantic_features.py', 'mlp/structured_text.py', 'mlp/run/run_metadata_round.py',
             'mlp/local_ridge.py', '../doc/复现与组合.md', 'D:/paiec_payload/data/corpus_manifest.json',
             '../tests/test_amortized_irt.py', '../tests/test_irt_training.py']
    for fold in range(3):
        paths.extend([f'runs/research_loop/R25/fold{fold}_selection.json',
                      f'runs/research_loop/R43/fold{fold}_selection.json',
                      f'runs/mlp_failure_2026_10_22/stat_prior/fold{fold}_prior.json',
                      f'runs/mlp_cv/fold{fold}_split.json'])
    manifest = json.loads(Path('../research_loop/literature/reproduction_v4/amortized_irt_sources.json').read_text(encoding='utf-8-sig'))
    for source in manifest['artifacts']:
        path = Path(manifest['source_root'])/source['relative_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise RuntimeError(f'Pinned paper/code source changed: {path}')
        paths.append(str(path))
    return paths


def merge_ablation(candidate, ablation):
    lookup = {(row['benchmark'], row['subject_id'], row['seed']): row for row in ablation}
    result = []
    for row in candidate:
        match = lookup[(row['benchmark'], row['subject_id'], row['seed'])]
        rename = lambda name: name.replace('irt_calibration', 'irt_zero_subject_prior', 1)
        result.append({**row,
                       'brier': {**row['brier'], **{rename(k): v for k, v in match['brier'].items()}},
                       'accuracy': {**row['accuracy'], **{rename(k): v for k, v in match['accuracy'].items()}},
                       'ablation_decomposition': match['decomposition']})
    return result


def verify_completed_fits(selection):
    for folder in selection['fit_folders']:
        folder = Path(folder)
        summary = json.loads((folder/'fit_summary.json').read_text())
        if summary.get('status') != 'complete':
            raise IncompleteIRTFit(f'Previously selected fit is incomplete: {folder}')
        for name, digest in summary['checkpoint_sha256'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Previously selected checkpoint changed: {folder/name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/research_loop/R52')
    args = parser.parse_args(); root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); torch.set_num_threads(4)
    paths = source_paths()
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}
    if (root/'provenance.json').exists():
        if json.loads((root/'provenance.json').read_text()) != hashes:
            raise RuntimeError('Source changed: use a new R52 run directory')
    else:
        write_json(root/'provenance.json', hashes)
        (root/'code').mkdir(exist_ok=True)
        source_manifest = json.loads(Path('../research_loop/literature/reproduction_v4/amortized_irt_sources.json').read_text(encoding='utf-8-sig'))
        external_root = Path(source_manifest['source_root']).resolve()
        snapshot_manifest = []
        for index, path in enumerate(paths):
            source = Path(path).resolve()
            external = source == external_root or external_root in source.parents
            destination = None if external else root/'code'/f'{index:02d}_{Path(path).name}'
            if destination is not None:
                destination.write_bytes(source.read_bytes())
            snapshot_manifest.append({'source': path, 'sha256': hashes[path],
                                      'snapshot': str(destination) if destination is not None else None,
                                      'external_source_reference_only': external})
        write_json(root/'snapshot_manifest.json', {
            'policy': 'snapshot project dependencies only; external papers and author repositories remain at their pinned source_root',
            'external_source_root': str(external_root), 'artifacts': snapshot_manifest})
    engine = CachedSemantic(); folds = []
    for fold in range(3):
        selection_path = root/f'fold{fold}_selection.json'
        if selection_path.exists():
            selection = json.loads(selection_path.read_text()); verify_completed_fits(selection)
            folds.append(selection); continue
        data, info = source_fold(fold, engine); halves = info['inner_halves']
        all_episodes = [episode for name in sorted(data) for episode in data[name]]
        baselines = score_baselines(all_episodes, info['semantic_ridge'], info['selected_alpha'])
        fingerprint = hashlib.sha256(json.dumps({'cache': info['cache_fingerprint'], 'sources': hashes},
                                               sort_keys=True).encode()).hexdigest()
        rows = []; fit_summaries = []; fit_folders = []
        for inner in range(2):
            fit = [ep for benchmark in halves[inner] for ep in data[benchmark]]
            validation = [ep for benchmark in halves[1-inner] for ep in data[benchmark]]
            folder = root/f'fold{fold}_irt_calibration_inner{inner}'
            try:
                fit_summary = train_amortized_irt(fit, validation, folder, seed=5207+100*fold+inner,
                                                   source_fingerprint=fingerprint)
            except IncompleteIRTFit as exc:
                write_json(root/'resource_stop.json', {'status': 'incomplete', 'fold': fold, 'inner': inner,
                           'reason': str(exc), 'elapsed_seconds': time.monotonic()-started,
                           'admit_outer': False, 'not_a_completed_negative_result': True})
                raise
            candidate = evaluate_checkpoint(folder/'final.pt', validation)
            ablation = evaluate_checkpoint(folder/'zero_subject_prior.pt', validation)
            combined = merge_ablation(candidate, ablation)
            max_change = max(detail['permutation_mean_absolute_change']
                             for row in combined for detail in row['decomposition'])
            if max_change > 1e-7:
                raise AssertionError(f'Rasch support-label permutation unexpectedly changed predictions: {max_change}')
            rows.extend(combined); fit_summaries.append(fit_summary); fit_folders.append(str(folder))
        candidate_rows = rows
        rows = merge_rows(baselines, rows)
        ablation_details = {(row['benchmark'], row['subject_id'], row['seed']): row['ablation_decomposition']
                            for row in candidate_rows}
        for row in rows:
            row['ablation_decomposition'] = ablation_details[(row['benchmark'], row['subject_id'], row['seed'])]
        summary = aggregate(rows); gate = low_budget_gate(summary)
        selection = {'fold': fold, 'model_name': 'irt_calibration',
                     'source_benchmarks': info['train_benchmarks'], 'heldout_benchmarks': info['heldout_benchmarks'],
                     'inner_halves': halves, 'pairs': len(all_episodes),
                     'source_replay_error': info['replay_error'], 'source_fingerprint': fingerprint,
                     'fit_folders': fit_folders, 'fit_summaries': fit_summaries,
                     'checkpoint_rule': 'final fit-only convergence; no validation epoch selection',
                     'summary': summary, 'gate': gate, 'source_pool_admitted': gate['admitted'],
                     'permutation_expected': 'invariant sufficient statistic; not an admission improvement test'}
        write_json(root/f'fold{fold}_irt_calibration_pair_scores.json', rows)
        write_json(selection_path, selection); folds.append(selection)
        print(json.dumps({'fold': fold, 'gate': gate, 'alc': summary['results']['irt_calibration']['alc_mean']}), flush=True)
        del data, all_episodes, baselines, rows, candidate_rows
    admission = all(fold['source_pool_admitted'] for fold in folds)
    summary = {'scope': 'nested source benchmark development; reused public data; not outer scores',
               'model_name': 'irt_calibration', 'admit_outer': {'irt_calibration': admission},
               'outer_scores_produced': False, 'combination_attempted': False,
               'folds': folds, 'elapsed_seconds': time.monotonic()-started}
    for path in paths:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != hashes[path]:
            raise RuntimeError(f'Source changed during R52: {path}')
    write_json(root/'summary.json', summary)
    print(json.dumps({'admit_outer': summary['admit_outer'], 'elapsed_seconds': summary['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
