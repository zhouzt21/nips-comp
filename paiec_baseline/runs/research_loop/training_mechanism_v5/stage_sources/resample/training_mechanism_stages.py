"""Candidates 2--5, using the completed and frozen candidate-1 controls."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from .cohort_semantic import CachedSemantic
from .evaluate import BUDGETS, _stable_seed
from .evidence_reliability import statistics
from .reproduction_episodes import aggregate
from .training_mechanism_data import (write_json, digest, load_npz, mlp_training, mlp_validation,
    reliability_fixed, concatenate, resampled_banks, bank_load, raw_episodes, make_bank)
from .training_mechanism_training import (train, evaluate, trace_analysis, export, network,
    PARAMS, predict_mlp, predict_reliability, score_predictions, selections)
from .run.run_training_mechanism_round import ROOT, info, combined_metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fit_folders(root, fold, family):
    return [root/f'fold{fold}_{family}_inner{inner}' for inner in range(2)]


def metric(summary, name='model'):
    return summary['results'][name]['alc_mean']


def source_gate(summary, controls, source):
    target = metric(summary)
    baselines = source['baseline_summary']
    values = {name: metric(baselines, name) for name in ('R25','smoothed_mean_2','training_selected_smoothed')}
    values.update({name: metric(control) for name, control in controls.items()})
    deltas = {name: target-value for name,value in values.items()}
    benchmark_delta = summary['benchmark_equal_alc']['model']-baselines['benchmark_equal_alc']['R25']
    return {'alc': target, 'delta': deltas, 'benchmark_equal_delta_R25': benchmark_delta,
            'passes': all(d <= -1e-5 for d in deltas.values()) and benchmark_delta < 0}


def summarize_fit(root, fold, family, histories):
    counts = [history[0]['validation']['pairs'] for history in histories]
    curve = np.average([[h['validation']['results']['model']['alc_mean'] for h in history]
                        for history in histories], axis=0, weights=counts).tolist()
    selection = selections(curve, family)
    selection.pop('fixed30')
    selection.pop('best30')
    analysis = {'curve': curve, 'selection': selection, 'epochs_executed':20}
    folders = fit_folders(root, fold, family)
    epoch = analysis['selection']['best20']
    score = combined_metrics(folders, epoch)
    write_json(root/f'fold{fold}_{family}_selected.json', score)
    write_json(root/f'fold{fold}_{family}_analysis.json', analysis)
    return analysis, score['summary']


def resample():
    root = ROOT/'R55'
    engine = CachedSemantic()
    for fold in range(3):
        target = root/f'fold{fold}_selection.json'
        if target.exists():
            continue
        source = info(fold)
        halves, ridge = source['inner_halves'], source['semantic_ridge']
        names = halves[0]+halves[1]
        fixed = {name: reliability_fixed(name, engine, ridge) for name in names}
        banks, audit = {}, {}
        for name in names:
            print(json.dumps({'stage': 'resample_bank', 'fold': fold, 'benchmark': name}), flush=True)
            banks[name] = resampled_banks(name, engine, ridge)
            audit[name] = []
            for seed, path in banks[name].items():
                bank = bank_load(path)
                assert len(bank['z']) == len(fixed[name]['z'])
                assert [r['subject_id'] for r in bank['meta']] == [r['subject_id'] for r in fixed[name]['meta']]
                audit[name].append({'seed': seed, 'path': str(path), 'sha256': digest(path.read_bytes()),
                    'rows': len(bank['z']), 'pairs': int(bank['pairs']),
                    'identity_hashes': [r['identity_hash'] for r in bank['meta']],
                    'all_support_query_disjoint': all(set(r['support_ids']).isdisjoint(r['query_ids']) for r in bank['meta'])})
        write_json(root/f'fold{fold}_episode_bank.json', audit)
        summaries, analyses = {}, {}
        for family in ('query','episode'):
            histories = []
            for inner, folder in enumerate(fit_folders(root, fold, family)):
                training = concatenate([fixed[name] for name in halves[inner]])
                validation = concatenate([fixed[name] for name in halves[1-inner]])
                def epoch_data(epoch):
                    return concatenate([bank_load(banks[name][5500+epoch]) for name in halves[inner]])
                histories.append(train(family, training, validation, folder, epochs=20,
                                       episode_only=family == 'episode', epoch_data=epoch_data))
                control = read(ROOT/'R54'/folder.name/'history.json')
                assert [h['steps'] for h in histories[-1]] == [h['steps'] for h in control[:21]]
                for key in PARAMS+('mean','scale','episode_only'):
                    assert np.array_equal(load_npz(folder/'epoch0.npz')[key],
                                          load_npz(ROOT/'R54'/folder.name/'epoch0.npz')[key])
                del training, validation
            analyses[family], summaries[family] = summarize_fit(root, fold, family, histories)
        control = read(ROOT/'R54'/f'fold{fold}_query_best20.json')['summary']
        gate = source_gate(summaries['query'], {'fixed_query': control, 'resampled_episode': summaries['episode']}, source)
        permutation = permutation_audit(root, fold, analyses['query']['selection']['best20'], fixed, engine, source)
        gate['correspondence_net_gain'] = permutation['difference_of_gains']
        gate['passes'] = bool(gate['passes'] and permutation['difference_of_gains'] < 0)
        write_json(target, {'fold': fold, 'models': summaries, 'gate': gate,
                            'permutation': permutation, 'epochs': {k:v['selection']['best20'] for k,v in analyses.items()}})
        print(json.dumps({'stage': 'resample_complete', 'fold': fold, 'gate': gate}), flush=True)
        del fixed, banks
        gc.collect()
    write_json(root/'summary.json', {'folds': [read(root/f'fold{f}_selection.json') for f in range(3)]})


def permutation_audit(root, fold, epoch, fixed, engine, source):
    true_rows, true_baseline, perm_rows, perm_baseline = [], [], [[] for _ in range(4)], [[] for _ in range(4)]
    zero = export(network('query'), {'mean': np.zeros(16), 'scale': np.ones(16), 'episode_only': np.array(False)})
    for inner, folder in enumerate(fit_folders(root, fold, 'query')):
        artifact = load_npz(folder/f'epoch{epoch}.npz')
        for name in source['inner_halves'][1-inner]:
            data = fixed[name]
            true_rows.extend(evaluate('query', data, artifact)[0])
            true_baseline.extend(evaluate('query', data, zero)[0])
            episodes = list(raw_episodes(name, engine, source['semantic_ridge']))
            for repeat in range(4):
                bank = make_bank(episodes)
                for ep, meta in zip(episodes, bank['meta']):
                    rng = np.random.default_rng(_stable_seed(5573+repeat, name, ep['subject_id']))
                    for j,n in enumerate(BUDGETS[2:]):
                        labels = ep['labels'][:n][rng.permutation(n)]
                        z,b,d = statistics(ep['support'][:n], ep['query'], labels)
                        sl = slice(meta['start']+j*meta['q'], meta['start']+(j+1)*meta['q'])
                        bank['z'][sl], bank['base'][sl], bank['delta'][sl] = z,b,d
                perm_rows[repeat].extend(evaluate('query', bank, artifact)[0])
                perm_baseline[repeat].extend(evaluate('query', bank, zero)[0])
    true, baseline = aggregate(true_rows), aggregate(true_baseline)
    shuffled = [aggregate(rows) for rows in perm_rows]
    shuffled_base = [aggregate(rows) for rows in perm_baseline]
    true_gain = metric(true)-metric(baseline)
    perm_gains = [metric(a)-metric(b) for a,b in zip(shuffled, shuffled_base)]
    assert true['results']['model']['brier_mean'][:2] == baseline['results']['model']['brier_mean'][:2]
    for summary in shuffled:
        assert summary['results']['model']['brier_mean'][:2] == true['results']['model']['brier_mean'][:2]
    result = {'seeds': list(range(5573,5577)), 'true_model': true, 'true_R25': baseline,
              'permuted_models': shuffled, 'permuted_R25': shuffled_base,
              'true_gain': true_gain, 'permuted_gains': perm_gains,
              'difference_of_gains': float(true_gain-np.mean(perm_gains)), 'B0_B1_unchanged': True}
    write_json(root/f'fold{fold}_permutation_rows.json', {'true': true_rows, 'R25': true_baseline,
               'permuted': perm_rows, 'permuted_R25': perm_baseline})
    return result


def benchmark_risk():
    root = ROOT/'R56'
    engine = CachedSemantic()
    for fold in range(3):
        target = root/f'fold{fold}_selection.json'
        if target.exists():
            continue
        source = info(fold)
        halves = source['inner_halves']
        data = {name: mlp_validation(name, engine) for name in sum(halves, [])}
        histories = []
        for inner, folder in enumerate(fit_folders(root, fold, 'mlp')):
            training = mlp_training(halves[inner])
            validation = [ep for name in halves[1-inner] for ep in data[name]]
            histories.append(train('mlp', training, validation, folder, risk='mixture'))
            old = read(ROOT/'R54'/folder.name/'history.json')
            assert [h['steps'] for h in histories[-1]] == [h['steps'] for h in old]
            assert read(folder/'config.json')['fit_hash'] == read(ROOT/'R54'/folder.name/'config.json')['fit_hash']
            for key in PARAMS:
                assert np.array_equal(load_npz(folder/'epoch0.npz')[key], load_npz(ROOT/'R54'/folder.name/'epoch0.npz')[key])
            del training, validation
        analysis = trace_analysis(histories, 'mlp')
        epoch = analysis['selection']['best30']
        score = combined_metrics(fit_folders(root, fold, 'mlp'), epoch)
        control = read(ROOT/'R54'/f'fold{fold}_mlp_best30.json')['summary']
        gate = source_gate(score['summary'], {'pair_risk': control}, source)
        gate['benchmark_equal_delta_pair_risk'] = score['summary']['benchmark_equal_alc']['model']-control['benchmark_equal_alc']['model']
        gate['passes'] = bool(gate['passes'] and gate['benchmark_equal_delta_pair_risk'] <= 0)
        write_json(root/f'fold{fold}_selected.json', score)
        write_json(target, {'fold': fold, 'epoch': epoch, 'analysis': analysis,
                            'summary': score['summary'], 'gate': gate})
        print(json.dumps({'stage': 'benchmark_risk_complete', 'fold': fold, 'gate': gate}), flush=True)
        del data, histories
        gc.collect()
    write_json(root/'summary.json', {'folds': [read(root/f'fold{f}_selection.json') for f in range(3)]})


def averages():
    root = ROOT/'R57'
    engine = CachedSemantic()
    result = {}
    for family in ('mlp','query','episode'):
        analyses = [read(ROOT/'R54'/f'fold{fold}_{family}_analysis.json') for fold in range(3)]
        windows = [a['selection']['stable_window'] for a in analyses]
        if any(w is None for w in windows):
            result[family] = {'status': 'condition_not_met', 'windows': windows}
            continue
        folds = []
        for fold, window in enumerate(windows):
            source = info(fold)
            rows = {method: [] for method in ('ema','swa','probability_average','window_best')}
            best_epoch = min(window, key=lambda epoch: analyses[fold]['curve'][epoch])
            for inner, folder in enumerate(fit_folders(ROOT/'R54', fold, family)):
                names = source['inner_halves'][1-inner]
                validation = ([ep for name in names for ep in mlp_validation(name, engine)] if family == 'mlp'
                    else concatenate([reliability_fixed(name, engine, source['semantic_ridge']) for name in names]))
                snapshots = [load_npz(folder/f'epoch{epoch}.npz') for epoch in window]
                averaged = {}
                for method in ('ema','swa'):
                    artifact = {k:v.copy() for k,v in snapshots[0].items()}
                    for key in PARAMS:
                        if method == 'swa':
                            artifact[key] = np.mean([a[key].astype(float) for a in snapshots],axis=0).astype(np.float32)
                        else:
                            p = snapshots[0][key].astype(float).copy()
                            for a in snapshots[1:]:
                                p = .9*p+.1*a[key]
                            artifact[key] = p.astype(np.float32)
                    averaged[method] = artifact
                    out = root/f'fold{fold}_{family}_inner{inner}'
                    out.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(out/f'{method}.npz', **artifact)
                    rows[method].extend(evaluate(family, validation, artifact)[0])
                rows['window_best'].extend(evaluate(family, validation, snapshots[window.index(best_epoch)])[0])
                pred = [predict_mlp(validation, a) if family == 'mlp' else predict_reliability(validation, a) for a in snapshots]
                mean_pred = [[np.mean([p[i][j] for p in pred],axis=0) for j in range(6)] for i in range(len(pred[0]))]
                metadata = validation if family == 'mlp' else validation['meta']
                rows['probability_average'].extend(score_predictions(metadata, mean_pred)[0])
                del validation
            summaries = {method: aggregate(r) for method,r in rows.items()}
            control = read(ROOT/'R54'/f'fold{fold}_{family}_best30.json')['summary']
            gates = {m: source_gate(s, {'single_best':control, 'window_best':summaries['window_best']}, source)
                     for m,s in summaries.items() if m != 'window_best'}
            for m,g in gates.items():
                g['benchmark_equal_delta_single'] = summaries[m]['benchmark_equal_alc']['model']-control['benchmark_equal_alc']['model']
                g['passes'] = bool(g['passes'] and g['benchmark_equal_delta_single'] < 0)
            write_json(root/f'fold{fold}_{family}_scores.json', {'rows':rows,'summaries':summaries,'gates':gates,
                                                               'window':window,'window_best':best_epoch})
            folds.append({'fold':fold,'summaries':summaries,'gates':gates})
        result[family] = {'status':'evaluated','windows':windows,'folds':folds}
    write_json(root/'summary.json', result)
    print(json.dumps({'stage':'averages_complete','families':{k:v['status'] for k,v in result.items()}}),flush=True)


def initialization():
    root = ROOT/'R58'
    analyses = [read(ROOT/'R54'/f'fold{fold}_mlp_analysis.json') for fold in range(3)]
    triggers = [all(d['initialization_trigger'] for d in a['directions']) for a in analyses]
    diagnosis = {'qualifying_source_pools': sum(triggers),'source_pool_triggers':triggers,
                 'diagnostics':[a['directions'] for a in analyses]}
    if sum(triggers) < 2:
        write_json(root/'summary.json', {**diagnosis,'status':'condition_not_met','fits_executed':0})
        print(json.dumps({'stage':'initialization_complete','status':'condition_not_met','triggers':triggers}),flush=True)
        return
    engine = CachedSemantic()
    folds = []
    for fold in range(3):
        source = info(fold)
        halves = source['inner_halves']
        data = {name:mlp_validation(name,engine) for name in sum(halves,[])}
        histories = []
        for inner,folder in enumerate(fit_folders(root,fold,'mlp')):
            training = mlp_training(halves[inner])
            validation = [ep for name in halves[1-inner] for ep in data[name]]
            histories.append(train('mlp',training,validation,folder,output_scale=.1))
        analysis = trace_analysis(histories,'mlp')
        score = combined_metrics(fit_folders(root,fold,'mlp'),analysis['selection']['best30'])
        control = read(ROOT/'R54'/f'fold{fold}_mlp_best30.json')['summary']
        gate = source_gate(score['summary'],{'default_initialization':control},source)
        write_json(root/f'fold{fold}_scores.json',{'analysis':analysis,'scores':score,'gate':gate})
        folds.append({'fold':fold,'summary':score['summary'],'gate':gate})
        del data,histories
        gc.collect()
    write_json(root/'summary.json',{**diagnosis,'status':'evaluated','fits_executed':6,'folds':folds})
