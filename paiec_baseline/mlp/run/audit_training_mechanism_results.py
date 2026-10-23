"""Reconcile completed artifacts, matched interventions, and historical replay."""
import json
from pathlib import Path
import numpy as np
from mlp.training_mechanism_data import load_npz, write_json
from mlp.training_mechanism_training import PARAMS
from mlp.training_mechanism_guard import ROOT, freeze_core, cache_inventory, audit_completions


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    freeze_core()
    cache_inventory()
    audit_completions()
    results = {'historical_R43_replay':[], 'matching':[], 'source_identity':[]}
    for fold in range(3):
        old = read(f'runs/research_loop/R43/fold{fold}_selection.json')
        for family in ('query','episode'):
            fresh = read(ROOT/'R54'/f'fold{fold}_{family}_analysis.json')
            max_error = float(np.max(np.abs(np.asarray(old['models'][family]['pooled_inner_alc'])-fresh['curve'][:21])))
            results['historical_R43_replay'].append({'fold':fold,'family':family,'max_pooled_alc_error':max_error,
                 'historical_selected_epoch':old['models'][family]['selected_epoch'],'new_best20':fresh['selection']['best20']})
        for stage,families in [('R55',('query','episode')),('R56',('mlp',))]:
            for family in families:
                for inner in range(2):
                    folder = ROOT/stage/f'fold{fold}_{family}_inner{inner}'
                    if not (folder/'complete.json').exists():
                        continue
                    control = ROOT/'R54'/folder.name
                    h, c = read(folder/'history.json'), read(control/'history.json')
                    keys = PARAMS if family=='mlp' else PARAMS+('mean','scale','episode_only')
                    init, base = load_npz(folder/'epoch0.npz'),load_npz(control/'epoch0.npz')
                    assert all(np.array_equal(init[k],base[k]) for k in keys)
                    assert [r['steps'] for r in h] == [r['steps'] for r in c[:len(h)]]
                    config = read(folder/'config.json')
                    assert config['fit_hash'] == read(control/'config.json')['fit_hash']
                    assert config['validation_identity'] == read(control/'config.json')['validation_identity']
                    if stage == 'R56':
                        assert [r['active_data_hash'] for r in h] == [r['active_data_hash'] for r in c]
                    else:
                        assert len({r['active_data_hash'] for r in h[1:]}) == 20
                    results['matching'].append({'fit':str(folder),'init_equal':True,'reference_fit_equal':True,
                         'update_schedule_equal':True,'validation_pairs_equal':True,'epochs':len(h)-1})
        source = read(ROOT/'R54'/f'source_fold{fold}.json')
        halves = source['inner_halves']
        assert not set(halves[0])&set(halves[1])
        assert not set(halves[0]+halves[1])&set(source['heldout_benchmarks'])
        for family in ('mlp','query','episode'):
            ids0 = read(ROOT/'R54'/f'fold{fold}_{family}_inner0'/'config.json')['validation_identity']
            ids1 = read(ROOT/'R54'/f'fold{fold}_{family}_inner1'/'config.json')['validation_identity']
            assert len(ids0)+len(ids1) == source['training_pairs']
        results['source_identity'].append({'fold':fold,'pairs':source['training_pairs'],'source_halves_disjoint':True,'outer_excluded':True})
    results['source_admission'] = {}
    for family in ('mlp','query','episode'):
        folds = []
        for fold in range(3):
            source = read(ROOT/'R54'/f'source_fold{fold}.json')
            analysis = read(ROOT/'R54'/f'fold{fold}_{family}_analysis.json')
            score = analysis['scores']['best30']['results']['model']['alc_mean']
            baselines = source['baseline_summary']['results']
            deltas = {name:score-baselines[name]['alc_mean'] for name in
                      ('R25','smoothed_mean_2','training_selected_smoothed')}
            benchmark_delta = (analysis['scores']['best30']['benchmark_equal_alc']['model']-
                               source['baseline_summary']['benchmark_equal_alc']['R25'])
            folds.append({'fold':fold,'delta':deltas,'benchmark_equal_delta_R25':benchmark_delta,
                          'passes':all(d <= -1e-5 for d in deltas.values()) and benchmark_delta < 0})
        results['source_admission']['R54_'+family] = {'folds':folds,'all_pass':all(f['passes'] for f in folds)}
    for stage in ('R55','R56'):
        path = ROOT/stage/'summary.json'
        if path.exists():
            values = read(path)['folds']
            passed = all(v['gate']['passes'] for v in values)
            results['source_admission'][stage] = {'fold_pass':[v['gate']['passes'] for v in values],'all_pass':passed}
    path = ROOT/'R57'/'summary.json'
    if path.exists():
        for family,values in read(path).items():
            if values['status']=='evaluated':
                results['source_admission']['R57_'+family] = {m:all(f['gates'][m]['passes'] for f in values['folds'])
                      for m in ('ema','swa','probability_average')}
            else:
                results['source_admission']['R57_'+family] = {'status':'condition_not_met'}
    path = ROOT/'R58'/'summary.json'
    if path.exists():
        results['R58_trigger'] = read(path)['source_pool_triggers']
    write_json(ROOT/'result_audit.json', results)
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
