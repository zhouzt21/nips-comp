"""R59: replay frozen predictions and audit low-budget complementarity; no fits."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from mlp.cohort_semantic import CachedSemantic
from mlp.evaluate import ALC_WEIGHTS, BUDGETS, _load_directory, _vectors
from mlp.model import forward_batch
from mlp.reproduction_episodes import (DATA_ROOT, DATA_REVISION, source_episodes,
    baseline_predictions, brier, accuracy, aggregate)
from mlp.reproduction_training import decomposition
from mlp.training_mechanism_data import mlp_validation, load_npz, write_json
from mlp.training_mechanism_training import PARAMS, predict_mlp

OUT = Path('runs/research_loop/R59')
OLD = Path('runs/research_loop/training_mechanism_v5')
SPEC = Path('../research_loop/LOW_BUDGET_PRIOR_COMPOSITION_SPEC.md')
DECOMP = ['outcome_variance', 'mean_bias_squared', 'prediction_variance', 'minus_twice_covariance']
BASES = ['R25', 'smoothed_mean_2', 'training_selected_smoothed', 'constant_0.5']
MODELS = ['R54', 'R56']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    old = read(OLD/'core_provenance.json')['sha256']
    for p, expected in old.items():
        assert sha(p) == expected, f'Historical source changed: {p}'
    paths = list(Path('mlp').rglob('*.py')) + [SPEC, Path('../doc/plan_3.md')]
    paths += list(Path('runs/research_loop/reproduction_v4').glob('source_fold*.json'))
    paths += [OLD/'environment.json', OLD/'core_provenance.json']
    for fold in range(3):
        paths += [OLD/'R54'/f'fold{fold}_mlp_analysis.json', OLD/'R56'/f'fold{fold}_selection.json']
        for method in MODELS:
            epoch = selected_epoch(method, fold)
            for inner in range(2):
                folder = OLD/method/f'fold{fold}_mlp_inner{inner}'
                paths += [folder/f'epoch{epoch}.npz', folder/f'metrics{epoch}.json', folder/'config.json']
    hashes = {str(p): sha(p) for p in sorted(set(paths))}
    provenance = OUT/'provenance.json'
    if provenance.exists():
        assert read(provenance)['sha256'] == hashes, 'R59 contract changed; use a new run version'
    else:
        for p in paths:
            target = OUT/'source_snapshot'/p.resolve().relative_to(Path.cwd().parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
        write_json(provenance, {'sha256': hashes, 'captured_utc': datetime.now(timezone.utc).isoformat(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'python': sys.version, 'platform': platform.platform(), 'data_revision': DATA_REVISION,
            'command': 'powershell -NoProfile -ExecutionPolicy Bypass -File D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1 -m mlp.run.run_low_budget_prior',
            'cpu_threads': 4, 'new_fits': 0, 'source_selection_bias': True,
            'environment_reference': str(OLD/'environment.json')})
    return hashes


def selected_epoch(method, fold):
    if method == 'R54':
        return read(OLD/method/f'fold{fold}_mlp_analysis.json')['selection']['best30']
    return read(OLD/method/f'fold{fold}_selection.json')['epoch']


def align(name, mlp_eps, semantic_eps):
    """Recover cached MLP row order and map R25 item probabilities to those rows."""
    responses, subjects, items = _load_directory(DATA_ROOT, name)
    groups = {str(sid): (sid, g) for sid, g in responses.groupby('subject_id', sort=False)}
    sem = {ep['subject_id']: ep for ep in semantic_eps}
    output = []
    for ep in mlp_eps:
        other = sem[ep['subject_id']]
        assert ep['support_ids'] == other['support_item_ids']
        assert ep['query_ids'] == other['query_item_ids']
        assert set(ep['support_ids']).isdisjoint(ep['query_ids'])
        assert np.array_equal(ep['labels'], other['labels'])
        sid, group = groups[ep['subject_id']]
        query = group[group.item_id.astype(str).isin(ep['query_ids'])].copy()
        query['_variant'] = query.interactors.map(str)
        keys = ['item_id', '_variant']
        ag = query.groupby(keys, sort=False, dropna=False).response.agg(['size', 'sum'])
        reps = query.drop_duplicates(keys).set_index(keys).loc[ag.index].reset_index()
        assert np.array_equal(ag['size'].to_numpy(float), ep['counts'])
        assert np.array_equal(ag['sum'].to_numpy(float), ep['successes'])
        assert np.array_equal(_vectors(subjects[sid], items, reps), ep['query'])
        lookup = {item: i for i, item in enumerate(ep['query_ids'])}
        order = np.array([lookup[str(item)] for item in reps.item_id], dtype=int)
        for i in range(len(lookup)):
            assert ep['counts'][order == i].sum() == other['counts'][i]
            assert ep['successes'][order == i].sum() == other['successes'][i]
        output.append((ep, other, order,
            [{'item_id': str(i), 'interactor_variant_sha256': hashlib.sha256(str(v).encode()).hexdigest()}
             for i, v in zip(reps.item_id, reps['_variant'])]))
    return output


def bootstrap(deltas, benchmarks):
    names = sorted(set(benchmarks))
    blocks = [deltas[np.asarray(benchmarks) == n] for n in names]
    sums = np.array([a.sum() for a in blocks]); sizes = np.array([len(a) for a in blocks])
    rng = np.random.default_rng(5907)
    indices = rng.integers(0, len(names), (2000, len(names)))
    pair = sums[indices].sum(1)/sizes[indices].sum(1)
    bm = (sums/sizes)[indices].mean(1)
    return {'pair_ci95': np.quantile(pair, [.025, .975]).tolist(),
            'benchmark_equal_ci95': np.quantile(bm, [.025, .975]).tolist(),
            'seed': 5907, 'replicates': 2000, 'clusters': len(names),
            'interpretation': 'descriptive after adaptive source reuse, not confirmation'}


def correlations(frame):
    output = []
    signals = ['confidence', 'prediction_variance', 'model_disagreement', 'first_support_sqerr']
    for (fold, method, budget), group in frame[frame.method.isin(MODELS) & frame.budget.isin([0, 1])].groupby(['fold', 'method', 'budget']):
        base = frame[(frame.fold == fold) & (frame.method == 'R25') & (frame.budget == budget)]
        group = group.merge(base[['benchmark', 'subject_id', 'brier']], on=['benchmark', 'subject_id'], suffixes=('', '_R25'))
        group['gain'] = group.brier_R25 - group.brier
        for level, values in [('pair', group), ('benchmark', group.groupby('benchmark')[signals+['gain']].mean())]:
            for signal in signals:
                if signal == 'first_support_sqerr' and budget == 0:
                    continue
                rho = None
                if values[signal].nunique() > 1 and values.gain.nunique() > 1:
                    rho = float(spearmanr(values[signal], values.gain).statistic)
                output.append({'fold': int(fold), 'method': method, 'budget': int(budget),
                    'level': level, 'signal': signal, 'rho_gain': rho, 'n': len(values)})
    return pd.DataFrame(output)


def fold_replay(fold, engine):
    source = read(Path('runs/research_loop/reproduction_v4')/f'source_fold{fold}.json')
    halves = source['inner_halves']
    assert set(halves[0]).isdisjoint(halves[1])
    assert set(sum(halves, [])).isdisjoint(source['heldout_benchmarks'])
    all_rows, detail, meta, arrays, replay = [], [], [], {}, []
    start = 0
    for inner in range(2):
        artifacts = {m: load_npz(OLD/m/f'fold{fold}_mlp_inner{inner}'/f'epoch{selected_epoch(m, fold)}.npz') for m in MODELS}
        references = {m: {(r['benchmark'], r['subject_id']): r for r in
            read(OLD/m/f'fold{fold}_mlp_inner{inner}'/f'metrics{selected_epoch(m, fold)}.json')['rows']} for m in MODELS}
        for name in halves[1-inner]:
            eps = mlp_validation(name, engine)
            sem = source_episodes(name, engine)
            if not eps:
                continue
            aligned = align(name, eps, sem)
            predictions = {m: predict_mlp(eps, artifacts[m]) for m in MODELS}
            for i, (ep, other, order, identities) in enumerate(aligned):
                counts, successes = ep['counts'], ep['successes']
                raw = {m: np.asarray(predictions[m][i]) for m in MODELS}
                for b in BASES:
                    raw[b] = np.asarray([baseline_predictions(other, n, source['semantic_ridge'], source['selected_alpha'])[b][order] for n in BUDGETS])
                for m in MODELS:
                    raw[m+'_segment'] = np.concatenate([raw[m][:2], raw['R25'][2:]])
                # Oracle uses evaluation labels: diagnostic ceiling only, never deployable.
                low_loss = {m: sum(w*brier(p, counts, successes) for w, p in zip(ALC_WEIGHTS[:2], raw[m][:2])) for m in ['R25']+MODELS}
                best = min(low_loss, key=low_loss.get)
                raw['pair_label_oracle'] = np.concatenate([raw[best][:2], raw['R25'][2:]])
                identifier = {'fold': fold, 'inner': inner, 'benchmark': name, 'subject_id': ep['subject_id'], 'seed': 73}
                row = {**identifier, 'brier': {}, 'accuracy': {}}
                disagreement = float(counts@((raw['R54'][0]-raw['R56'][0])**2)/counts.sum())
                for method, probs in raw.items():
                    row['brier'][method] = [brier(p, counts, successes) for p in probs]
                    row['accuracy'][method] = [accuracy(p, counts, successes) for p in probs]
                    if method in MODELS:
                        old = references[method][(name, ep['subject_id'])]
                        error = float(np.max(np.abs(np.asarray(old['brier']['model'])-row['brier'][method])))
                        accerr = float(np.max(np.abs(np.asarray(old['accuracy']['model'])-row['accuracy'][method])))
                        assert max(error, accerr) < 1e-7, (identifier, method, error, accerr)
                        replay.append({**identifier, 'method': method, 'brier_error': error, 'accuracy_error': accerr})
                    for j, (budget, p) in enumerate(zip(BUDGETS, probs)):
                        terms = decomposition(p, counts, successes)
                        pmean = float(counts@p/counts.sum())
                        first_error = 0.
                        if method in MODELS:
                            ps = float(forward_batch(ep['support'][:1], tuple(artifacts[method][k] for k in PARAMS))[0])
                            first_error = float((ps-ep['labels'][0])**2)
                        detail.append({**identifier, 'method': method, 'budget': int(budget),
                            'brier': row['brier'][method][j], 'accuracy_at_0_5': row['accuracy'][method][j],
                            'responses': int(counts.sum()), 'query_items': len(ep['query_ids']),
                            'response_groups': len(counts), 'mean_p': pmean,
                            'mean_y': float(successes.sum()/counts.sum()), **terms,
                            'variation_cost': terms['prediction_variance']+terms['minus_twice_covariance'],
                            'confidence': abs(pmean-.5), 'model_disagreement': disagreement,
                            'first_support_sqerr': first_error})
                all_rows.append(row)
                meta.append({**identifier, 'start': start, 'size': len(counts), 'support_ids': ep['support_ids'],
                    'support_labels': ep['labels'].tolist(), 'query_groups': identities,
                    'selected_epochs': {m: selected_epoch(m, fold) for m in MODELS}, 'oracle_selected': best})
                start += len(counts)
                for key, value in {'counts': counts, 'successes': successes, **{m: p.T for m, p in raw.items()}}.items():
                    arrays.setdefault(key, []).append(value)
    summary = aggregate(all_rows)
    expected = source['baseline_summary']
    for method in BASES:
        error = np.max(np.abs(np.asarray(summary['results'][method]['brier_mean'])-expected['results'][method]['brier_mean']))
        assert error < 1e-7, (fold, method, error)
    np.savez_compressed(OUT/f'fold{fold}_predictions.npz', **{k: np.concatenate(v) for k,v in arrays.items()})
    write_json(OUT/f'fold{fold}_prediction_index.json', meta)
    write_json(OUT/f'fold{fold}_paired_scores.json', all_rows)
    write_json(OUT/f'fold{fold}_summary.json', summary)
    return all_rows, detail, replay


def analysis(all_rows, detail):
    frame = pd.DataFrame(detail)
    frame.to_csv(OUT/'pair_budget_decomposition.csv', index=False)
    numeric = ['brier', 'accuracy_at_0_5', 'mean_p', 'mean_y']+DECOMP+['variation_cost']
    bm = frame.groupby(['fold','method','benchmark','budget'])[numeric].mean().reset_index()
    bm.to_csv(OUT/'benchmark_budget_decomposition.csv', index=False)
    frame.groupby(['fold','method','budget'])[numeric].mean().reset_index().to_csv(OUT/'pool_budget_decomposition.csv', index=False)
    corr = correlations(frame)
    corr.to_csv(OUT/'reliability_signal_correlations.csv', index=False)
    contrasts = []
    for method, baseline in [('R54','R25'),('R56','R25'),('R56','R54')]:
        keys = ['fold','inner','benchmark','subject_id','seed','budget']
        left = frame[frame.method == method][keys+numeric]
        right = frame[frame.method == baseline][keys+numeric]
        merged = left.merge(right, on=keys, suffixes=('', '_ref'), validate='one_to_one')
        rows = merged[keys].copy(); rows['method'] = method; rows['baseline'] = baseline
        for k in numeric:
            rows['delta_'+k] = merged[k]-merged[k+'_ref']
        assert np.max(abs(rows['delta_brier']-sum(rows['delta_'+k] for k in DECOMP))) < 1e-10
        contrasts.append(rows)
    pd.concat(contrasts).to_csv(OUT/'paired_decomposition_deltas.csv', index=False)
    outcomes, metrics, slices, opportunities = [], [], [], []
    for fold in range(3):
        rows = [r for r in all_rows if r['fold'] == fold]
        summary = aggregate(rows)
        names = [r['benchmark'] for r in rows]
        for method, result in summary['results'].items():
            d = np.array([(np.array(r['brier'][method])-r['brier']['R25'])@ALC_WEIGHTS for r in rows])
            bmd = {b: s['alc_mean'][method]-s['alc_mean']['R25'] for b,s in summary['by_benchmark'].items()}
            loo = {b: float(d[np.array(names) != b].mean()) for b in bmd}
            outcome = {'fold': fold, 'method': method, 'alc': result['alc_mean'],
                'delta_R25': float(d.mean()), 'benchmark_equal_alc': summary['benchmark_equal_alc'][method],
                'benchmark_equal_delta_R25': summary['benchmark_equal_alc'][method]-summary['benchmark_equal_alc']['R25'],
                'benchmarks_improved': sum(v < -1e-10 for v in bmd.values()), 'benchmarks': len(bmd),
                'severe_regressions': {b:v for b,v in bmd.items() if v >= .005},
                'leave_one_benchmark_out_delta': loo, **bootstrap(d, names)}
            outcomes.append(outcome)
            for j,n in enumerate(BUDGETS):
                metrics.append({'fold': fold, 'method': method, 'pairs': len(rows), 'budget': int(n),
                    'brier': result['brier_mean'][j], 'accuracy_at_0_5': result['accuracy_at_0_5'][j],
                    'alc': result['alc_mean'], 'benchmark_equal_alc': summary['benchmark_equal_alc'][method]})
            for name, values in summary['by_benchmark'].items():
                for j,n in enumerate(BUDGETS):
                    slices.append({'fold': fold, 'benchmark': name, 'method': method, 'pairs': values['pairs'],
                        'budget': int(n), 'brier': values['brier_mean'][method][j],
                        'alc': values['alc_mean'][method], 'delta_alc_R25': bmd[name]})
        # Benchmark-label oracle selects a fixed method for both low budgets per Benchmark.
        bmoracle = 0.
        for name in sorted(set(names)):
            group = [r for r in rows if r['benchmark'] == name]
            means = {m: np.mean([r['brier'][m] for r in group], axis=0) for m in ['R25']+MODELS}
            bmoracle += len(group)*(min(v[:2]@ALC_WEIGHTS[:2] for v in means.values())+means['R25'][2:]@ALC_WEIGHTS[2:])
        opportunities.append({'fold': fold, 'R25_alc': summary['results']['R25']['alc_mean'],
            'pair_label_oracle_alc': summary['results']['pair_label_oracle']['alc_mean'],
            'benchmark_label_oracle_alc': float(bmoracle/len(rows)), 'deployable': False})
    write_json(OUT/'composition_audit.json', outcomes)
    write_json(OUT/'oracle_diagnostic.json', opportunities)
    report = Path('../reports')
    pd.DataFrame(metrics).to_csv(report/'low_budget_prior_composition_metrics.csv', index=False)
    pd.DataFrame(slices).to_csv(report/'low_budget_prior_composition_benchmark_metrics.csv', index=False)
    write_json(report/'low_budget_prior_composition_metrics.json', {'metrics': metrics, 'audit': outcomes})
    return outcomes, corr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = freeze()
    if (OUT/'complete.json').exists():
        print(json.dumps(read(OUT/'complete.json')))
        return
    start = time.perf_counter()
    rows, detail, replay = [], [], []
    engine = CachedSemantic()
    with threadpool_limits(limits=4):
        for fold in range(3):
            r, d, a = fold_replay(fold, engine)
            rows.extend(r); detail.extend(d); replay.extend(a)
            print(json.dumps({'fold': fold, 'pairs': len(r), 'replay_max_error': max(x['brier_error'] for x in a)}), flush=True)
        outcomes, corr = analysis(rows, detail)
    for p, expected in hashes.items():
        assert sha(p) == expected, f'Execution input changed: {p}'
    write_json(OUT/'replay_audit.json', {'records': replay, 'max_error': max(r['brier_error'] for r in replay),
        'response_group_mapping': 'all variants exact features/counts/successes matched; R25 item probabilities broadcast',
        'decomposition_identity_max_tolerance': 1e-10, 'new_fits': 0})
    write_json(OUT/'complete.json', {'status': 'diagnostics_complete_operator_pending', 'pairs_with_pool_repetition': len(rows),
        'pair_budget_method_rows': len(detail), 'seconds': time.perf_counter()-start,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'new_fits': 0})
    print(json.dumps({'seconds': time.perf_counter()-start,
        'segment_results': [r for r in outcomes if r['method'].endswith('_segment')]}), flush=True)


if __name__ == '__main__':
    main()
