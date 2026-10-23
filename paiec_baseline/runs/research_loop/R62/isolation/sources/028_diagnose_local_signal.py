"""R10: separate mean correction, item discrimination and support correspondence."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.evaluate import ALC_WEIGHTS, BUDGETS, _stable_seed
from mlp.local_ridge import kernel_predict
from mlp.semantic_features import MODEL_REVISION
from mlp.run.run_semantic_round import episodes
from mlp.reporting.summarize_candidate import summarize

ROOT = Path('runs/research_loop/R10')
MODES = ['semantic', 'text_hash']
REPEATS = 16


def decompose(predictions, labels, baseline):
    p, y = np.asarray(predictions, float), np.asarray(labels, float)
    m, q = p.mean(), y.mean()
    mean_gain = (baseline-q)**2 - (m-q)**2
    item_gain = 2*np.mean((p-m)*(y-q))-np.var(p)
    brier = np.mean((p-y)**2)
    baseline_brier = np.mean((baseline-y)**2)
    residual = baseline_brier-brier-mean_gain-item_gain
    return {'actual': float(brier), 'baseline': float(baseline_brier),
            'mean_only': float(np.mean((m-y)**2)),
            'query_permutation_expectation': float(np.mean(p*p)-2*m*q+np.mean(y*y)),
            'mean_gain': float(mean_gain), 'item_gain': float(item_gain),
            'prediction_variance': float(np.var(p)), 'identity_residual': float(residual)}


class CacheOnly:
    def __init__(self):
        self.fingerprint = hashlib.sha256(MODEL_REVISION.encode()+Path('mlp/semantic_features.py').read_bytes()+Path('mlp/structured_text.py').read_bytes()).hexdigest()

    def encode_many(self, _):
        raise RuntimeError('R7 episode cache missing; restore it before this diagnostic')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    start = time.time(); engine = CacheOnly(); max_error = 0.; rows_csv = []
    names = ['actual', 'baseline', 'mean_only', 'query_permutation_expectation', 'support_permuted']
    for fold in range(3):
        selection = json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text())
        old = json.loads(Path(f'runs/research_loop/R7_verify/fold{fold}_evaluation.json').read_text())
        all_benchmarks = {mode: {} for mode in MODES}
        components = {mode: {} for mode in MODES}
        counts = 0
        totals = {mode: {k: np.zeros(6) for k in names} for mode in MODES}
        for directory in selection['heldout_benchmarks']:
            bench = {mode: {k: [] for k in names} for mode in MODES}
            comp = {mode: {k: [] for k in ['mean_gain', 'item_gain', 'prediction_variance']} for mode in MODES}
            pair_count = None
            for seed in [0, 1, 2]:
                data = episodes(directory, seed, 100, engine)
                if not data: continue
                if pair_count is None: pair_count = len(data)
                assert len(data) == pair_count
                for pair_index, row in enumerate(data):
                    for mode in MODES:
                        values = {k: [] for k in names}; parts = {k: [] for k in comp[mode]}
                        ridge = selection['selected_ridge'][mode]
                        for n in BUDGETS:
                            labels = row['a'][:n]; b = (1+labels.sum())/(2+n)
                            p = kernel_predict(row['k_'+mode], row['c_'+mode], labels, ridge)
                            result = decompose(p, row['y'], b)
                            max_error = max(max_error, abs(result['identity_residual']))
                            rng = np.random.default_rng(_stable_seed(90210, directory, f'{seed}|{pair_index}|{n}'))
                            permuted_scores = []
                            if n < 2 or ridge is None:
                                permuted_scores = [result['actual']]*REPEATS
                            else:
                                for _ in range(REPEATS):
                                    permuted = kernel_predict(row['k_'+mode], row['c_'+mode], rng.permutation(labels), ridge)
                                    permuted_scores.append(float(np.mean((permuted-row['y'])**2)))
                            result['support_permuted'] = float(np.mean(permuted_scores))
                            for k in names: values[k].append(result[k])
                            for k in parts: parts[k].append(result[k])
                            rows_csv.append({'fold': fold, 'benchmark': directory, 'seed': seed, 'pair_index': pair_index,
                                             'mode': mode, 'budget': int(n), 'evaluation_responses': len(row['y']),
                                             **result, 'permutation_sd': float(np.std(permuted_scores))})
                        for k in names: bench[mode][k].append(values[k])
                        for k in parts: comp[mode][k].append(parts[k])
            if pair_count is None: continue
            counts += pair_count
            for mode in MODES:
                averages = {k: np.mean(v, axis=0) for k, v in bench[mode].items()}
                np.testing.assert_allclose(averages['actual'], old['by_benchmark'][directory]['brier_mean'][mode], atol=1e-7, rtol=0)
                np.testing.assert_allclose(averages['baseline'], old['by_benchmark'][directory]['brier_mean']['smoothed_mean_2'], atol=1e-7, rtol=0)
                all_benchmarks[mode][directory] = {'pairs_per_seed': [pair_count]*3,
                    'brier_mean': {k: v.tolist() for k, v in averages.items()},
                    'alc_mean': {k: float(v@ALC_WEIGHTS) for k, v in averages.items()}}
                components[mode][directory] = {k: np.mean(v, axis=0).tolist() for k, v in comp[mode].items()}
                for k, v in averages.items(): totals[mode][k] += pair_count*v
            print(f'fold{fold} {directory}: {pair_count} pairs x3seeds', flush=True)
        for mode in MODES:
            folder = ROOT/mode; folder.mkdir(exist_ok=True)
            payload = {'selection': selection, 'seeds': [0, 1, 2], 'pairs_per_seed': [counts]*3,
                       'by_benchmark': all_benchmarks[mode], 'components': components[mode],
                       'results': {k: {'brier_mean': (v/counts).tolist(), 'alc_mean': float(v/counts@ALC_WEIGHTS)} for k, v in totals[mode].items()}}
            (folder/f'fold{fold}_evaluation.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    assert max_error < 1e-10, max_error
    with (ROOT/'pair_budget_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows_csv[0].keys()); writer.writeheader(); writer.writerows(rows_csv)
    summaries = {}
    for mode in MODES:
        folder = ROOT/mode
        summary = summarize([folder/f'fold{i}_evaluation.json' for i in range(3)], 'actual', names[1:])
        summary['scope'] = 'Post hoc mechanism diagnosis on reused public benchmarks; mean/query-shuffle controls are not deployable predictors'
        summary['bootstrap_unit'] = 'benchmark after averaging three trajectory seeds; permutations are not independent test units'
        summary['components'] = {'mean_gain': summary['pair_equal_alc']['baseline']-summary['pair_equal_alc']['mean_only'],
                                 'item_gain': summary['pair_equal_alc']['mean_only']-summary['pair_equal_alc']['actual']}
        (folder/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8'); summaries[mode] = summary
    sources = ['mlp/audit/diagnose_local_signal.py', 'mlp/local_ridge.py', 'mlp/run/run_semantic_round.py', 'mlp/semantic_features.py', 'mlp/reporting/summarize_candidate.py', '../research_loop/R10_protocol.md']
    snapshot = ROOT/'code'; snapshot.mkdir(exist_ok=True)
    for source in sources: (snapshot/Path(source).name).write_bytes(Path(source).read_bytes())
    provenance = {'elapsed_seconds': time.time()-start, 'max_identity_residual': max_error, 'permutations': REPEATS,
                  'source_revision': '54074453da7b9911be0f48e4c560be547bc84d21',
                  'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}}
    (ROOT/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps({m: {'alc': s['pair_equal_alc'], 'components': s['components'], 'comparisons': s['comparisons']} for m, s in summaries.items()}, indent=2))


if __name__ == '__main__': main()
