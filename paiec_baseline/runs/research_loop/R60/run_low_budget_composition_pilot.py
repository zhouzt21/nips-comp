"""R60 frozen single-label proper-loss selector and R61 source evaluation."""
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO/'paiec_baseline'
sys.path.insert(0, str(BASE))
from mlp.evaluate import ALC_WEIGHTS, BUDGETS
from mlp.reproduction_episodes import brier, accuracy, aggregate
from mlp.training_mechanism_data import write_json
from mlp.run.run_low_budget_prior import bootstrap

R59 = BASE/'runs/research_loop/R59'
R60 = BASE/'runs/research_loop/R60'
R61 = BASE/'runs/research_loop/R61'
METHOD = 'R56_support1_gate'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def predict(r25, mlp, first_support_sqerr):
    """Prediction contract: no evaluation labels, IDs, or future support labels."""
    assert r25.shape == mlp.shape and r25.shape[1] == 6
    assert np.isfinite(first_support_sqerr) and 0 <= first_support_sqerr <= 1
    chosen = bool(first_support_sqerr < .25)
    result = r25.copy()
    if chosen:
        result[:, 1] = mlp[:, 1]
    assert np.array_equal(result[:, [0,2,3,4,5]], r25[:, [0,2,3,4,5]])
    return result, chosen


def main():
    start = time.perf_counter()
    files = [Path(__file__), REPO/'research_loop/LOW_BUDGET_PRIOR_OPERATOR_DECISION.md',
             REPO/'research_loop/LOW_BUDGET_PRIOR_COMPOSITION_SPEC.md',
             R59/'pair_budget_decomposition.csv', R59/'provenance.json']
    files += list(R59.glob('fold*_predictions.npz')) + list(R59.glob('fold*_prediction_index.json'))
    files += list(R59.glob('fold*_paired_scores.json'))
    hashes = {str(p): sha(p) for p in files}
    R60.mkdir(parents=True, exist_ok=True); R61.mkdir(parents=True, exist_ok=True)
    contract = {'candidate': METHOD, 'threshold': .25, 'budget': 1, 'new_fits': 0,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': hashes,
        'command': 'powershell -NoProfile -ExecutionPolicy Bypass -File D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1 D:/nips-comp/nips-comp/research_loop/run_low_budget_composition_pilot.py',
        'r59_source_contract': read(R59/'provenance.json')}
    if (R60/'contract.json').exists():
        assert read(R60/'contract.json')['source_sha256'] == hashes
        if (R61/'summary.json').exists():
            print(json.dumps(read(R61/'summary.json'))); return
    else:
        write_json(R60/'contract.json', contract)
        for p in files[:3]:
            (R60/p.name).write_bytes(p.read_bytes())
    # Project to the only legal reliability signal before evaluating the candidate.
    df = pd.read_csv(R59/'pair_budget_decomposition.csv', dtype={'subject_id': str})
    legal = df[(df.method == 'R56') & (df.budget == 1)][['fold','benchmark','subject_id','first_support_sqerr']]
    lookup = legal.set_index(['fold','benchmark','subject_id']).first_support_sqerr.to_dict()
    predictions, decisions = {}, []
    for fold in range(3):
        with np.load(R59/f'fold{fold}_predictions.npz', allow_pickle=False) as bank:
            p25, p56 = bank['R25'].copy(), bank['R56'].copy()
        result = p25.copy()
        for index in read(R59/f'fold{fold}_prediction_index.json'):
            a, b = index['start'], index['start']+index['size']
            error = lookup[(fold,index['benchmark'],index['subject_id'])]
            p, chosen = predict(p25[a:b],p56[a:b],error)
            result[a:b] = p
            decisions.append({'fold': fold,'benchmark': index['benchmark'],'subject_id': index['subject_id'],
                'first_support_sqerr': error, 'use_R56_at_B1': chosen})
        predictions[fold] = result
        np.savez_compressed(R61/f'fold{fold}_candidate_predictions.npz', probabilities=result)
    write_json(R60/'decisions.json', decisions)
    # Only after predictions are frozen, load evaluation outcomes and score.
    rows, metrics, slices, gates = [], [], [], []
    for fold in range(3):
        with np.load(R59/f'fold{fold}_predictions.npz', allow_pickle=False) as bank:
            counts, successes = bank['counts'], bank['successes']
        original = {(r['benchmark'],r['subject_id']):r for r in read(R59/f'fold{fold}_paired_scores.json')}
        foldrows = []
        for index in read(R59/f'fold{fold}_prediction_index.json'):
            a, b = index['start'], index['start']+index['size']
            c, s = counts[a:b], successes[a:b]
            p = predictions[fold][a:b].T
            row = original[(index['benchmark'],index['subject_id'])]
            row['brier'][METHOD] = [brier(x,c,s) for x in p]
            row['accuracy'][METHOD] = [accuracy(x,c,s) for x in p]
            foldrows.append(row)
        rows.extend(foldrows)
        summary = aggregate(foldrows)
        write_json(R61/f'fold{fold}_paired_scores.json', foldrows)
        write_json(R61/f'fold{fold}_summary.json', summary)
        bmd = {b: s['alc_mean'][METHOD]-s['alc_mean']['R25'] for b,s in summary['by_benchmark'].items()}
        delta = np.array([(np.array(r['brier'][METHOD])-r['brier']['R25'])@ALC_WEIGHTS for r in foldrows])
        bm_delta = summary['benchmark_equal_alc'][METHOD]-summary['benchmark_equal_alc']['R25']
        severe = {b:d for b,d in bmd.items() if d >= .005}
        decisions_here = [d for d in decisions if d['fold']==fold]
        gate = {'fold': fold, 'pairs': len(foldrows), 'selected_R56_pairs': sum(d['use_R56_at_B1'] for d in decisions_here),
            'alc': summary['results'][METHOD]['alc_mean'], 'delta_R25': float(delta.mean()),
            'benchmark_equal_delta_R25': bm_delta, 'benchmarks_improved': sum(v < -1e-10 for v in bmd.values()),
            'benchmarks': len(bmd), 'severe_regressions': severe,
            'passes_numeric': bool(delta.mean() < 0 and bm_delta <= 0 and not severe),
            **bootstrap(delta, [r['benchmark'] for r in foldrows])}
        gates.append(gate)
        for method, values in summary['results'].items():
            for j,n in enumerate(BUDGETS):
                metrics.append({'fold':fold,'method':method,'pairs':len(foldrows),'budget':int(n),
                    'brier':values['brier_mean'][j], 'accuracy_at_0_5':values['accuracy_at_0_5'][j],
                    'alc':values['alc_mean'], 'benchmark_equal_alc':summary['benchmark_equal_alc'][method]})
            for benchmark, group in summary['by_benchmark'].items():
                for j,n in enumerate(BUDGETS):
                    slices.append({'fold':fold,'benchmark':benchmark,'method':method,'pairs':group['pairs'],
                        'budget':int(n),'brier':group['brier_mean'][method][j], 'alc':group['alc_mean'][method],
                        'delta_alc_R25':group['alc_mean'][method]-group['alc_mean']['R25']})
    for p,expected in hashes.items():
        assert sha(p)==expected
    report = REPO/'reports'
    pd.DataFrame(metrics).to_csv(report/'low_budget_prior_composition_metrics.csv',index=False)
    pd.DataFrame(slices).to_csv(report/'low_budget_prior_composition_benchmark_metrics.csv',index=False)
    passed = all(g['passes_numeric'] for g in gates)
    outcome = {'candidate':METHOD,'source_gates':gates, 'all_source_numeric_pass':passed,
        'outer_status':'pending_mechanism_review' if passed else 'condition_not_met',
        'new_fits':0, 'seconds':time.perf_counter()-start,
        'finished_utc':datetime.now(timezone.utc).isoformat()}
    write_json(R61/'summary.json',outcome)
    write_json(report/'low_budget_prior_composition_metrics.json',{'metrics':metrics,'R61':outcome,
        'R59_audit':read(R59/'composition_audit.json')})
    print(json.dumps(outcome))


if __name__ == '__main__':
    main()
