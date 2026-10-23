"""Derive source-only tables and figures from retained R51/R52 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / 'paiec_baseline/runs/research_loop'
OUT = ROOT / 'reports'
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = np.array([.1, .2, .2, .2, .2, .1])


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r51 = read(RUNS/'R51/summary.json')
    tables, diagnostics, files = [], [], []
    figures = OUT/'figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for fold in r51['folds']:
        f = fold['fold']; ax = axes[f]
        for name, color in [('cnp', '#1663a6'), ('attentive_cnp', '#b44f17')]:
            selected = fold['models'][name]
            summary = selected['summary']
            for method in [name] + (['R25', 'smoothed_mean_2', 'training_selected_smoothed'] if name == 'cnp' else []):
                result = summary['results'][method]
                for j, budget in enumerate(BUDGETS):
                    tables.append({'round': 'R51', 'source_pool': f, 'pairs': fold['pairs'],
                                   'method': method, 'budget': budget,
                                   'brier': result['brier_mean'][j],
                                   'accuracy_at_0_5': result['accuracy_at_0_5'][j],
                                   'alc': result['alc_mean']})
            pair_path = RUNS/'R51'/f'fold{f}_{name}_selected_pair_scores.json'
            rows = read(pair_path); files.append(pair_path)
            histories, weights = [], []
            details = []
            for inner in range(2):
                base = RUNS/'R51'/f'fold{f}_{name}_inner{inner}'
                history = read(base/'history.json'); contract = read(base/'fit_contract.json')
                histories.append(history); weights.append(contract['validation_pairs'])
                details.append({'inner': inner, 'training_pairs': contract['training_pairs'],
                                'validation_pairs': contract['validation_pairs'],
                                'epoch1_training_nll': history[1]['training_nll'],
                                'last_training_nll': history[-1]['training_nll'],
                                'selected_training_nll': history[selected['selected_epoch']]['training_nll'],
                                'last_validation_alc': history[-1]['validation_alc']})
                files.extend([base/'history.json', base/'fit_contract.json'])
            curve = np.average([[h['validation_alc'] for h in hist] for hist in histories], weights=weights, axis=0)
            ax.plot(range(len(curve)), curve, label=name, color=color)
            ax.scatter([selected['selected_epoch']], [curve[selected['selected_epoch']]], color=color, s=32, zorder=4)
            decomposed = {key: np.mean([[row['decomposition'][j][key] for j in range(6)] for row in rows], axis=0).tolist()
                          for key in rows[0]['decomposition'][0]}
            diagnostics.append({'source_pool': f, 'method': name, 'selected_epoch': selected['selected_epoch'],
                                'selected_alc': summary['results'][name]['alc_mean'],
                                'last_validation_alc': float(curve[-1]),
                                'permutation_delta_alc': selected['comparisons'][name+'_permuted'],
                                'decomposition_by_budget': decomposed, 'fits': details})
        baseline = fold['models']['cnp']['summary']['results']['R25']['alc_mean']
        ax.axhline(baseline, ls='--', color='#222222', label='R25')
        ax.set_title(f'Source pool {f} ({fold["pairs"]} pairs)')
        ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Inner held-out Benchmark Brier ALC')
    axes[2].legend(loc='upper left', fontsize=8)
    fig.suptitle('R51: fixed 60-epoch runs; dots show development-selected epochs')
    fig.tight_layout()
    for extension in ['svg', 'png']:
        fig.savefig(figures/f'reproduction_v4_r51_learning.{extension}', dpi=160)
    plt.close(fig)
    files.append(RUNS/'R51/summary.json')

    # R52 uses one fixed final calibration and a zero-subject-head ablation.
    r52_path = RUNS/'R52/summary.json'
    irt_diagnostics = []
    if r52_path.exists():
        r52 = read(r52_path)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for fold in r52['folds']:
            summary = fold['summary']
            f = fold['fold']; ax = axes[f]
            for method, result in summary['results'].items():
                if method.endswith('_permuted'): continue
                for j, budget in enumerate(BUDGETS):
                    tables.append({'round': 'R52', 'source_pool': fold['fold'], 'pairs': summary['pairs'],
                                   'method': method, 'budget': budget,
                                   'brier': result['brier_mean'][j],
                                   'accuracy_at_0_5': result['accuracy_at_0_5'][j], 'alc': result['alc_mean']})
            pair_path = RUNS/'R52'/f'fold{f}_irt_calibration_pair_scores.json'
            rows = read(pair_path); files.append(pair_path)
            decomposed = {key: np.mean([[row['decomposition'][j][key] for j in range(6)] for row in rows], axis=0).tolist()
                          for key in rows[0]['decomposition'][0]}
            worst = sorted([{'benchmark': name, 'alc_delta_R25': result['alc_mean']['irt_calibration']-result['alc_mean']['R25']}
                            for name, result in summary['by_benchmark'].items()], key=lambda r: -r['alc_delta_R25'])
            irt_diagnostics.append({'source_pool': f, 'gate': fold['gate'], 'fits': fold['fit_summaries'],
                                    'decomposition_by_budget': decomposed, 'benchmark_deltas': worst})
            for method, label, color in [('R25', 'R25', '#222222'),
                                         ('irt_calibration', 'IRT + subject prior', '#9b2a28'),
                                         ('irt_zero_subject_prior', 'IRT + zero ability', '#7c61a0')]:
                ax.plot(range(6), summary['results'][method]['brier_mean'], marker='o',
                        color=color, label=label, ls='--' if method.endswith('prior') else '-')
            ax.axhline(.25, color='#777777', lw=.9, ls=':', label='Constant 0.5')
            ax.set_title(f'Source pool {f} ({summary["pairs"]} pairs)')
            ax.set_xticks(range(6), BUDGETS); ax.set_xlabel('Support labels'); ax.grid(alpha=.2)
        axes[0].set_ylabel('Pair-equal Brier score')
        axes[2].legend(fontsize=8)
        fig.suptitle('R52: fixed calibration and zero-subject-head ablation; source validation only')
        fig.tight_layout()
        for extension in ['svg', 'png']:
            fig.savefig(figures/f'reproduction_v4_r52_budgets.{extension}', dpi=160)
        plt.close(fig)
        files.append(r52_path)

    with (OUT/'reproduction_v4_metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tables[0]))
        writer.writeheader(); writer.writerows(tables)
    (OUT/'reproduction_v4_metrics.json').write_text(json.dumps({
        'scope': 'separate overlapping source pools; no outer scores; accuracy uses p>=0.5',
        'budgets': BUDGETS, 'metrics': tables, 'R51_diagnostics': diagnostics, 'R52_diagnostics': irt_diagnostics,
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    }, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'metric_rows': len(tables), 'diagnostics': len(diagnostics), 'R52_included': r52_path.exists()}))


if __name__ == '__main__': main()
