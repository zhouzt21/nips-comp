"""R32 design-based cap-effect estimate, with exact dominant-benchmark replacement."""
import json
from pathlib import Path
import numpy as np
from mlp.evaluate import ALC_WEIGHTS


def main():
    root = Path('runs/research_loop/R32')
    original = json.loads((root / 'summary.json').read_text())
    dominant = json.loads(Path('runs/research_loop/R32c/summary.json').read_text())
    previous = json.loads(Path('runs/research_loop/R31/summary.json').read_text())
    selected = json.loads((root / 'selection.json').read_text())
    rows = [r for directory in sorted({r['benchmark'] for r in selected})
            for r in json.loads((root / (directory.split('/')[0] + '_full_query.json')).read_text())]
    dominant_name = 'alignment_faking/formatted_tables'
    strata = {}
    shift_sum = 0.
    by = {}
    for r in rows:
        values = np.array([s['full_delta_alc'] - s['capped_delta_alc'] for s in r['seeds']])
        shift = float(values.mean())
        key = (r['benchmark'], r['stratum'])
        strata.setdefault(key, []).append((shift, r))
        b = by.setdefault(r['benchmark'], {'represented_pairs': 0., 'selected_pairs': 0,
                                         'shift_sum': 0., 'weighted_subset_variance': 0.})
        b['represented_pairs'] += r['design_weight']
        b['selected_pairs'] += 1
        b['shift_sum'] += r['design_weight'] * shift
        b['weighted_subset_variance'] += r['design_weight'] * np.mean([
            np.var(s['conditional_64_subset_delta_alc'], ddof=1) for s in r['seeds']])
        if r['benchmark'] != dominant_name:
            shift_sum += r['design_weight'] * shift
    variance = 0.
    for (directory, _), values in strata.items():
        if directory == dominant_name:
            continue
        n = len(values)
        population = values[0][1]['population_count']
        if n < population:
            variance += population**2 * (1 - n/population) * np.var([v for v, _ in values], ddof=1)/n
    exact = ((dominant['full_alc']['R25'] - dominant['full_alc']['smoothed_mean_2']) -
             (dominant['capped_alc']['R25'] - dominant['capped_alc']['smoothed_mean_2']))
    shift_sum += dominant['pairs'] * exact
    count = int(sum(b['represented_pairs'] for b in by.values()))
    assert count == 5242
    shift = shift_sum / count
    se = float(np.sqrt(variance) / count)
    capped_delta = previous['pair_equal_alc']['R25'] - previous['pair_equal_alc']['smoothed_mean_2']
    for name, b in by.items():
        b['estimated_cap_effect'] = b.pop('shift_sum') / b['represented_pairs']
        b['conditional_subset_delta_rms_sd'] = float(np.sqrt(b.pop('weighted_subset_variance') / b['represented_pairs']))
        if name == dominant_name:
            b['actual_cap_effect_all_pairs'] = exact
    d = {
        'scope': 'cap-effect estimate: exact 4278-pair dominant benchmark plus count-stratified sample of other 964 pairs; NOT actual all-pair full-query performance',
        'original_stratified_cap_effect': original['cap_removal_effect_on_candidate_minus_baseline'],
        'hybrid_cap_effect': shift, 'design_standard_error': se,
        'approx_design_ci95': [shift - 1.96*se, shift + 1.96*se],
        'R31_actual_capped_delta': capped_delta,
        'cap_corrected_delta_estimate': capped_delta + shift,
        'cap_corrected_gain_estimate': -capped_delta - shift,
        'remaining_gain_to_gate_estimate': .005 + capped_delta + shift,
        'by_benchmark': by,
        'interpretation_limits': [
            'Design interval concerns pair sampling within this fixed reused benchmark population, not generalization to unseen benchmarks.',
            'Seed-averaged pairs are the sampling units; repeated seeds are not counted as independent pairs.',
            'Conditional subset SD is within pair with support fixed, not uncertainty of the final competition score.',
            'Empirical within-item variance cancels between model Briers; it is not a population irreducible-noise estimate.'
        ]
    }
    (root / 'measurement_conclusion.json').write_text(json.dumps(d, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in d.items() if k != 'by_benchmark'}, indent=2))


if __name__ == '__main__':
    main()
