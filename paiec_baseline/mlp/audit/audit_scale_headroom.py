"""R33 nondeployable oracle: current R25 ray with attenuation vs amplification."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.evaluate import BUDGETS, ALC_WEIGHTS
from mlp.cohort_semantic import CachedSemantic
from mlp.measurement_episodes import episodes, weighted_brier
from mlp.run.run_metadata_round import prediction
from mlp.oracle_scale import optimum_scale
from mlp.reporting.summarize_candidate import summarize

NAMES = ['oracle_amplified', 'oracle_max4', 'oracle_attenuated', 'R25', 'smoothed_mean_2']


def main():
    measurement = json.loads(Path('runs/research_loop/R32/measurement_conclusion.json').read_text())
    # More favorable endpoint still must leave positive gap; otherwise R32 is unresolved.
    assert .005 + measurement['R31_actual_capped_delta'] + measurement['approx_design_ci95'][0] > 0
    start = time.time(); root = Path('runs/research_loop/R33'); root.mkdir(parents=True, exist_ok=True)
    sources = ['mlp/audit/audit_scale_headroom.py', 'mlp/oracle_scale.py', 'mlp/measurement_episodes.py',
               'mlp/run/run_metadata_round.py', '../research_loop/R33_protocol.md']
    snap = root/'code'; snap.mkdir(exist_ok=True)
    hashes = {}
    for p in sources:
        content = Path(p).read_bytes(); (snap/Path(p).name).write_bytes(content)
        hashes[p] = hashlib.sha256(content).hexdigest()
    engine = CachedSemantic()
    full_dominant = None
    for fold in range(3):
        selection = json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text())
        historic = json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text())
        by = {}; total = {k: np.zeros(6) for k in NAMES}; count = 0
        for directory, old in historic['by_benchmark'].items():
            is_dominant = directory == 'alignment_faking/formatted_tables'
            sums = {k: np.zeros(6) for k in NAMES}; full_sums = {k: np.zeros(6) for k in NAMES}
            nrows = 0; coefficients = [[] for _ in BUDGETS]
            with (root/(directory.split('/')[0]+'_oracle.csv')).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream); writer.writerow(['subject_id','seed','budget','scope',*NAMES,'oracle_c'])
                for row in episodes(directory, engine, selection['semantic_ridge'], selection['metadata_ridge'], full=is_dominant):
                    for index, n in enumerate(BUDGETS):
                        a = row['labels'][:n]; base = (1+a.sum())/(2+n)
                        pred = prediction(row['support'][:n], row['query'], a, 1.)
                        for scope, length in [('capped', min(64, len(pred))), *([('full',len(pred))] if is_dominant else [])]:
                            p = pred[:length]; m = row['counts'][:length]; s = row['successes'][:length]
                            amplified, c = optimum_scale(base, p-base, m, s)
                            bounded, _ = optimum_scale(base, p-base, m, s, 4.)
                            attenuated, _ = optimum_scale(base, p-base, m, s, 1.)
                            vals = [amplified, bounded, attenuated, weighted_brier(p,m,s), weighted_brier(base,m,s)]
                            assert amplified <= bounded+1e-10 and bounded <= attenuated+1e-10 and attenuated <= min(vals[3:])+1e-10
                            target = sums if scope == 'capped' else full_sums
                            for k, v in zip(NAMES,vals): target[k][index] += v
                            writer.writerow([row['subject_id'],row['seed'],n,scope,*vals,c])
                            if scope == 'capped': coefficients[index].append(c)
                    nrows += 1
            assert nrows == 3*old['pairs_per_seed'][0]
            avg = {k:v/nrows for k,v in sums.items()}; pairs = nrows//3; count += pairs
            for k in ['R25','smoothed_mean_2']:
                np.testing.assert_allclose(avg[k], old['brier_mean'][k], atol=1e-7, rtol=0)
            by[directory] = {'pairs_per_seed':[pairs]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},
                             'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},
                             'oracle_c_median': [float(np.median(v)) for v in coefficients],
                             'oracle_c_above1_fraction': [float(np.mean(np.asarray(v)>1.00001)) for v in coefficients]}
            for k in NAMES: total[k] += pairs*avg[k]
            if is_dominant:
                d = {k:v/nrows for k,v in full_sums.items()}
                ref = json.loads(Path('runs/research_loop/R32c/summary.json').read_text())
                for k in ['R25','smoothed_mean_2']: np.testing.assert_allclose(d[k],ref['full_brier'][k],atol=1e-7,rtol=0)
                full_dominant = {'pairs':pairs,'full_brier':{k:v.tolist() for k,v in d.items()},'full_alc':{k:float(v@ALC_WEIGHTS) for k,v in d.items()}}
            print('completed',directory,'pairs',pairs,flush=True)
        result = {'pairs_per_seed':[count]*3,'by_benchmark':by,
                  'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    summary = summarize([root/f'fold{i}_evaluation.json' for i in range(3)],NAMES[0],NAMES[1:])
    summary.update(scope='evaluation-label oracle for a single fixed clipped R25 ray; not deployable performance or a universal information bound',
                   dominant_full_query=full_dominant,elapsed_seconds=time.time()-start)
    summary['amplified_family_can_reach_gain_005'] = summary['pair_equal_alc']['smoothed_mean_2']-summary['pair_equal_alc']['oracle_amplified'] >= .005
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for p in sources: assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == hashes[p]
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['pair_equal_alc','amplified_family_can_reach_gain_005','elapsed_seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
