"""Check new count-aggregated capped scores against actual R31 response-level scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root=Path('runs/research_loop/R32');selection=json.loads((root/'selection.json').read_text());directories=sorted({r['benchmark'] for r in selection});checks=[];pending=[]
    for directory in directories:
        path=root/(directory.split('/')[0]+'_full_query.json')
        if not path.exists():pending.append(directory);continue
        rows=json.loads(path.read_text());fold=rows[0]['fold'];old=json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text())['by_benchmark'][directory]
        original=Path('runs/research_loop/R31')/(directory.split('/')[0]+'_pair_brier.csv')
        if original.exists():
            table=pd.read_csv(original,dtype={'subject_id':str})
            for row in rows:
                for run in row['seeds']:
                    match=table[(table.subject_id==row['subject_id'])&(table.seed==run['seed'])].sort_values('budget')
                    assert len(match)==6
                    for name,values in run['capped_brier'].items():
                        np.testing.assert_allclose(values,match[name],atol=1e-7,rtol=0)
            checks.append({'benchmark':directory,'type':'same pair/seed/budget response replay','pairs':len(rows)})
        elif len(rows)==old['pairs_per_seed'][0]:
            for name in ['R25','semantic','smoothed_mean_2','training_selected_smoothed']:
                actual=np.mean([np.mean([s['capped_brier'][name] for s in row['seeds']],axis=0) for row in rows],axis=0)
                np.testing.assert_allclose(actual,old['brier_mean'][name],atol=1e-7,rtol=0)
            checks.append({'benchmark':directory,'type':'complete benchmark three-seed response replay','pairs':len(rows)})
    result={'tolerance':1e-7,'checks_passed':checks,'pending_benchmarks':pending,'completed_benchmarks':len(directories)-len(pending)}
    (root/'replay_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(f'{len(checks)} historical replay checks passed; {len(pending)} benchmarks pending')


if __name__=='__main__':main()
