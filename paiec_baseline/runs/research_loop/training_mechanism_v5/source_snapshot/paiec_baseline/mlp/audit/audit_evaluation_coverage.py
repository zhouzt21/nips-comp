"""R30: exact counts and clearly labelled stratified score extrapolation."""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('runs/research_loop/R30');root.mkdir(parents=True,exist_ok=True)
    cache=json.loads(Path('runs/mlp_failure_2026_10_22/stat_prior/pair_rates.json').read_text())
    old=json.loads(Path('runs/research_loop/R25_verify/summary.json').read_text())
    rows=[]
    for row in old['by_benchmark']:
        directory=row['benchmark'];count=len(cache['by_benchmark'][directory]['pair_rates'])
        assert row['pairs']<=count
        rows.append({'benchmark':directory,'evaluated_pairs':row['pairs'],'eligible_pairs':count,
                     'coverage':row['pairs']/count,'frozen_R25_benchmark_alc':row['separate'],
                     'frozen_baseline_benchmark_alc':row['smoothed_mean_2']})
    sampled=sum(r['evaluated_pairs'] for r in rows);full=sum(r['eligible_pairs'] for r in rows)
    for r in rows:r['proxy_weight']=r['evaluated_pairs']/sampled;r['full_count_weight']=r['eligible_pairs']/full
    weighted=lambda key:sum(r['eligible_pairs']*r[key] for r in rows)/full
    d={'source_revision':cache['source_revision'],'scope':'counts exact; score reweighting is stratified extrapolation, NOT evaluation of omitted pairs/items',
       'benchmarks':len(rows),'evaluated_pairs':sampled,'eligible_pairs':full,'pair_coverage':sampled/full,
       'absolute_weight_shift_total_variation':sum(abs(r['proxy_weight']-r['full_count_weight']) for r in rows)/2,
       'observed_capped_scores':{k:old['pair_equal_alc'][k] for k in ['separate','smoothed_mean_2']},
       'stratified_extrapolation':{'R25':weighted('frozen_R25_benchmark_alc'),'smoothed2':weighted('frozen_baseline_benchmark_alc')},
       'by_benchmark':rows,'evaluation_items_per_pair_cap':64,'omitted_pair_scores_observed':False}
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/audit/audit_evaluation_coverage.py','../research_loop/R30_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in d.items() if k!='by_benchmark'},indent=2))


if __name__=='__main__':main()
