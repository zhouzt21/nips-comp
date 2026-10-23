"""Selection-context factor effects; do not erase fold2 ridge contrast by dedup."""
from shared_slope_analyze import source_comparison
from historical_positive_common import *

def main():
    factors=[('pool_both','R25'),('pool_sem','own_sem'),('pool_both','pool_sem'),('pool_global_center','pool_both'),
             ('pool_both_R36ridge','pool_both'),('pool_sem_R36ridge','pool_sem'),('partial8','pool_both')]
    records=[]
    for f in range(3):
        rows=[]
        for p in sorted((ROOT/f'R75/source_fold{f}').glob('*/paired_scores.json')):rows+=read(p)
        for m,b in factors:records.append({'selection_fold':f,'method':m,'baseline':b,**source_comparison(rows,m,b)})
    write_json(ROOT/'R75/factor_comparisons_by_selection_fold.json',records)
    print(json.dumps([v for v in records if 'R36ridge' in v['method']]))

if __name__=='__main__':main()
