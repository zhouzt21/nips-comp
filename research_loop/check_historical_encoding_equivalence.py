"""Input-only exactness and timing audit, motivated by R62 full-scoring cost."""
import time
from historical_positive_common import *
from historical_exact_encoding_cache import ExactEncodingCache
from mlp.evaluate import _load_directory
from mlp.model import encode_mode

def main():
    cache=ExactEncodingCache();cases=[]
    # Varied public tasks, subjects, item lengths and interactors. No response labels used.
    for name in ['alignment_faking','bird_sql','emoji_attack','ceval']:
        rows,subjects,items=_load_directory(Path('D:/paiec_payload/data'),name+'/formatted_tables')
        for sid,g in list(rows.groupby('subject_id',sort=False))[:4]:
            for row in g.drop_duplicates('item_id').head(20).itertuples(index=False):
                item=items[row.item_id]
                cases.append([subjects[sid],{'item_content':item['content'],'item_features':item['item_features'],
                                             'interactors':row.interactors}])
    start=time.perf_counter();expected=[encode_mode(x,'all') for x in cases]; original=time.perf_counter()-start
    start=time.perf_counter();actual=[cache(x) for x in cases];first=time.perf_counter()-start
    start=time.perf_counter();again=[cache(x) for x in cases];warm=time.perf_counter()-start
    assert all(np.array_equal(a,b) and np.array_equal(a,c) for a,b,c in zip(expected,actual,again))
    result={'cases':len(cases),'bitwise_equal':True,'max_error':0.,'original_seconds':original,
            'cache_first_seconds':first,'cache_warm_seconds':warm,'stats':cache.stats(),
            'method':'same per-coordinate ordered additions, same float32 rounding; no summation shortcut',
            'scope':'input-only exactness/compute audit, not candidate or hyperparameter selection',
            'source_sha256':{str(p):sha(p) for p in [Path(__file__),REPO/'research_loop/historical_exact_encoding_cache.py',BASE/'mlp/model.py']}}
    write_json(ROOT/'R62/encoding_equivalence.json',result);print(json.dumps(result))

if __name__=='__main__':main()
