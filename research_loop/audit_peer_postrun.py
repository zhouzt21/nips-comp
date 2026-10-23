"""Post-result identity and own-operator audit; no selection or candidate changes."""
import time
import pandas as pd
from historical_positive_common import *
from peer_current_predictor import joint
from mlp.cohort_semantic import CachedSemantic
from mlp.metadata_features import encode

start=time.perf_counter()
from threadpoolctl import threadpool_limits
threadpool_limits(4)
engine=CachedSemantic();checks=[]
for f in range(3):
    ridge=source(f)['semantic_ridge']
    for folder in sorted((ROOT/f'R72/full/outer/fold{f}').iterdir()):
        ip=sorted(folder.glob('index_*.json'))[0];idx=read(ip)
        arr=dict(np.load(folder/f'predictions_{ip.stem.split("_")[-1]}.npz'))
        items=pd.read_parquet(Path('D:/paiec_payload/data')/folder.name/'formatted_tables/items.parquet',columns=['item_id','content','item_features']).drop_duplicates('item_id')
        data={str(v.item_id):(v.content,v.item_features) for v in items.itertuples(index=False)};cache={}
        def vector(key):
            if key not in cache:
                text,meta=data[key];cache[key]=np.r_[engine.vector(text)/np.sqrt(ridge),encode(meta)/4.]
            return cache[key]
        seen=set()
        for row in idx:
            if row['seed'] in seen:continue
            seen.add(row['seed'])
            support=np.stack([vector(v) for v in row['support_ids']])
            query=np.stack([vector(row['query_ids'][i]) for i in row['query_variant_item_indices']])
            labels=np.array(row['support_labels'],float)
            p=np.column_stack([joint([{'x':support[:n],'y':labels[:n]}],query) for n in BUDGETS])
            a=row['start'];b=a+row['size'];error=float(np.max(np.abs(p-arr['R25'][a:b])))
            assert error<1e-12
            assert set(row['support_ids']).isdisjoint(row['query_ids'])
            checks.append({'benchmark':row['benchmark'],'seed':row['seed'],'subject_id':row['subject_id'],'own_joint_vs_saved_R25_error':error})
result={'scope':'post-result deterministic operator audit, no selection; first available trajectory per benchmark/seed in first chunk',
        'trajectories':len(checks),'benchmarks':len({v['benchmark'] for v in checks}),'max_error':max(v['own_joint_vs_saved_R25_error'] for v in checks),
        'checks':checks,'seconds':time.perf_counter()-start,'finished':now()}
write_json(ROOT/'R72/postrun_own_geometry_audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
