"""R24 input-only structured metadata coverage; does not load response labels."""
import collections
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .evaluate import _stable_seed
from .visible_item_evidence import plain


def main():
    root=Path('runs/research_loop/R24');root.mkdir(parents=True,exist_ok=True);start=time.time();rows=[]
    dirs=sorted({b for f in range(3) for b in json.loads(Path(f'runs/mlp_cv/fold{f}_split.json').read_text())['development_benchmarks']})
    for directory in dirs:
        data=pd.read_parquet(Path('D:/paiec_payload/data')/directory/'items.parquet',columns=['item_id','item_features'])
        if len(data)>100:
            rng=np.random.default_rng(_stable_seed(123,directory,'input_audit'));data=data.iloc[np.sort(rng.choice(len(data),100,replace=False))]
        values=[];keys=collections.Counter();perkey=collections.defaultdict(set);json_objects=0
        for raw in data['item_features']:
            text=plain(raw)
            try:value=json.loads(text)
            except (json.JSONDecodeError,TypeError):value=text
            if isinstance(value,dict):
                json_objects+=1
                for k,v in value.items():keys[k]+=1;perkey[k].add(json.dumps(v,sort_keys=True,ensure_ascii=False))
            if value in ('',None) or value=={} or value==[]:values.append('')
            else:values.append(json.dumps(value,sort_keys=True,ensure_ascii=False))
        row={'benchmark':directory,'items':len(values),'nonempty':sum(bool(v) for v in values),
             'distinct_including_empty':len(set(values)),'distinct_nonempty':len(set(values)-{''}),'json_objects':json_objects,
             'keys':dict(keys),'distinct_per_key':{k:len(v) for k,v in perkey.items()}}
        rows.append(row)
    output={'scope':'input-only metadata audit, no response labels loaded','items':sum(r['items'] for r in rows),
        'nonempty':sum(r['nonempty'] for r in rows),'benchmarks_with_varying_metadata':sum(r['distinct_including_empty']>1 for r in rows),
        'benchmarks':rows,'elapsed_seconds':time.time()-start}
    (root/'metadata_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    sources=['mlp/audit_item_metadata.py','research_loop/R24_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'audit_provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='benchmarks'},indent=2))
    print([(r['benchmark'],r['nonempty'],r['distinct_nonempty'],r['distinct_per_key']) for r in rows if r['nonempty']])


if __name__=='__main__':main()
