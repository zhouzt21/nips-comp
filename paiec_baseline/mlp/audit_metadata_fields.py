"""Second R24 input audit: the corpus uses semicolon key=value strings."""
import collections
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import _stable_seed
from .metadata_features import fields,FIELDS


def main():
    root=Path('runs/research_loop/R24');by=[]
    dirs=sorted({b for f in range(3) for b in json.loads(Path(f'runs/mlp_cv/fold{f}_split.json').read_text())['development_benchmarks']})
    for directory in dirs:
        data=pd.read_parquet(Path('D:/paiec_payload/data')/directory/'items.parquet',columns=['item_features'])
        if len(data)>100:
            rng=np.random.default_rng(_stable_seed(123,directory,'input_audit'));data=data.iloc[np.sort(rng.choice(len(data),100,replace=False))]
        mapped=[fields(t) for t in data['item_features']];values=collections.defaultdict(set)
        for item in mapped:
            for k,v in item.items():values[k].add(v)
        by.append({'benchmark':directory,'items':len(mapped),'nonempty':sum(bool(x) for x in mapped),
                   'distinct':len({json.dumps(x,sort_keys=True) for x in mapped}),'fields':{k:len(v) for k,v in values.items()}})
    result={'scope':'input-only, declared generic fields; no outcome-based selection','allowed_fields':sorted(FIELDS),
            'nonempty':sum(r['nonempty'] for r in by),'varying_benchmarks':sum(r['distinct']>1 for r in by),'by_benchmark':by}
    (root/'field_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='by_benchmark'})
    print([(r['benchmark'],r['distinct'],r['fields']) for r in by if r['distinct']>1])


if __name__=='__main__':main()
