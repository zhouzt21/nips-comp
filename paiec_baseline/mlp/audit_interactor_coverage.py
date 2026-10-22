"""R50 inspect omitted official input variation, without outcome correlations."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .evaluate import _load_directory


def main():
    start=time.time();root=Path('runs/research_loop/R50');root.mkdir(parents=True,exist_ok=True);by={}
    paths=['mlp/audit_interactor_coverage.py','mlp/evaluate.py','research_loop/R50_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    for fold in range(3):
        old=json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text())
        for directory,expected in old['by_benchmark'].items():
            rows,subjects,_=_load_directory(Path('D:/paiec_payload/data'),directory)
            rows['interactors']=rows['interactors'].fillna('').astype(str).str.strip()
            records=[]
            for sid,g in rows.groupby('subject_id',sort=False):
                if sid not in subjects or g.item_id.nunique()<80:continue
                v=g['interactors'];records.append({'subject_id':str(sid),'items':int(g.item_id.nunique()),'distinct_interactors':int(v.nunique()),
                    'nonempty_rows':int((v!='').sum()),'responses':len(g),'nonempty_strings':int(v[v!=''].nunique())})
            assert len(records)==expected['pairs_per_seed'][0]
            by[directory]={'fold':fold,'pairs':len(records),'variable_interactor_pairs':sum(r['distinct_interactors']>1 for r in records),
                'nonempty_pairs':sum(r['nonempty_rows']>0 for r in records),'max_interactors_per_pair':max(r['distinct_interactors'] for r in records)}
            (root/(directory.split('/')[0]+'_counts.json')).write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(directory,by[directory],flush=True)
    d={'scope':'input variation in36reused public benchmarks; no label association or predictor performance','by_benchmark':by,
       'pairs':sum(v['pairs'] for v in by.values()),'variable_interactor_pairs':sum(v['variable_interactor_pairs'] for v in by.values()),
       'nonempty_pairs':sum(v['nonempty_pairs'] for v in by.values()),'elapsed_seconds':time.time()-start}
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8');(root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    snap=root/'code';snap.mkdir(exist_ok=True)
    for p in paths:
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
        (snap/Path(p).name).write_bytes(Path(p).read_bytes())
    print(json.dumps({k:v for k,v in d.items() if k!='by_benchmark'},indent=2),flush=True)


if __name__=='__main__':main()
