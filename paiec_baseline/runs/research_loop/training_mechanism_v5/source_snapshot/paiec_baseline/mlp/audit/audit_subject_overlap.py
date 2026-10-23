"""R15a input-only audit of cross-benchmark subject identity coverage."""
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import pandas as pd
from mlp.model import _SUBJECT_FIELDS,_string
from mlp.run.run_shared_round import records

VARIANTS={'name':['normalized_name'],'config':['normalized_name','provider','reasoning_effort'], 'full':list(_SUBJECT_FIELDS)}


def identity(subject,variant='config'):
    if not _string(subject.get('normalized_name')):return None
    return tuple(_string(subject.get(k)) for k in VARIANTS[variant])


def main():
    root=Path('runs/research_loop/R15');root.mkdir(parents=True,exist_ok=True);out=[]
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text())
        sources={v:defaultdict(set) for v in VARIANTS}
        for directory in split['train_benchmarks']:
            df=pd.read_parquet(Path('D:/paiec_payload/data')/directory/'subjects.parquet',columns=list(_SUBJECT_FIELDS))
            for subject in df.to_dict('records'):
                for v in VARIANTS:
                    key=identity(subject,v)
                    if key is not None:sources[v][key].add(directory)
        per={};total=0
        for directory in split['development_benchmarks']:
            # records carry outcomes for other experiments; only subject dictionaries
            # are projected here. No acquisition/evaluation value enters this audit.
            subjects=[r['subject'] for r in records(directory,0,100)]
            if not subjects:continue
            per[directory]={'pairs':len(subjects)};total+=len(subjects)
            for v in VARIANTS:
                counts=[len(sources[v].get(identity(s,v),set())) for s in subjects]
                per[directory][v]={'seen_any':sum(n>=1 for n in counts),'seen_2plus':sum(n>=2 for n in counts),'seen_3plus':sum(n>=3 for n in counts)}
        row={'fold':fold,'train_benchmarks':split['train_benchmarks'],'heldout_benchmarks':split['development_benchmarks'],
             'pairs':total,'by_benchmark':per,'coverage':{v:{k:sum(d[v][k] for d in per.values()) for k in ['seen_any','seen_2plus','seen_3plus']} for v in VARIANTS},
             'source_identity_benchmark_counts':{v:dict(Counter(len(dirs) for dirs in table.values())) for v,table in sources.items()}}
        out.append(row)
    result={'scope':'Input-only potential overlap; no target outcomes used; source response availability not yet verified', 'folds':out,
            'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['mlp/audit/audit_subject_overlap.py','../research_loop/R15_protocol.md']}}
    (root/'input_overlap.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['fold','pairs','coverage']} for r in out],indent=2))


if __name__=='__main__':main()
