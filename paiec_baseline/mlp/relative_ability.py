"""Benchmark-centered source ability with explicit coverage support."""
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from mlp.audit.audit_subject_overlap import identity
from .model import _SUBJECT_FIELDS


def key(subject):
    value=identity(subject,'config')
    return json.dumps(value) if value is not None else None


def aggregate(directory):
    fingerprint=hashlib.sha256(directory.encode()+Path(__file__).read_bytes()).hexdigest()
    cache=Path('D:/paiec_research_cache/ability_counts')/(fingerprint+'.json')
    if cache.exists():return json.loads(cache.read_text())
    root=Path('D:/paiec_payload/data')/directory
    responses=pd.read_parquet(root/'response.parquet',columns=['subject_id','item_id','response'])
    responses=responses[responses['response'].isin([0,1])].dropna(subset=['subject_id','item_id'])
    counts=responses.groupby('subject_id').agg(n=('response','size'),s=('response','sum'),items=('item_id','nunique'))
    subjects=pd.read_parquet(root/'subjects.parquet',columns=['subject_id',*_SUBJECT_FIELDS]).drop_duplicates('subject_id').set_index('subject_id').to_dict('index')
    output=[]
    for sid,row in counts[counts['items']>=80].iterrows():
        if sid not in subjects:continue
        name=key(subjects[sid])
        if name is None:continue
        output.append({'benchmark':directory,'key':name,'n':int(row['n']),'s':int(row['s']),'items':int(row['items']),'q':float(row['s']/row['n'])})
    cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(output),encoding='utf-8')
    return output


def fit(rows,allowed):
    if any(r['benchmark'] not in allowed for r in rows):raise ValueError('nontraining benchmark in fit')
    names=sorted({r['key'] for r in rows});benchmarks=sorted({r['benchmark'] for r in rows})
    si={s:i for i,s in enumerate(names)};bi={s:i for i,s in enumerate(benchmarks)}
    s=np.array([si[r['key']] for r in rows]);b=np.array([bi[r['benchmark']] for r in rows]);q=np.array([r['q'] for r in rows])
    ns=np.bincount(s,minlength=len(names));nb=np.bincount(b,minlength=len(benchmarks));a=np.zeros(len(names));d=np.zeros(len(benchmarks))
    for iteration in range(2000):
        d_new=np.bincount(b,weights=q-a[s],minlength=len(benchmarks))/nb
        a_new=np.bincount(s,weights=q-d_new[b],minlength=len(names))/(ns+1.)
        change=max(np.max(abs(a_new-a)),np.max(abs(d_new-d)));a,d=a_new,d_new
        if change<1e-10:break
    gradient_a=np.bincount(s,weights=a[s]+d[b]-q,minlength=len(names))+a
    gradient_d=np.bincount(b,weights=a[s]+d[b]-q,minlength=len(benchmarks))
    max_grad=float(max(abs(gradient_a).max(),abs(gradient_d).max()))
    if max_grad>1e-6:raise RuntimeError(f'fit not converged: {max_grad}')
    support={name:len({r['benchmark'] for r in rows if r['key']==name}) for name in names}
    return {'train_benchmarks':sorted(allowed),'ability':{name:float(a[i]) for i,name in enumerate(names) if support[name]>=2},
            'source_benchmark_count':support,'nuisance_intercepts':dict(zip(benchmarks,d.tolist())),
            'training_pairs':len(rows),'iterations':iteration+1,'max_normal_equation_residual':max_grad,
            'training_pair_mse':float(np.mean((q-d[b]-a[s])**2)),'lambda':1.}
