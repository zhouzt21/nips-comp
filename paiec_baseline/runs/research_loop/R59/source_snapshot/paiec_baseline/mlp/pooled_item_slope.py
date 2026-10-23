"""Shared within-group-centered ridge from current evidence only."""
import numpy as np
from .peer_item_predictor import means


def prepare(context):
    groups={}
    for r in context:groups.setdefault(r['key'],[]).append(r)
    result={}
    for key,rows in groups.items():
        y=np.concatenate([np.asarray(r['labels'],float) for r in rows])
        x=np.concatenate([np.asarray(r['vectors'],float) for r in rows])
        center=x.mean(0) if len(y) else np.zeros(x.shape[1])
        result[key]=(center,x-center,y-y.mean() if len(y) else y)
    return result


def predict_many(target_key,query,context,ridge,pooled=True):
    rate=means(context).get(target_key,(.5,.5))[1]
    if ridge is None:return np.full(len(query),rate)
    groups=prepare(context)
    if target_key not in groups:raise ValueError('target group absent')
    chosen=list(groups.values()) if pooled else [groups[target_key]]
    x=np.concatenate([g[1] for g in chosen]);y=np.concatenate([g[2] for g in chosen])
    if not len(y):return np.full(len(query),rate)
    beta=x.T@np.linalg.solve(x@x.T+ridge*np.eye(len(y)),y)
    return np.clip(rate+(np.asarray(query)-groups[target_key][0])@beta,0,1)
