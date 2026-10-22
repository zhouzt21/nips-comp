"""Exact lifted-space group centering, without constructing tensor features."""
import numpy as np
from .peer_item_predictor import means


def predict_many(target_key,query,context,ridge,pooled=True,power=4):
    rate=means(context).get(target_key,(.5,.5))[1]
    if ridge is None:return np.full(len(query),rate)
    chosen=context if pooled else [r for r in context if r['key']==target_key]
    x=np.concatenate([r['vectors'] for r in chosen]);y=np.concatenate([r['labels'] for r in chosen])
    if not len(y):return np.full(len(query),rate)
    keys=np.concatenate([np.full(len(r['labels']),r['key'],dtype=object) for r in chosen]);groups=[np.flatnonzero(keys==k) for k in dict.fromkeys(keys)]
    p=np.eye(len(y))
    for idx in groups:p[np.ix_(idx,idx)]-=1/len(idx)
    target=np.flatnonzero(keys==target_key)
    if not len(target):raise ValueError('target support absent')
    kernel=(x@x.T)**power;centered=p@kernel@p
    cross=(np.asarray(query)@x.T)**power
    cross=(cross-kernel[target].mean(axis=0))@p
    return np.clip(rate+cross@np.linalg.solve(centered+ridge*np.eye(len(y)),p@y),0,1)
