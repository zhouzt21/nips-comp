"""Separate allowed current-group rates from current-item effects; pure NumPy."""
import numpy as np
from .pooled_item_slope import prepare
from .peer_item_predictor import means


def components(context, queries, pooled_ridge, own_ridge, original_ridge, rng, permutations=4):
    groups=prepare(context); rates=means(context); keys=list(groups)
    x=np.concatenate([groups[k][1] for k in keys]); y=np.concatenate([groups[k][2] for k in keys])
    inverse=np.linalg.inv(x@x.T+pooled_ridge*np.eye(len(y))) if len(y) and pooled_ridge is not None else None
    segments={};offset=0
    for key in keys: segments[key]=slice(offset,offset+len(groups[key][2]));offset+=len(groups[key][2])
    own={};original={}
    for key,(center,xx,yy) in groups.items():
        def beta(ridge):
            return xx.T@np.linalg.solve(xx@xx.T+ridge*np.eye(len(yy)),yy) if len(yy) and ridge is not None else np.zeros(xx.shape[1])
        own[key]=beta(own_ridge);original[key]=beta(original_ridge)
    out=[]
    for row,query in zip(context,queries):
        key=row['key'];q=query-groups[key][0];base,peer=rates[key]
        cross=(q@x.T)@inverse if inverse is not None else np.zeros((len(q),len(y)))
        shuffled=[]
        for _ in range(permutations):
            py=y.copy()
            for other in keys:
                if other!=key: py[segments[other]]=rng.permutation(py[segments[other]])
            shuffled.append(cross@py)
        out.append({'base':base,'peer':peer,'pooled':cross@y,'own':q@own[key],
                    'original':q@original[key],'permuted':shuffled})
    return out
