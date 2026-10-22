"""Exact small-cohort prevalence partitions with uniform individual marginal priors."""
from functools import lru_cache
import math
import numpy as np


@lru_cache(maxsize=8)
def partitions(n):
    if n<1 or n>4:raise ValueError('research scope requires 1 to 4 visible groups')
    rows=[(0,)]
    for _ in range(1,n):rows=[r+(k,) for r in rows for k in range(max(r)+2)]
    return tuple(tuple(tuple(i for i,v in enumerate(r) if v==k) for k in range(max(r)+1)) for r in rows)


@lru_cache(maxsize=64)
def prior(n,alpha):
    if alpha<=0:raise ValueError('positive concentration required')
    rows=partitions(n)
    logs=np.array([len(p)*math.log(alpha)+math.lgamma(alpha)-math.lgamma(alpha+n)+sum(math.lgamma(len(b)) for b in p) for p in rows])
    probability=np.exp(logs);np.testing.assert_allclose(probability.sum(),1.,atol=1e-12)
    return rows,logs


@lru_cache(maxsize=65536)
def log_beta_counts(s,n):
    return math.lgamma(1+s)+math.lgamma(1+n-s)-math.lgamma(2+n)


def predict_counts(counts,alpha,update_partition=True):
    counts=np.asarray(counts,float)
    if counts.ndim!=2 or counts.shape[1]!=2 or np.any(counts[:,0]<0) or np.any(counts[:,0]>counts[:,1]) or not np.all(counts==np.floor(counts)):
        raise ValueError('integer (successes,trials) required')
    g=len(counts)
    if not 1<=g<=4:raise ValueError('research scope requires 1 to 4 visible groups')
    if alpha is None:return (1+counts[:,0])/(2+counts[:,1]),1.
    rows,logprior=prior(g,float(alpha));logw=logprior.copy();values=np.empty((len(rows),g));singletons=[]
    for index,partition in enumerate(rows):
        singletons.append(all(len(block)==1 for block in partition))
        for block in partition:
            s,n=counts[list(block)].sum(0)
            if update_partition:logw[index]+=log_beta_counts(int(s),int(n))
            values[index,list(block)]=(1+s)/(2+n)
    weights=np.exp(logw-logw.max());weights/=weights.sum()
    return weights@values,float(weights[np.asarray(singletons)].sum())
