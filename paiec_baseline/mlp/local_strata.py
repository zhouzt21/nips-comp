"""Input-only binary strata with exact pooled/independent Bernoulli model averaging."""
from math import lgamma
import numpy as np


def partition(support, query, direction=None):
    x=np.asarray(support,float);q=np.asarray(query,float)
    if len(x)<2:return None
    center=x.mean(0);xc=x-center
    if direction is None:
        values,vectors=np.linalg.eigh(xc@xc.T)
        top=float(values[-1]);second=float(values[-2])
        if top<=1e-14 or top-second<=1e-8*top:return None
        direction=xc.T@vectors[:,-1]/np.sqrt(top)
        if direction[np.argmax(np.abs(direction))]<0:direction=-direction
    direction=np.asarray(direction,float)
    a=xc@direction;b=(q-center)@direction;threshold=float(np.median(a))
    # Equal feature values receive the same group, regardless of support order.
    tolerance=1e-12*max(float(np.max(np.abs(a))),1e-12)
    lower=a[a<threshold-tolerance];upper=a[a>threshold+tolerance]
    tie_upper=False
    if np.any(np.abs(a-threshold)<=tolerance):
        if not len(lower) or not len(upper):return None
        dl=threshold-lower.mean();du=upper.mean()-threshold
        if abs(dl-du)<=tolerance:return None
        tie_upper=du<dl
    g=(a>threshold+tolerance)|((np.abs(a-threshold)<=tolerance)&tie_upper)
    h=(b>threshold+tolerance)|((np.abs(b-threshold)<=tolerance)&tie_upper)
    if not g.any() or g.all():return None
    return g,h


def count_predict(groups,query_groups,labels):
    y=np.asarray(labels,float);g=np.asarray(groups,bool);q=np.asarray(query_groups,bool)
    if not np.isin(y,[0,1]).all():raise ValueError('binary labels required')
    if len(g)!=len(y) or not g.any() or g.all():raise ValueError('two nonempty groups required')
    def marginal(a):
        s=float(a.sum());f=len(a)-s
        return lgamma(s+1)+lgamma(f+1)-lgamma(s+f+2)
    base=(1+y.sum())/(2+len(y));means=np.array([(1+y[g==j].sum())/(2+sum(g==j)) for j in [False,True]])
    log_odds=sum(marginal(y[g==j]) for j in [False,True])-marginal(y)
    weight=1/(1+np.exp(-np.clip(log_odds,-700,700)))
    return (1-weight)*base+weight*means[q.astype(int)],float(weight)


def predict(support,query,labels,direction=None):
    y=np.asarray(labels,float)
    if len(support)!=len(y) or not np.isin(y,[0,1]).all():raise ValueError('support/label mismatch or nonbinary label')
    groups=partition(support,query,direction)
    if groups is None:return np.full(len(query),(1+y.sum())/(2+len(y))),{'degenerate':True,'split_weight':0.}
    p,w=count_predict(*groups,y)
    return p,{'degenerate':False,'split_weight':w}
