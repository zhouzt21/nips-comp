"""Isolated-pair linear covariance predictor with explicit variance accounting."""
import numpy as np


def matrices(k,c,n,eta,fixed_diagonal=False,uniform=False):
    if not 0<=eta<2:raise ValueError('eta must preserve positive independent noise')
    r=np.exp(-(1-np.clip(np.asarray(k)[:n,:n],-1,1))/.25);rq=np.exp(-(1-np.clip(np.asarray(c)[:,:n],-1,1))/.25)
    if uniform:
        rho=float((r.sum()-np.trace(r))/(n*(n-1))) if n>1 else 0.
        r=np.full((n,n),rho);np.fill_diagonal(r,1.);rq=np.full_like(rq,rho)
    return np.ones((n,n))+eta*r+(2-eta if fixed_diagonal else 2)*np.eye(n),1+eta*rq


def predict(k,c,labels,eta,fixed_diagonal=False,uniform=False):
    y=np.asarray(labels,float);n=len(y)
    if not np.isin(y,[0,1]).all():raise ValueError('binary labels required')
    if n==0:return np.full(len(c),.5)
    matrix,cross=matrices(k,c,n,eta,fixed_diagonal,uniform)
    return np.clip(.5+cross@np.linalg.solve(matrix,y-.5),0,1)
