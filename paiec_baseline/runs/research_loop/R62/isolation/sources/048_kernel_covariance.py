"""Constant plus RBF local kernel; eta=0 reproduces smoothed mean 2."""
import numpy as np


def predict_kernel(k,c,labels,eta):
    y=np.asarray(labels,float);n=len(y)
    if not n:return np.full(len(c),.5)
    if eta<0 or not np.isin(y,[0,1]).all():raise ValueError("invalid eta or labels")
    r=np.exp(-(1-np.clip(np.asarray(k)[:n,:n],-1,1))/.25)
    rq=np.exp(-(1-np.clip(np.asarray(c)[:,:n],-1,1))/.25)
    matrix=np.ones((n,n))+eta*r+2*np.eye(n)
    cross=1+eta*rq
    return np.clip(.5+cross@np.linalg.solve(matrix,y-.5),0,1)
