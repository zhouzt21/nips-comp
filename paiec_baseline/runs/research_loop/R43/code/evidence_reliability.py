"""Complement-invariant current-evidence summaries and NumPy MLP correction."""
import numpy as np
from .local_scale import loo_components

FEATURES=['log_n','base_variance','support_energy','effective_rank','top_fraction','effective_df',
          'coefficient_norm','loo_brier','loo_base_brier','loo_direction_energy','loo_gain',
          'abs_delta','delta_base_alignment','query_energy','influence_energy','posterior_variance']
QUERY_START=11


def statistics(support,query,labels):
    y=np.asarray(labels,float);x=np.asarray(support,float);q=np.asarray(query,float);n=len(y)
    if n<3 or len(x)!=n or not np.isin(y,[0,1]).all():raise ValueError('at least three aligned binary support labels required')
    b=(1+y.sum())/(2+n);center=x.mean(0);x=x-center;q=q-center;k=x@x.T
    inverse=np.linalg.inv(k+np.eye(n));a=inverse@(y-y.mean());cross=q@x.T;influence=cross@inverse;delta=cross@a
    eigen=np.maximum(np.linalg.eigvalsh(k),0);trace=eigen.sum();df=float(np.sum(eigen/(eigen+1)))
    hat=np.ones((n,n))/n+k@inverse;loob,lood=loo_components(hat,y)
    base_error=float(np.mean((loob-y)**2));loo_error=float(np.mean((loob+lood-y)**2))
    common=np.array([np.log2(n)/5,4*b*(1-b),trace/n,trace*trace/max(np.sum(eigen*eigen),1e-20)/n,
                     eigen[-1]/max(trace,1e-20),df/n,float(a@k@a),loo_error,base_error,float(np.mean(lood**2)),base_error-loo_error])
    energy=np.sum(q*q,axis=1);variance=np.maximum(energy-np.sum(cross*influence,axis=1),0)
    query_features=np.column_stack([np.abs(delta),delta*(2*b-1),energy,np.sum(influence*influence,axis=1),variance])
    z=np.column_stack([np.broadcast_to(common,(len(q),len(common))),query_features])
    assert np.isfinite(z).all()
    return z,float(b),delta


def forward(z,artifact):
    x=np.array(z,float,copy=True)
    if bool(artifact['episode_only']):x[:,QUERY_START:]=0
    x=np.clip((x-artifact['mean'])/artifact['scale'],-5,5)
    x=np.maximum(x@artifact['W1']+artifact['b1'],0);x=np.maximum(x@artifact['W2']+artifact['b2'],0)
    output=(x@artifact['W3']+artifact['b3']).reshape(-1)+np.log(1/3)
    return 4/(1+np.exp(-np.clip(output,-60,60)))


def predict(support,query,labels,artifact):
    if len(labels)<3:return np.full(len(query),(1+sum(labels))/(2+len(labels)))
    z,b,d=statistics(support,query,labels)
    return np.clip(b+forward(z,artifact)*d,0,1)
