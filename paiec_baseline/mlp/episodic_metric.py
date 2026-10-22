"""A NumPy-exportable diagonal metric learned through local ridge episodes."""
import numpy as np
from .local_ridge import kernel_predict


def probabilities(x, labels, log_weight, budget):
    z=np.asarray(x,np.float64)*np.exp(np.asarray(log_weight,np.float64)/2)
    z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    support=z[:31]
    return kernel_predict(support@support.T,z[31:]@support.T,np.asarray(labels)[:budget],1.)


def torch_probabilities(x, labels, log_weight, budget):
    import torch
    n=budget
    y=labels[:n]
    local=(1+y.sum())/(2+n)
    if n<2:return local.expand(len(x)-31)
    z=x*torch.exp(log_weight/2)
    z=z/torch.linalg.vector_norm(z,dim=1,keepdim=True).clamp_min(1e-12)
    a=z[:n]; q=z[31:]; center=a.mean(0)
    a=a-center; q=q-center
    beta=torch.linalg.solve(a@a.T+torch.eye(n,device=x.device),y-y.mean())
    return (local+(q@a.T)@beta).clamp(0,1)
