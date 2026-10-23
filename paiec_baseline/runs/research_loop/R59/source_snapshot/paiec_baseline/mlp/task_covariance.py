"""Sign-invariant source task geometry; all slopes still fitted from current labels."""
import numpy as np
from .evaluate import BUDGETS
from .local_ridge import kernel_predict


def source_slope(row, rng=None):
    x=row['support']-row['support'].mean(0);y=row['a'].copy()
    if rng is not None:y=y[rng.permutation(len(y))]
    return x.T@np.linalg.solve(x@x.T+4*np.eye(len(y)),y-y.mean())


def fit_covariance(groups, seed=None):
    rng=np.random.default_rng(seed) if seed is not None else None
    matrices=[];pairs=0
    for rows in groups:
        if not rows:continue
        slopes=np.stack([source_slope(r,rng) for r in rows]);pairs+=len(rows)
        matrices.append(slopes.T@slopes/len(rows))
    matrix=np.mean(matrices,axis=0);val,vec=np.linalg.eigh(matrix);val=np.maximum(val[-16:],0);vec=vec[:,-16:]
    scale=np.sqrt(384*val/val.sum()) if val.sum()>0 else np.zeros(16)
    return vec*scale,{'benchmarks':len(matrices),'pairs':pairs,'top_eigenvalues':val.tolist(),'total_trace':float(np.trace(matrix))}


def score(row, projection, rho, ridge):
    x=row['support'];q=row['query'];xp=x@projection;qp=q@projection
    k=(1-rho)*(x@x.T)+rho*(xp@xp.T);c=(1-rho)*(q@x.T)+rho*(qp@xp.T)
    return np.array([np.mean((kernel_predict(k,c,row['a'][:n],ridge,2)-row['y'])**2) for n in BUDGETS])
