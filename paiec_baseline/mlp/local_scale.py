"""Support-only local correction scale. No evaluation labels or cross-call state."""
import numpy as np
from mlp.run.run_metadata_round import prediction


def smoother(support):
    n = len(support)
    x = support - support.mean(0)
    k = x @ x.T
    return np.ones((n,n))/n + np.linalg.solve(k+np.eye(n), k)


def loo_components(hat, labels):
    y = np.asarray(labels, dtype=float); n = len(y)
    assert n >= 3
    fitted = hat @ y
    standard = y - (y-fitted)/(1-np.diag(hat))
    mean_minus = (y.sum()-y)/(n-1)
    base_minus = (1+y.sum()-y)/(n+1)
    direction = np.clip(standard-mean_minus+base_minus,0,1)-base_minus
    return base_minus, direction


def coefficient(hat, labels, tau):
    if tau is None or len(labels) < 3:
        return 1.
    base, d = loo_components(hat, labels)
    return float(np.clip((d@(labels-base)+tau)/(d@d+tau),0,4))


def predict(support, query, labels, tau):
    y = np.asarray(labels, dtype=float)
    p = prediction(support,query,y,1.)
    c = coefficient(smoother(support),y,tau) if len(y)>=3 and tau is not None else 1.
    base = (1+y.sum())/(2+len(y))
    return np.clip(base+c*(p-base),0,1)
