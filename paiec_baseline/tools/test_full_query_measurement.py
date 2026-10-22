import numpy as np
from mlp.audit_full_queries import brier


def test_count_aggregation_keeps_every_response_and_brier_identity():
    outcomes=[np.array([0,1]),np.array([1,1,0]),np.array([0])]
    m=np.array([len(y) for y in outcomes]);s=np.array([y.sum() for y in outcomes]);p=np.array([[.2,.8,.3],[.6,.1,.5]])
    expanded=np.stack([np.concatenate([np.full(len(y),v) for v,y in zip(row,outcomes)]) for row in p]);y=np.concatenate(outcomes)
    np.testing.assert_allclose(brier(p,m,s),((expanded-y)**2).mean(1),atol=1e-12)
    mean=s/m;noise=(s*(1-mean)).sum()/m.sum()
    np.testing.assert_allclose(brier(p,m,s),(((p-mean)**2)*m).sum(1)/m.sum()+noise,atol=1e-12)


def test_within_item_noise_cancels_between_predictors():
    # Equal item means but different response disagreement cannot explain score differences.
    m=np.array([4,8]);s=np.array([1,6]);p=np.array([.1,.7]);q=np.array([.5,.5]);means=s/m
    observed=brier(p,m,s)-brier(q,m,s)
    explained=np.sum(m*((p-means)**2-(q-means)**2))/m.sum()
    np.testing.assert_allclose(observed,explained,atol=1e-12)
