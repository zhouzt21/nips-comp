import numpy as np
from mlp.task_covariance import source_slope,fit_covariance,score
from mlp.local_ridge import kernel_predict
from mlp.evaluate import BUDGETS


def test_source_covariance_invariant_to_pair_label_complement():
    rng=np.random.default_rng(22);row={'support':rng.normal(size=(31,384)),'a':rng.integers(0,2,31).astype(float)}
    rev={**row,'a':1-row['a']};np.testing.assert_allclose(source_slope(row),-source_slope(rev),atol=1e-12)
    u,_=fit_covariance([[row]]);v,_=fit_covariance([[rev]])
    np.testing.assert_allclose(u@u.T,v@v.T,atol=1e-10)


def test_zero_covariance_mix_replays_original_local_prediction():
    rng=np.random.default_rng(23);x=rng.normal(size=(31,384));q=rng.normal(size=(20,384))
    row={'support':x,'query':q,'a':rng.integers(0,2,31).astype(float),'y':rng.integers(0,2,20).astype(float)}
    p=rng.normal(size=(384,16));expected=[np.mean((kernel_predict(x@x.T,q@x.T,row['a'][:n],4,2)-row['y'])**2) for n in BUDGETS]
    np.testing.assert_allclose(score(row,p,0,4),expected,atol=1e-12)
