import numpy as np
from mlp.split_calibration import predictions
from mlp.run.run_metadata_round import prediction


def test_exact_small_budget_fallback():
    rng=np.random.default_rng(49);s=rng.normal(size=(3,7));q=rng.normal(size=(4,7));y=np.array([1,0,1.])
    for n in [0,1,3]:
        out,c=predictions(s[:n],q,y[:n]);assert c==[]
        for p in out.values():np.testing.assert_array_equal(p,prediction(s[:n],q,y[:n],1.))


def test_calibration_complement_and_query_independence():
    rng=np.random.default_rng(491);s=rng.normal(size=(15,7));q=rng.normal(size=(8,7));y=rng.integers(0,2,15)
    out,c=predictions(s,q,y);reverse,d=predictions(s,q,1-y)
    np.testing.assert_allclose(c,d,atol=1e-10)
    for k,p in out.items():
        np.testing.assert_allclose(p,1-reverse[k],atol=1e-10)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    _,c2=predictions(s,q*7,y);np.testing.assert_allclose(c,c2,atol=1e-12)
    again,c3=predictions(s,q,y);np.testing.assert_array_equal(c,c3)
    for k in out:np.testing.assert_array_equal(out[k],again[k])
