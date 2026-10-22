import numpy as np
from mlp.local_scale import smoother, loo_components, coefficient, predict
from mlp.run.run_metadata_round import prediction


def test_leave_one_out_matches_refitting_with_centering_and_smoothed_mean():
    rng = np.random.default_rng(3401)
    for n in [3,7,15,31]:
        x = rng.normal(size=(n,12)); y = rng.integers(0,2,n).astype(float)
        b,d = loo_components(smoother(x),y)
        brute = [prediction(np.delete(x,i,0),x[i:i+1],np.delete(y,i),1.)[0] for i in range(n)]
        np.testing.assert_allclose(b+d,brute,atol=1e-11,rtol=0)


def test_local_scale_has_no_state_or_labels_at_zero_budget():
    rng = np.random.default_rng(3402); x = rng.normal(size=(31,12)); q = rng.normal(size=(8,12)); y = rng.integers(0,2,31)
    empty = predict(x[:0],q,y[:0],.1)
    p = predict(x,q,y,.1)
    np.testing.assert_array_equal(predict(x[:0],q,y[:0],.1),empty)
    np.testing.assert_array_equal(empty,np.full(len(q),.5))
    np.testing.assert_array_equal(predict(x,q,y,.1),p)
    assert np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))


def test_infinite_shrinkage_and_constant_labels_restore_reference():
    rng = np.random.default_rng(3403); x = rng.normal(size=(15,12)); q = rng.normal(size=(7,12)); y = rng.integers(0,2,15)
    np.testing.assert_array_equal(predict(x,q,y,None),prediction(x,q,y,1.))
    for value in [0.,1.]:
        yy = np.full(15,value)
        np.testing.assert_allclose(coefficient(smoother(x),yy,.1),1.,atol=1e-11)
        np.testing.assert_allclose(predict(x,q,yy,.1),prediction(x,q,yy,1.),atol=1e-11)
