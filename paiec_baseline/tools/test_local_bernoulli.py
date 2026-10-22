import numpy as np
from mlp.local_bernoulli import fit_predict,sigmoid


def test_first_newton_step_matches_original_linear_tangent():
    rng=np.random.default_rng(3901);x=rng.normal(size=(15,17));q=rng.normal(size=(9,17));y=rng.integers(0,2,15)
    _,first,_=fit_predict(x,q,y);b=(1+y.sum())/(2+len(y));xc=x-x.mean(0)
    beta=xc.T@np.linalg.solve(xc@xc.T+np.eye(len(y)),y-y.mean())
    expected=sigmoid(np.log(b/(1-b))+(q-x.mean(0))@beta/(b*(1-b)))
    np.testing.assert_allclose(first,expected,atol=1e-11,rtol=0)


def test_support_objective_decreases_and_primal_gradient_converges():
    rng=np.random.default_rng(3902)
    for n in [3,7,15,31]:
        x=rng.normal(size=(n,40))*.3;q=rng.normal(size=(8,40))*.3;y=np.arange(n)%2
        p,_,diag=fit_predict(x,q,y)
        assert diag['converged'] and diag['gradient_inf']<1e-7
        assert np.all(np.diff(diag['objective_history'])<=1e-10)
        assert np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))


def test_empty_constant_complement_and_no_state():
    rng=np.random.default_rng(3903);x=rng.normal(size=(7,12));q=rng.normal(size=(5,12));y=np.arange(7)%2
    zero=fit_predict(x[:0],q,y[:0])[0];p=fit_predict(x,q,y)[0]
    np.testing.assert_allclose(fit_predict(x,q,1-y)[0],1-p,atol=1e-10)
    np.testing.assert_array_equal(fit_predict(x[:0],q,y[:0])[0],zero)
    np.testing.assert_array_equal(zero,np.full(5,.5))
    np.testing.assert_array_equal(fit_predict(x,q,np.ones(7))[0],np.full(5,8/9))
