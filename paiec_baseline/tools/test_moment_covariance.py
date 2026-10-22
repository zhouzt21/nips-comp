import numpy as np
from mlp.moment_covariance import predict,matrices
from mlp.kernel_covariance import predict_kernel


def test_baseline_old_formula_and_psd_diagonal():
    rng=np.random.default_rng(4401);x=rng.normal(size=(15,6));x/=np.linalg.norm(x,axis=1,keepdims=True);q=x[:4];k=x@x.T;c=q@x.T;y=np.arange(15)%2
    for n in [0,1,3,7,15]:
        np.testing.assert_allclose(predict(k,c,y[:n],0,True),(1+y[:n].sum())/(2+n),atol=1e-12)
        np.testing.assert_allclose(predict(k,c,y[:n],.25),predict_kernel(k,c,y[:n],.25),atol=1e-12)
    matrix,_=matrices(k,c,15,1,True)
    np.testing.assert_allclose(np.diag(matrix),3,atol=1e-12)
    assert np.linalg.eigvalsh(matrix).min()>=1-1e-12


def test_single_label_closed_form_and_complement():
    k=np.ones((1,1));c=np.array([[1.],[0.]])
    expected=.5+(1+np.exp(-(1-c[:,0])/.25))*.5/3
    np.testing.assert_allclose(predict(k,c,[1],1,True),expected,atol=1e-12)
    np.testing.assert_allclose(predict(k,c,[0],1,True),1-expected,atol=1e-12)
