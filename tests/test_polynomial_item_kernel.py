import numpy as np
from mlp.polynomial_item_kernel import predict_many
from mlp.pooled_item_slope import predict_many as linear


def test_power1_identity_and_power4_explicit_lift():
    rng=np.random.default_rng(9)
    context=[{'key':k,'labels':rng.integers(2,size=n).astype(float),'vectors':rng.normal(size=(n,2))} for k,n in [('a',3),('b',4),('a',2)]]
    query=rng.normal(size=(5,2))
    def lift(x):return np.array([np.outer(np.outer(v,v).ravel(),np.outer(v,v).ravel()).ravel() for v in x])
    expanded=[{**r,'vectors':lift(r['vectors'])} for r in context]
    for pooled in [True,False]:
        np.testing.assert_allclose(predict_many('a',query,context,4,pooled,1),linear('a',query,context,4,pooled),atol=1e-12)
        np.testing.assert_allclose(predict_many('a',query,context,4,pooled,4),linear('a',lift(query),expanded,4,pooled),atol=1e-11)


def test_empty_and_complement():
    q=np.ones((2,2));empty=[{'key':'a','labels':np.zeros(0),'vectors':np.zeros((0,2))}]
    np.testing.assert_array_equal(predict_many('a',q,empty,1),[.5,.5])
    rows=[{'key':'a','labels':np.array([1.,0.]),'vectors':np.array([[1.,0.],[0.,1.]])}]
    flipped=[{**r,'labels':1-r['labels']} for r in rows]
    np.testing.assert_allclose(1-predict_many('a',q,rows,4),predict_many('a',q,flipped,4),atol=1e-12)
