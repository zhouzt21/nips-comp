import numpy as np
from mlp.pooled_item_slope import predict_many


def test_peer_correspondence_and_no_peers_equivalence():
    context=[{'key':'a','labels':np.array([0.,1.]),'vectors':np.array([[0.,0.],[1.,0.]])},
             {'key':'b','labels':np.array([0.,1.]),'vectors':np.array([[0.,0.],[1.,0.]])}]
    query=np.array([[1.,0.],[0.,0.]])
    np.testing.assert_allclose(predict_many('a',query,context[:1],1),predict_many('a',query,context[:1],1,False))
    shared=predict_many('a',query,context,1)
    scrambled=[context[0],{**context[1],'labels':np.array([1.,0.])}]
    assert shared[0]>predict_many('a',query,scrambled,1)[0]
    np.testing.assert_allclose(shared,predict_many('a',query,context[::-1],1),atol=1e-12)
    flipped=[{**r,'labels':1-r['labels']} for r in context]
    np.testing.assert_allclose(1-shared,predict_many('a',query,flipped,1),atol=1e-12)


def test_empty_budget_and_numerical_constant_features():
    query=np.ones((3,2));empty=[{'key':'a','labels':np.zeros(0),'vectors':np.zeros((0,2))}]
    np.testing.assert_array_equal(predict_many('a',query,empty,1),[.5]*3)
    constant=[{'key':'a','labels':np.array([1.,0.]),'vectors':np.ones((2,2))}]
    np.testing.assert_array_equal(predict_many('a',query,constant,1),[.5]*3)
