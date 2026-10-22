import numpy as np
from mlp.joint_current_kernel import predict_many


def test_noitem_analytic_mean_and_empty_state():
    empty=[{'key':'a','vectors':np.zeros((0,2)),'labels':np.zeros(0)}];q=np.ones((2,2))
    np.testing.assert_array_equal(predict_many('a',q,empty,4),[.5,.5])
    rows=[{'key':'a','vectors':np.zeros((3,2)),'labels':np.array([1.,1.,0.])}]
    strength=8/3;expected=(strength*.5+2)/(strength+3)
    np.testing.assert_allclose(predict_many('a',q,rows,None),[expected]*2,atol=1e-12)
    predict_many('a',q,rows,1)
    np.testing.assert_array_equal(predict_many('a',q,empty,4),[.5,.5])


def test_lowbudget_information_order_and_complement():
    rows=[{'key':k,'vectors':np.array([v]),'labels':np.array([y])} for k,v,y in [('a',[0.,1.],0.),('b',[1.,0.],1.),('c',[-1.,0.],0.)]]
    q=np.array([[1.,0.],[-1.,0.]])
    p=predict_many('a',q,rows,1)
    assert p[0]>p[1]
    np.testing.assert_allclose(p,predict_many('a',q,rows[::-1],1),atol=1e-12)
    flipped=[{**r,'labels':1-r['labels']} for r in rows]
    np.testing.assert_allclose(1-p,predict_many('a',q,flipped,1),atol=1e-12)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
