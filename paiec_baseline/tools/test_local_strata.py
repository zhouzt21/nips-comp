import numpy as np
from mlp.local_strata import predict,count_predict,partition


def test_exact_sequence_likelihood():
    # M0 likelihood B(3,3)=1/30; M1 B(1,3)B(3,1)=1/9.
    y=np.array([0,0,1,1]);g=np.array([0,0,1,1],bool)
    p,w=count_predict(g,np.array([0,1],bool),y)
    np.testing.assert_allclose(w,10/13,atol=1e-14)
    np.testing.assert_allclose(p,(1-10/13)*.5+(10/13)*np.array([.25,.75]),atol=1e-14)


def test_input_order_scale_complement_and_batch_invariance():
    rng=np.random.default_rng(4002);x=rng.normal(size=(15,8));q=rng.normal(size=(9,8));y=np.arange(15)%2
    p=predict(x,q,y)[0];order=rng.permutation(15)
    np.testing.assert_allclose(predict(x[order],q,y[order])[0],p,atol=1e-12)
    np.testing.assert_allclose(predict(-3*x,-3*q,y)[0],p,atol=1e-12)
    np.testing.assert_allclose(predict(x,q,1-y)[0],1-p,atol=1e-12)
    np.testing.assert_allclose(predict(x,q[:1],y)[0],p[:1],atol=1e-12)
    predict(x[:3],q,1-y[:3])
    np.testing.assert_array_equal(predict(x,q,y)[0],p)


def test_degenerate_and_empty_inputs_do_not_split_arbitrarily():
    q=np.array([[0.,0.],[1.,1.]])
    for x,y in [(np.zeros((0,2)),np.zeros(0)),(np.ones((7,2)),np.arange(7)%2),
                (np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),np.array([0,0,1,1]))]:
        p,d=predict(x,q,y)
        assert d['degenerate']
        np.testing.assert_array_equal(p,np.full(len(q),(1+y.sum())/(2+len(y))))


def test_direction_sign_and_identical_inputs():
    x=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]]);q=x[[0,0,3]];y=np.array([0,0,1,1])
    p=predict(x,q,y,np.array([1.,0.]))[0]
    np.testing.assert_allclose(p,predict(x,q,y,np.array([-1.,0.]))[0])
    assert p[0]==p[1] and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
