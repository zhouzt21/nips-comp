import numpy as np
from mlp.evidence_reliability import statistics,forward,predict
from mlp.run.run_metadata_round import prediction


def artifact():
    rng=np.random.default_rng(4300)
    return dict(mean=np.zeros(16),scale=np.ones(16),episode_only=False,W1=rng.normal(size=(16,16)),b1=np.zeros(16),
                W2=rng.normal(size=(16,8)),b2=np.zeros(8),W3=np.zeros((8,1)),b3=np.zeros(1))


def test_initial_gate_reproduces_original_and_complement():
    rng=np.random.default_rng(4301);x=rng.normal(size=(15,20))*.3;q=rng.normal(size=(8,20))*.3;y=np.arange(15)%2
    a=artifact();z,b,d=statistics(x,q,y);opposite,b2,d2=statistics(x,q,1-y)
    np.testing.assert_allclose(z,opposite,atol=1e-10)
    np.testing.assert_allclose(b2,1-b);np.testing.assert_allclose(d2,-d)
    np.testing.assert_allclose(predict(x,q,y,a),prediction(x,q,y,1.),atol=1e-10)
    np.testing.assert_allclose(forward(z,a),np.ones(len(q)))


def test_query_batch_prefix_and_history_independence():
    rng=np.random.default_rng(4302);x=rng.normal(size=(7,12));q=rng.normal(size=(9,12));y=np.arange(7)%2;a=artifact()
    a['W3'][:]=.03
    p=predict(x[:3],q,y[:3],a);predict(x,q,1-y,a)
    np.testing.assert_allclose(p,predict(x[:3],q,y[:3],a),atol=1e-12)
    np.testing.assert_allclose(p[:1],predict(x[:3],q[:1],y[:3],a),atol=1e-12)
    np.testing.assert_array_equal(predict(x[:0],q,y[:0],a),np.full(9,.5))
    np.testing.assert_allclose(predict(x[:3],q,1-y[:3],a),1-p,atol=1e-10)
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
