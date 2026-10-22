import numpy as np
from mlp.online_variance_acquisition import acquire,select_indices


def test_budget_forced_tail_and_random_replay():
    x=np.zeros((40,8))
    np.testing.assert_array_equal(select_indices(x,0),np.arange(31))
    selected=select_indices(x,999)
    assert len(selected)==31 and selected[0]==0 and np.all(np.diff(selected)>0)
    assert selected[-1]==39
    for n in [0,1,3,7,15,31]:assert set(selected[:n])<=set(selected)


def test_only_current_and_past_inputs_used():
    accepted=np.zeros((2,3));current=np.array([1.,0.,0.])
    assert acquire(current,accepted,10,3,.1)
    assert not acquire(np.zeros(3),accepted,10,3,.1)
    assert acquire(np.zeros(3),accepted,3,3,.1)
    assert not acquire(current,accepted,10,0,.1)
