import numpy as np
from mlp.run_complementarity_pilot import identity


def test_exact_brier_dividend_with_repeated_responses():
    p=np.array([.2,.7,.8]);q=np.array([.6,.5,.9]);y=np.array([0.,1.,0.])
    repeat=np.array([0,0,0,1,2,2]);score,div=identity(p[repeat],q[repeat],y[repeat])
    assert div>=0
    expected=.5*np.mean((p[repeat]-y[repeat])**2)+.5*np.mean((q[repeat]-y[repeat])**2)-div
    np.testing.assert_allclose(score,expected,atol=1e-12)
