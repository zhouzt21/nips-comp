import numpy as np
from mlp.oracle_scale import optimum_scale


def test_amplification_can_remove_underconfidence_beyond_convex_family():
    capped, c1 = optimum_scale(.5, [.2, -.2], [1, 1], [1, 0], 1.)
    full, c2 = optimum_scale(.5, [.2, -.2], [1, 1], [1, 0])
    np.testing.assert_allclose([capped, c1, full, c2], [.09, 1., 0., 2.5], atol=1e-12)


def test_oracle_keeps_repeated_response_weights_and_zero_direction():
    args = (.4, np.array([.2, -.15]), np.array([7, 2]), np.array([3, 1]))
    score, coefficient = optimum_scale(*args)
    prediction = np.repeat(np.clip(args[0]+coefficient*args[1], 0, 1), args[2])
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(score, np.mean((prediction-y)**2), atol=1e-12)
    np.testing.assert_allclose(optimum_scale(.5, [0, 0], [3, 8], [2, 1]), [.25, 0])


def test_unbounded_ray_is_invariant_to_direction_units():
    m = np.array([4, 3, 7]); s = np.array([3, 1, 6]); d = np.array([.2, -.1, .3])
    loss, c = optimum_scale(.5, d, m, s)
    tiny_loss, tiny_c = optimum_scale(.5, d*1e-20, m, s)
    np.testing.assert_allclose(tiny_loss, loss, atol=1e-12)
    np.testing.assert_allclose(tiny_c*1e-20, c, atol=1e-10)


def test_piecewise_solver_is_at_least_as_good_as_dense_grid():
    rng = np.random.default_rng(3301)
    for _ in range(30):
        base = rng.uniform(.05, .95)
        p = rng.uniform(0, 1, 12)
        m = rng.integers(1, 15, 12)
        s = rng.binomial(m, .5)
        loss, c = optimum_scale(base, p-base, m, s)
        grid = np.r_[np.linspace(0, 2, 2001), np.geomspace(2, 10000, 2000)]
        pred = np.clip(base+grid[:, None]*(p-base), 0, 1)
        dense = ((m*pred*pred-2*s*pred+s).sum(1)/m.sum()).min()
        assert loss <= dense+1e-10 and c >= 0
