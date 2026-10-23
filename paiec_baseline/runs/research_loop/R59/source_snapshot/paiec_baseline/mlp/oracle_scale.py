"""Evaluation-label diagnostic ONLY. Never import from a submitted predictor."""
import numpy as np


def optimum_scale(base, direction, counts, successes, upper=np.inf):
    """Exact piecewise quadratic minimum of clipped probability ray, c>=0.

    Item sufficient statistics preserve every response, including repeats.
    Returns (Brier, coefficient); coefficient uses evaluation outcomes.
    """
    d = np.asarray(direction, dtype=float)
    m = np.asarray(counts, dtype=float)
    s = np.asarray(successes, dtype=float)
    assert d.shape == m.shape == s.shape and m.sum() > 0 and upper >= 0
    def loss(c):
        p = np.clip(base + c*d, 0., 1.)
        return float(np.sum(m*p*p - 2*s*p + s)/m.sum())
    nonzero = d != 0
    if not np.any(nonzero) or upper == 0:
        return loss(0.), 0.
    z = d[nonzero]
    mm, ss = m[nonzero], s[nonzero]
    boundary = (z > 0).astype(float)
    times = (boundary-base)/z
    keep = times < upper
    order = np.argsort(times[keep])
    t = times[keep][order]
    aa = (mm*z*z)[keep][order]
    bb = (z*(mm*base-ss))[keep][order]
    cc = (mm*(boundary**2-base**2)-2*ss*(boundary-base))[keep][order]
    # Each segment includes its left boundary; coincident events create harmless zero-width segments.
    c0 = float(np.sum(m*base*base-2*s*base+s))
    left = np.r_[0., t]
    right = np.r_[t, upper]
    # Suffix sums avoid subtracting nearly equal curvature totals near saturation.
    a = np.r_[np.cumsum(aa[::-1])[::-1], 0.] + np.sum((mm*z*z)[~keep])
    b = np.r_[np.cumsum(bb[::-1])[::-1], 0.] + np.sum((z*(mm*base-ss))[~keep])
    c = c0+np.r_[0., np.cumsum(cc)]
    roots = np.divide(-b, a, out=left.copy(), where=a > 0)
    candidate = np.maximum(left, np.minimum(roots, right))
    scores = a*candidate*candidate + 2*b*candidate + c
    index = int(np.argmin(scores))
    # Check actual losses at the computed optimum and fixed references to guard rounding.
    choices = [0., min(1., upper), float(candidate[index])]
    best = min(((loss(v), v) for v in choices), key=lambda x: x[0])
    return best
