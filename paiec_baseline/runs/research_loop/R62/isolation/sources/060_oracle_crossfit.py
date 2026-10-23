"""Evaluation-label diagnostics ONLY. Never import in a submitted predictor."""
import numpy as np
from .oracle_scale import optimum_scale
from .measurement_episodes import weighted_brier


def fitted_predictions(base, direction, counts, successes, calibration, assessment, permutation=None):
    """Fit ONLY calibration outcomes; return predictions for assessment indices."""
    d=np.asarray(direction,float);m=np.asarray(counts,float);s=np.asarray(successes,float)
    calibration=np.asarray(calibration,int);assessment=np.asarray(assessment,int)
    assert len(calibration)>0 and len(assessment)>0
    assert not np.intersect1d(calibration,assessment).size
    dc=d[calibration] if permutation is None else d[calibration][permutation]
    _,coefficient=optimum_scale(base,dc,m[calibration],s[calibration],4.)
    p=np.clip(base+coefficient*d[assessment],0.,1.)
    mean=(1+s[calibration].sum())/(2+m[calibration].sum())
    return p,coefficient,mean


def crossfit(base, direction, counts, successes, order, permutation_seed):
    d=np.asarray(direction,float);m=np.asarray(counts,float);s=np.asarray(successes,float)
    order=np.asarray(order,int)
    assert sorted(order.tolist())==list(range(len(d)))
    halves=np.array_split(order,2);p=np.empty(len(d));permuted=p.copy();mean=p.copy();coefficients=[]
    rng=np.random.default_rng(permutation_seed)
    for calibration,assessment in [halves,halves[::-1]]:
        p[assessment],c,mean[assessment]=fitted_predictions(base,d,m,s,calibration,assessment)
        permuted[assessment],_,_=fitted_predictions(base,d,m,s,calibration,assessment,rng.permutation(len(calibration)))
        coefficients.append(c)
    return {'crossfit':weighted_brier(p,m,s),'permuted_calibration':weighted_brier(permuted,m,s),
            'extra_label_mean':weighted_brier(mean,m,s)},coefficients
