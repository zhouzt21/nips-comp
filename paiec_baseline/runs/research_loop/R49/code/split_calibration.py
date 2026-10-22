"""R49 support-only direction/calibration splitting; no scored outcomes accepted."""
import numpy as np
from .run_metadata_round import prediction
# Shared numerical solver, not the diagnostic runner or its fitted coefficients.
# In this experimental module it receives ONLY current allowed support outcomes.
from .oracle_scale import optimum_scale


def predictions(support,query,labels):
    y=np.asarray(labels,float);n=len(y)
    original=prediction(support,query,y,1.)
    if n<7:
        return {'split_calibrated':original.copy(),'split_fixed':original.copy(),'permuted_calibration':original.copy(),'R25':original},[]
    rng=np.random.default_rng(4901+n);actual=[];fixed=[];shuffled=[];coefficients=[]
    for _ in range(4):
        halves=np.array_split(rng.permutation(n),2)
        for fit,cal in [halves,halves[::-1]]:
            base=(1+y[fit].sum())/(2+len(fit))
            # The same fitted head evaluates its calibration panel and queries.
            p=prediction(support[fit],np.concatenate([support[cal],query]),y[fit],1.)
            dc=p[:len(cal)]-base;dq=p[len(cal):]-base
            _,c=optimum_scale(base,dc,np.ones(len(cal)),y[cal],4.)
            _,cperm=optimum_scale(base,dc[rng.permutation(len(cal))],np.ones(len(cal)),y[cal],4.)
            actual.append(np.clip(base+c*dq,0,1));fixed.append(p[len(cal):]);shuffled.append(np.clip(base+cperm*dq,0,1));coefficients.append(c)
    return {'split_calibrated':np.mean(actual,axis=0),'split_fixed':np.mean(fixed,axis=0),
            'permuted_calibration':np.mean(shuffled,axis=0),'R25':original},coefficients
