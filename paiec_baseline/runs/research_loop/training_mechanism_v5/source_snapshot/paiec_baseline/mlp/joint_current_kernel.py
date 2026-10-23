"""R23 Gaussian conditional mean for current subject and item evidence."""
import numpy as np


def predict_many(target_key,query,context,ridge):
    x=np.concatenate([r['vectors'] for r in context]);y=np.concatenate([r['labels'] for r in context])
    if not len(y):return np.full(len(query),.5)
    keys=np.concatenate([np.full(len(r['labels']),r['key'],dtype=object) for r in context])
    covariance=np.full((len(y),len(y)),.25/8)+(.25/4)*(keys[:,None]==keys[None,:])+.25*np.eye(len(y))
    cross=np.tile(.25/8+(.25/4)*(keys==target_key),(len(query),1))
    if ridge is not None:
        if ridge<=0:raise ValueError('ridge must be positive or None')
        center=x.mean(0);xc=x-center
        covariance+=(.25/ridge)*(xc@xc.T)
        cross+=(.25/ridge)*((np.asarray(query)-center)@xc.T)
    return np.clip(.5+cross@np.linalg.solve(covariance,y-.5),0,1)
