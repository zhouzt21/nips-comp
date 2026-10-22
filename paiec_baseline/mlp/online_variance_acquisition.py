"""Online decisions from current/past accepted features and remaining count only."""
import numpy as np


def acquire(current,accepted,remaining_items,remaining_labels,threshold):
    if remaining_labels<=0:return False
    if not len(accepted) or remaining_items<=remaining_labels or threshold<=0:return True
    support=np.asarray(accepted);center=support.mean(0);s=support-center;q=np.asarray(current)-center
    k=s@q;variance=float(q@q-k@np.linalg.solve(s@s.T+np.eye(len(s)),k))
    return max(variance,0.)>=threshold


def select_indices(candidates,threshold,budget=31):
    accepted=[]
    for i,current in enumerate(candidates):
        if len(accepted)==budget:break
        if acquire(current,candidates[accepted],len(candidates)-i,budget-len(accepted),threshold):accepted.append(i)
    if len(accepted)!=budget:raise ValueError('insufficient candidates')
    return np.asarray(accepted,dtype=int)
