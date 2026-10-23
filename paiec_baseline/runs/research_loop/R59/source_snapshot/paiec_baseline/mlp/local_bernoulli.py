"""Current-support logistic slope with curvature-matched ridge and fixed smoothed offset."""
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.,-np.asarray(x,float)))


def fit_predict(support,query,labels,max_iter=30,tolerance=1e-8):
    y=np.asarray(labels,float);n=len(y);base=(1+y.sum())/(2+n)
    if n<2 or np.ptp(y)==0:
        p=np.full(len(query),base)
        return p,p,{'iterations':0,'converged':True,'objective_history':[],'gradient_inf':0.}
    center=support.mean(0);x=support-center;q=query-center;k=x@x.T;cross=q@x.T
    offset=np.log(base)-np.log1p(-base);penalty=base*(1-base)
    alpha=np.zeros(n);identity=np.eye(n)
    def objective(a):
        z=offset+k@a
        return float(np.sum(np.logaddexp(0,z)-y*z)+.5*penalty*(a@k@a))
    history=[objective(alpha)];first=None;converged=False;gradient=np.inf
    for iteration in range(max_iter):
        probability=sigmoid(offset+k@alpha)
        residual=probability-y+penalty*alpha
        gradient=float(np.max(np.abs(x.T@residual)))
        if gradient<tolerance:
            converged=True;break
        weight=probability*(1-probability)
        step=np.linalg.solve(weight[:,None]*k+penalty*identity,residual)
        if first is None:first=sigmoid(offset-cross@step)
        multiplier=1.
        for _ in range(30):
            trial=alpha-multiplier*step;value=objective(trial)
            if value<=history[-1]+1e-12:break
            multiplier*=.5
        else:raise RuntimeError('Bernoulli Newton backtracking failed')
        change=float(np.max(np.abs(multiplier*(k@step))))
        alpha=trial;history.append(value)
        if change<tolerance:
            converged=True;break
    final=sigmoid(offset+cross@alpha)
    probability=sigmoid(offset+k@alpha)
    gradient=float(np.max(np.abs(x.T@(probability-y+penalty*alpha))))
    if first is None:first=final.copy()
    return final,first,{'iterations':len(history)-1,'converged':bool(converged or gradient<tolerance),
                       'objective_history':history,'gradient_inf':gradient}
